import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom
import zipfile
import io
import logging
from typing import List, Dict
import time
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuration ---
class Config:
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    SUPPORTED_FILE_TYPES = ["csv", "xlsx"]
    XML_SCHEMAS = {
        "FacilityLoad": "Facility",
        "PractitionerLoad": "Practitioner",
        "MemberEnrollment": "Member"
    }
    SCHEMA_FIELDS = {
        "FacilityLoad": ["FACILITY_ID", "FACILITY_NAME", "ADDRESS", "CITY", "STATE"],
        "PractitionerLoad": ["PRACTITIONER_ID", "FIRST_NAME", "LAST_NAME", "SPECIALTY"],
        "MemberEnrollment": ["MEMBER_ID", "ENROLLMENT_DATE", "PLAN_TYPE", "MEMBER_NAME"]
    }
    PROCESSING_TIMEOUT = 10  # Seconds

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def safe_import_components():
    """Handle imports with fallbacks"""
    try:
        from sentence_transformers import SentenceTransformer
        from langchain_community.embeddings import HuggingFaceEmbeddings
        # Initialize models with explicit device='cpu' to avoid device issues
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        return embedder, embeddings, None
    except (ImportError, NotImplementedError, RuntimeError) as e:
        logger.warning(f"AI components not available: {str(e)}")
        return None, None, f"AI features disabled: {str(e)}"

def initialize_rag(reference_dfs: List[pd.DataFrame], embeddings):
    """Initialize RAG with reference documents"""
    if not reference_dfs or embeddings is None:
        return None
    try:
        # Convert DataFrames to LangChain documents
        documents = []
        for df in reference_dfs:
            loader = DataFrameLoader(df, page_content_column=df.columns[0])
            documents.extend(loader.load())
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=None)
        
        # Define prompt template
        prompt_template = """
        You are an expert in mapping healthcare data fields to XML structures for HealthRules Payor APIs.
        Given the input field "{input_field}" and the target XML schema fields {schema_fields}, suggest:
        1. Whether to map (Y/N)
        2. The target XML field (if mapped)
        3. Mapping logic (e.g., direct, rename, combine, date format)
        4. Reasoning for the mapping
        Use the provided reference documents for context.
        Answer in JSON format:
        ```json
        {
            "map": "Y/N",
            "xml_field": "string",
            "logic": "string",
            "reasoning": "string"
        }
        ```
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input_field", "schema_fields"]
        )
        
        # Mock RAG response (replace with xAI Grok API in production)
        qa_chain = RetrievalQA.from_chain_type(
            llm=None,  # No LLM for MVP; rely on mock response
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_chain
    except Exception as e:
        logger.error(f"RAG initialization failed: {str(e)}")
        return None

def map_fields_with_ai(embedder, qa_chain, source_fields: List[str], target_fields: List[str]) -> List[Dict]:
    """Map source fields to target fields using AI embeddings or RAG"""
    mappings = []
    for i, field in enumerate(source_fields):
        try:
            if qa_chain:
                try:
                    # Mock RAG response
                    result = {
                        "result": f'{{"map": "Y", "xml_field": "{target_fields[0]}", "logic": "Direct mapping", "reasoning": "Input field matches reference document naming convention"}}'
                    }
                    mapping = eval(result["result"])
                except Exception as e:
                    logger.error(f"RAG mapping failed for {field}: {str(e)}")
                    mapping = {
                        "map": "N",
                        "xml_field": "",
                        "logic": "",
                        "reasoning": "RAG mapping failed"
                    }
            elif embedder:
                source_emb = embedder.encode([field])
                target_emb = embedder.encode(target_fields)
                similarities = np.dot(source_emb, target_emb.T)[0]
                best_match_idx = np.argmax(similarities)
                mapping = {
                    "map": "Y",
                    "xml_field": target_fields[best_match_idx],
                    "logic": "Semantic similarity",
                    "reasoning": f"Matched based on embedding similarity ({similarities[best_match_idx]:.2f})"
                }
            else:
                mapping = {
                    "map": "Y",
                    "xml_field": field.upper().replace(" ", "_"),
                    "logic": "Direct mapping",
                    "reasoning": "Default mapping due to missing AI components"
                }
            
            mappings.append({
                "si": i + 1,
                "input_field": field,
                "map": mapping["map"],
                "xml_field": mapping["xml_field"],
                "logic": mapping["logic"],
                "comments": mapping["reasoning"]
            })
        except Exception as e:
            logger.error(f"Mapping failed for {field}: {str(e)}")
            mappings.append({
                "si": i + 1,
                "input_field": field,
                "map": "N",
                "xml_field": "",
                "logic": "",
                "comments": f"Mapping error: {str(e)}"
            })
    return mappings

def process_file(file, is_reference=False):
    """Handle file upload safely"""
    if file.size > Config.MAX_FILE_SIZE:
        st.error(f"File exceeds {Config.MAX_FILE_SIZE/1024/1024}MB limit")
        return None
    
    try:
        if file.type == "text/csv":
            return pd.read_csv(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            return pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def generate_xml(mapped_data: List[Dict], row: pd.Series, root_element: str, valid_fields: List[str]) -> str:
    """Generate XML with validation"""
    root = ET.Element(root_element)
    for field in mapped_data:
        if field["map"] == "Y" and field["xml_field"] in valid_fields:
            try:
                elem = ET.SubElement(root, field["xml_field"])
                value = str(row[field["input_field"]])
                elem.text = value.encode('utf-8', 'ignore').decode('utf-8')
            except Exception as e:
                logger.error(f"Invalid XML element {field['xml_field']}: {str(e)}")
                continue
    xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml()
    # Basic validation
    try:
        ET.fromstring(xml_str)
    except ET.ParseError as e:
        logger.error(f"Invalid XML generated: {str(e)}")
        return None
    return xml_str

def main():
    st.set_page_config(
        page_title="Smart Data Mapper",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Smart Data Mapper")
    
    # Initialize components
    embedder, embeddings, ai_error = safe_import_components()
    if ai_error:
        st.warning(ai_error)
    
    # API Selection
    st.sidebar.header("API Payload Selection")
    selected_api = st.sidebar.selectbox(
        "Select HRP API",
        list(Config.XML_SCHEMAS.keys())
    )
    
    # Reference Document Upload
    st.header("1. Upload Reference Documents")
    reference_files = st.file_uploader(
        "Upload reference documents (optional)",
        type=Config.SUPPORTED_FILE_TYPES,
        accept_multiple_files=True
    )
    reference_dfs = [process_file(f, is_reference=True) for f in reference_files] if reference_files else []
    reference_dfs = [df for df in reference_dfs if df is not None]
    
    # Initialize RAG if references provided
    qa_chain = initialize_rag(reference_dfs, embeddings) if reference_dfs and embeddings else None
    
    # Client Input File Upload
    st.header("2. Upload Client Input File")
    input_file = st.file_uploader(
        "Upload client data (CSV/Excel)",
        type=Config.SUPPORTED_FILE_TYPES
    )
    
    if input_file:
        df = process_file(input_file)
        if df is not None:
            # Display headers and sample rows
            st.subheader("Input File Preview")
            st.write("Headers:", df.columns.tolist())
            st.dataframe(df.head(5))
            
            # Field Mapping
            with st.expander("Field Mapping", expanded=True):
                fields = st.multiselect(
                    "Select fields to map",
                    df.columns.tolist(),
                    default=df.columns[:5].tolist()
                )
                
                if st.button("Generate Mappings"):
                    start_time = time.time()
                    st.session_state.mappings = map_fields_with_ai(
                        embedder,
                        qa_chain,
                        fields or df.columns,
                        Config.SCHEMA_FIELDS[selected_api]
                    )
                    elapsed_time = time.time() - start_time
                    if elapsed_time > Config.PROCESSING_TIMEOUT:
                        st.warning(f"Mapping took {elapsed_time:.2f}s, exceeding {Config.PROCESSING_TIMEOUT}s target")
                    st.success("Mappings generated successfully")
                
                if st.button("Reset Mappings"):
                    if "mappings" in st.session_state:
                        del st.session_state.mappings
                        st.success("Mappings cleared")
            
            # Mapping Editor and Outputs
            if "mappings" in st.session_state:
                st.header("3. Review and Edit Mappings")
                edited_mappings = st.data_editor(
                    pd.DataFrame(st.session_state.mappings),
                    column_config={
                        "si": st.column_config.NumberColumn("SI#", disabled=True),
                        "input_field": st.column_config.TextColumn("Input Field", disabled=True),
                        "map": st.column_config.SelectboxColumn("Map to XML (Y/N)", options=["Y", "N"], required=True),
                        "xml_field": st.column_config.SelectboxColumn(
                            "XML Field",
                            options=[""] + Config.SCHEMA_FIELDS[selected_api],
                            required=False
                        ),
                        "logic": st.column_config.TextColumn("Mapping Logic"),
                        "comments": st.column_config.TextColumn("Comments")
                    },
                    use_container_width=True
                )
                st.session_state.mappings = edited_mappings.to_dict('records')
                
                # Generate Outputs
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate Mapper Sheet"):
                        mapper_df = pd.DataFrame(st.session_state.mappings)
                        buffer = io.BytesIO()
                        mapper_df.to_excel(buffer, index=False, engine='openpyxl')
                        st.download_button(
                            "Download Mapper Sheet",
                            buffer.getvalue(),
                            file_name=f"mapper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                with col2:
                    if st.button("Generate XML"):
                        with st.spinner("Creating XML files..."):
                            try:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                xml_outputs = []
                                start_time = time.time()
                                for _, row in df.iterrows():
                                    xml_str = generate_xml(
                                        st.session_state.mappings,
                                        row,
                                        Config.XML_SCHEMAS[selected_api],
                                        Config.SCHEMA_FIELDS[selected_api]
                                    )
                                    if xml_str:
                                        xml_outputs.append(xml_str)
                                
                                elapsed_time = time.time() - start_time
                                if elapsed_time > Config.PROCESSING_TIMEOUT:
                                    st.warning(f"XML generation took {elapsed_time:.2f}s, exceeding {Config.PROCESSING_TIMEOUT}s target")
                                
                                # Create ZIP package
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, "a") as zf:
                                    for i, xml in enumerate(xml_outputs):
                                        zf.writestr(f"record_{timestamp}_{i+1}.xml", xml)
                                
                                st.download_button(
                                    "Download XMLs",
                                    data=zip_buffer.getvalue(),
                                    file_name=f"mappings_{timestamp}.zip",
                                    mime="application/zip"
                                )
                                st.success("XML files generated successfully")
                            except Exception as e:
                                st.error(f"Error generating XML/ZIP: {str(e)}")
                                logger.error(f"XML/ZIP generation failed: {str(e)}")

if __name__ == "__main__":
    main()
