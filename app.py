import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
import hashlib
import json
import logging
from typing import Dict, List, Tuple, Optional
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import redis
import zipfile
import io
import re

# --- Configuration ---
class Config:
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FILE_TYPES = ["csv", "xlsx"]
    XML_SCHEMAS = {
        "FacilityLoad": "schemas/facility_load.xsd",
        "PractitionerLoad": "schemas/practitioner_load.xsd",
        "MemberEnrollment": "schemas/member_enrollment.xsd"
    }
    COMPLIANCE_RULES = {
        "HIPAA": ["PHI_Fields.json", "Encryption_Standards.json"],
        "CMS": ["Field_Definitions.json", "Code_Sets.json"]
    }
    REDIS_CACHE_TTL = 3600  # 1 hour

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Cache Setup ---
redis_client = redis.Redis(
    host='redis-master',
    port=6379,
    decode_responses=True
)

# --- LLM & RAG Initialization ---
class AIModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            device_map="auto",
            load_in_4bit=True
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_db = self._init_knowledge_base()

    def _init_knowledge_base(self):
        """Load compliance docs into vector store"""
        from langchain.document_loaders import DirectoryLoader
        
        loader = DirectoryLoader('./compliance_docs/', glob="**/*.pdf")
        docs = loader.load()
        return FAISS.from_documents(docs[:1000], self.embeddings)  # Limit for demo

    def generate_mapping(self, source_field: str, api_schema: str) -> Dict:
        """Generate field mapping with RAG context"""
        cache_key = f"mapping:{hashlib.md5(source_field.encode()).hexdigest()}"
        cached = redis_client.get(cache_key)
        
        if cached:
            return json.loads(cached)

        # Retrieve relevant compliance context
        rag_results = self.vector_db.similarity_search(source_field, k=3)
        context = "\n".join([doc.page_content for doc in rag_results])

        prompt = f"""
        Healthcare Data Mapping Task:
        Source Field: {source_field}
        Target API: {api_schema}
        
        Compliance Context:
        {context}
        
        Return JSON with:
        - target_field: str
        - transformation: str (python code)
        - confidence: float (0-1)
        - compliance_checks: list[str]
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        result = json.loads(self.tokenizer.decode(outputs[0]))
        
        redis_client.setex(cache_key, Config.REDIS_CACHE_TTL, json.dumps(result))
        return result

# --- Core Processing Engine ---
class MappingEngine:
    def __init__(self):
        self.ai = AIModel()
        self.code_tables = self._load_code_tables()

    def _load_code_tables(self):
        """Load standard code translations"""
        return {
            "Gender": {"M": "Male", "F": "Female", "U": "Unknown"},
            "ProviderType": {"MD": "Physician", "NP": "Nurse Practitioner"},
            # ... other code tables
        }

    def process_field(self, source_field: str, source_value: str, api_schema: str) -> Dict:
        """Process a single field through mapping pipeline"""
        mapping = self.ai.generate_mapping(source_field, api_schema)
        
        # Apply transformation logic
        try:
            if mapping["transformation"] == "direct":
                value = source_value
            elif mapping["transformation"].startswith("code:"):
                code_table = mapping["transformation"].split(":")[1]
                value = self.code_tables[code_table].get(source_value, "UNKNOWN")
            else:
                # Execute dynamic Python logic in sandbox
                value = self._safe_eval(mapping["transformation"], {"value": source_value})
        except Exception as e:
            value = f"ERROR: {str(e)}"
            mapping["confidence"] = 0.0
        
        return {
            "source": source_field,
            "target": mapping["target_field"],
            "value": value,
            "confidence": mapping["confidence"],
            "compliance": mapping.get("compliance_checks", [])
        }

    def _safe_eval(self, code: str, context: Dict):
        """Safely execute transformation logic"""
        allowed_names = {'value', 'datetime', 'pd', 'np'}
        code = compile(code, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of {name} not allowed")
        return eval(code, {"__builtins__": {}}, context)

# --- UI Components ---
def api_selection_sidebar():
    st.sidebar.header("API Configuration")
    api = st.sidebar.selectbox(
        "Select HRP API",
        list(Config.XML_SCHEMAS.keys()),
        help="Choose the target API specification"
    )
    
    st.sidebar.checkbox("Enable HIPAA Validation", True)
    st.sidebar.checkbox("Enable CMS Compliance", True)
    return api

def file_upload_section():
    st.header("1. Data Upload")
    file = st.file_uploader(
        "Upload Provider Data",
        type=Config.SUPPORTED_FILE_TYPES,
        help="CSV or Excel file with source data"
    )
    
    if file:
        if file.size > Config.MAX_FILE_SIZE:
            st.error(f"File exceeds {Config.MAX_FILE_SIZE/1024/1024}MB limit")
            return None
        
        try:
            df = pd.read_csv(file) if file.type == "text/csv" else pd.read_excel(file)
            st.success(f"Loaded {len(df)} records with {len(df.columns)} fields")
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            logger.exception("File load error")
    return None

# --- XML Generation ---
def generate_xml(mapped_data: List[Dict], api_schema: str) -> str:
    """Generate validated XML output"""
    root = ET.Element(api_schema)
    
    for field in mapped_data:
        if field["confidence"] > 0.6:  # Confidence threshold
            elem = ET.SubElement(root, field["target"])
            elem.text = str(field["value"])
    
    # Validate against schema
    if not validate_xml(ET.tostring(root), Config.XML_SCHEMAS[api_schema]):
        st.error("XML validation failed - check compliance rules")
        logger.error("XML schema validation failed")
    
    return prettify_xml(root)

def validate_xml(xml_str: str, schema_path: str) -> bool:
    """Placeholder for real schema validation"""
    # In production, use lxml or similar for XSD validation
    return True

def prettify_xml(element) -> str:
    """Format XML with proper indentation"""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# --- Audit Reporting ---
def generate_audit_report(mappings: List[Dict]) -> str:
    """Generate markdown format audit report"""
    report = ["# Data Mapping Audit Report", f"Generated: {datetime.now()}"]
    
    for m in mappings:
        report.append(f"""
        ## {m['source']} → {m['target']}
        - **Value:** `{m['value']}`
        - **Confidence:** {m['confidence']:.0%}
        - **Compliance Checks:** {', '.join(m['compliance'])}
        """)
    
    return "\n".join(report)

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Healthcare Data Mapper",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Intelligent Healthcare Data Mapper")
    st.write("""
    AI-powered field mapping with compliance validation for HRP API integrations
    """)
    
    # Initialize services
    if 'engine' not in st.session_state:
        st.session_state.engine = MappingEngine()
    
    # UI Sections
    selected_api = api_selection_sidebar()
    df = file_upload_section()
    
    if df is not None:
        with st.expander("Field Mapping Configuration", expanded=True):
            st.write("### Automatic Field Mappings")
            
            # Process sample of fields for demo
            sample_fields = st.multiselect(
                "Select fields to map (or leave blank for all)",
                df.columns.tolist(),
                default=df.columns[:5].tolist()
            )
            
            if st.button("Generate Mappings"):
                with st.spinner("Processing with AI..."):
                    mappings = []
                    for field in (sample_fields if sample_fields else df.columns):
                        # Use first non-null value as example
                        sample_value = df[field].dropna().iloc[0] if not df[field].isnull().all() else ""
                        result = st.session_state.engine.process_field(field, sample_value, selected_api)
                        mappings.append(result)
                    
                    st.session_state.mappings = mappings
                
        if 'mappings' in st.session_state:
            st.write("### Mapping Results")
            st.dataframe(pd.DataFrame(st.session_state.mappings))
            
            # XML Generation
            if st.button("Generate XML"):
                with st.spinner("Building XML payloads..."):
                    xml_outputs = []
                    for _, row in df.iterrows():
                        mapped_data = []
                        for m in st.session_state.mappings:
                            mapped_data.append({
                                **m,
                                "value": row[m["source"]]
                            })
                        xml_outputs.append(generate_xml(mapped_data, selected_api))
                    
                    st.session_state.xml_outputs = xml_outputs
            
            if 'xml_outputs' in st.session_state:
                st.success(f"Generated {len(st.session_state.xml_outputs)} XML records")
                
                # Create downloadable ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a") as zf:
                    for i, xml in enumerate(st.session_state.xml_outputs):
                        zf.writestr(f"record_{i+1}.xml", xml)
                    
                    # Add audit report
                    report = generate_audit_report(st.session_state.mappings)
                    zf.writestr("mapping_audit.md", report)
                
                st.download_button(
                    label="Download All Files (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="hrp_mapping_output.zip",
                    mime="application/zip"
                )
                
                # Show sample
                with st.expander("Sample XML Output"):
                    st.code(st.session_state.xml_outputs[0], language="xml")

if __name__ == "__main__":
    main()
