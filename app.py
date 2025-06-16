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

# --- Configuration ---
class Config:
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls"]
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
        return SentenceTransformer('all-MiniLM-L6-v2'), None
    except ImportError as e:
        logger.warning(f"AI components not available: {str(e)}")
        return None, "AI features disabled"

def map_fields_with_ai(embedder, source_fields: List[str], target_fields: List[str]) -> List[Dict]:
    """Map source fields to target fields using AI embeddings or fallback to simple mapping"""
    if embedder is None:
        return [{"source": f, "target": f.upper().replace(" ", "_"), "confidence": 0.9} for f in source_fields]
    try:
        source_emb = embedder.encode(source_fields)
        target_emb = embedder.encode(target_fields)
        similarities = np.dot(source_emb, target_emb.T)
        mappings = []
        for i, source in enumerate(source_fields):
            best_match_idx = np.argmax(similarities[i])
            mappings.append({
                "source": source,
                "target": target_fields[best_match_idx],
                "confidence": float(similarities[i][best_match_idx])
            })
        return mappings
    except Exception as e:
        logger.error(f"AI mapping failed: {str(e)}")
        return [{"source": f, "target": f.upper().replace(" ", "_"), "confidence": 0.9} for f in source_fields]

def process_file(file):
    """Handle file upload safely"""
    if file.size > Config.MAX_FILE_SIZE:
        st.error(f"File exceeds {Config.MAX_FILE_SIZE/1024/1024}MB limit")
        return None
    
    try:
        if file.type == "text/csv":
            return pd.read_csv(file)
        elif file.type in [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel"
        ]:
            try:
                return pd.read_excel(file, engine='openpyxl')
            except:
                return pd.read_excel(file, engine='xlrd')
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def generate_xml(mapped_data: List[Dict], root_element: str, valid_fields: List[str]) -> str:
    """Generate XML with validation"""
    root = ET.Element(root_element)
    for field in mapped_data:
        if field.get('confidence', 0) > 0.5 and field['target'] in valid_fields:
            try:
                elem = ET.SubElement(root, field['target'])
                elem.text = str(field['value']).encode('utf-8', 'ignore').decode('utf-8')
            except ValueError as e:
                logger.error(f"Invalid XML element {field['target']}: {str(e)}")
                continue
    return minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml()

def main():
    st.set_page_config(
        page_title="Healthcare Data Mapper",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Healthcare Data Mapper")
    
    # Initialize with error handling
    embedder, ai_error = safe_import_components()
    if ai_error:
        st.warning(ai_error)
    
    # API Selection
    selected_api = st.sidebar.selectbox(
        "Select HRP API",
        list(Config.XML_SCHEMAS.keys())
    )
    
    # File Processing
    st.header("1. Data Upload")
    file = st.file_uploader(
        "Upload Provider Data",
        type=Config.SUPPORTED_FILE_TYPES
    )
    
    if file:
        df = process_file(file)
        if df is not None:
            # Field Mapping
            with st.expander("Field Mapping", expanded=True):
                fields = st.multiselect(
                    "Select fields to map",
                    df.columns.tolist(),
                    default=df.columns[:5].tolist()
                )
                
                if st.button("Generate Mappings"):
                    st.session_state.mappings = map_fields_with_ai(
                        embedder,
                        fields or df.columns,
                        Config.SCHEMA_FIELDS[selected_api]
                    )
                    st.success("Mappings generated successfully")
                
                if st.button("Reset Mappings"):
                    if "mappings" in st.session_state:
                        del st.session_state.mappings
                        st.success("Mappings cleared")
            
            # Mapping Editor and XML Generation
            if "mappings" in st.session_state:
                st.header("2. Mappings")
                edited_mappings = st.data_editor(
                    pd.DataFrame(st.session_state.mappings),
                    column_config={
                        "source": st.column_config.TextColumn("Source Field", disabled=True),
                        "target": st.column_config.SelectboxColumn(
                            "Target Field",
                            options=Config.SCHEMA_FIELDS[selected_api],
                            required=True
                        ),
                        "confidence": st.column_config.NumberColumn(
                            "Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            disabled=(embedder is None)
                        )
                    },
                    use_container_width=True
                )
                st.session_state.mappings = edited_mappings.to_dict('records')
                
                if st.button("Generate XML"):
                    with st.spinner("Creating XML files..."):
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            xml_outputs = []
                            for _, row in df.iterrows():
                                mapped_data = [{**m, "value": row[m["source"]]} 
                                             for m in st.session_state.mappings]
                                xml_outputs.append(generate_xml(
                                    mapped_data,
                                    Config.XML_SCHEMAS[selected_api],
                                    Config.SCHEMA_FIELDS[selected_api]
                                ))
                            
                            # Create download package
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