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
    SUPPORTED_FILE_TYPES = ["csv", "xlsx"]
    XML_SCHEMAS = {
        "FacilityLoad": "Facility",
        "PractitionerLoad": "Practitioner",
        "MemberEnrollment": "Member"
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

def process_file(file):
    """Handle file upload safely"""
    if file.size > Config.MAX_FILE_SIZE:
        st.error(f"File exceeds {Config.MAX_FILE_SIZE/1024/1024}MB limit")
        return None
    
    try:
        return pd.read_csv(file) if file.type == "text/csv" else pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def generate_xml(mapped_data: List[Dict], root_element: str) -> str:
    """Generate XML with validation"""
    root = ET.Element(root_element)
    for field in mapped_data:
        if field.get('confidence', 0) > 0.5:
            elem = ET.SubElement(root, field['target'])
            elem.text = str(field['value'])
    return minidom.parseString(ET.tostring(root)).toprettyxml()

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
                    st.session_state.mappings = [{
                        "source": field,
                        "target": field.upper().replace(" ", "_"),
                        "value": "",
                        "confidence": 0.9
                    } for field in fields or df.columns]
            
            # XML Generation
            if "mappings" in st.session_state:
                st.header("2. Results")
                st.dataframe(pd.DataFrame(st.session_state.mappings))
                
                if st.button("Generate XML"):
                    with st.spinner("Creating XML files..."):
                        xml_outputs = []
                        for _, row in df.iterrows():
                            mapped_data = [{**m, "value": row[m["source"]]} 
                                         for m in st.session_state.mappings]
                            xml_outputs.append(generate_xml(
                                mapped_data, 
                                Config.XML_SCHEMAS[selected_api]
                            ))
                        
                        # Create download package
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "a") as zf:
                            for i, xml in enumerate(xml_outputs):
                                zf.writestr(f"record_{i+1}.xml", xml)
                        
                        st.download_button(
                            "Download XMLs",
                            data=zip_buffer.getvalue(),
                            file_name="mappings.zip",
                            mime="application/zip"
                        )

if __name__ == "__main__":
    main()