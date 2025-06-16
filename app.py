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

def initialize_components():
    """Lazy-load heavy components"""
    if "components_initialized" not in st.session_state:
        with st.spinner("Loading AI components..."):
            try:
                from sentence_transformers import SentenceTransformer
                st.session_state.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                st.session_state.components_initialized = True
            except ImportError as e:
                st.error(f"Failed to load AI components: {str(e)}")
                st.stop()

def process_file(file):
    """Handle file upload and validation"""
    if file.size > Config.MAX_FILE_SIZE:
        st.error(f"File exceeds {Config.MAX_FILE_SIZE/1024/1024}MB limit")
        return None
    
    try:
        if file.type == "text/csv":
            return pd.read_csv(file)
        return pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def generate_xml(mapped_data: List[Dict], root_element: str) -> str:
    """Generate XML from mapped data"""
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
    
    # API Selection
    selected_api = st.sidebar.selectbox(
        "Select HRP API",
        list(Config.XML_SCHEMAS.keys())
    )
    
    # File Upload
    st.header("1. Data Upload")
    file = st.file_uploader(
        "Upload Provider Data",
        type=Config.SUPPORTED_FILE_TYPES
    )
    
    if file:
        df = process_file(file)
        if df is not None:
            initialize_components()
            
            # Field Mapping
            with st.expander("Field Mapping", expanded=True):
                sample_fields = st.multiselect(
                    "Select fields to map",
                    df.columns.tolist(),
                    default=df.columns[:5].tolist()
                )
                
                if st.button("Generate Mappings"):
                    with st.spinner("Creating mappings..."):
                        st.session_state.mappings = [{
                            "source": field,
                            "target": field.upper().replace(" ", "_"),
                            "value": "",
                            "confidence": 0.9
                        } for field in (sample_fields or df.columns)]
            
            # Results
            if "mappings" in st.session_state:
                st.header("2. Results")
                st.dataframe(pd.DataFrame(st.session_state.mappings))
                
                if st.button("Generate XML"):
                    with st.spinner("Generating XML..."):
                        xml_outputs = []
                        for _, row in df.iterrows():
                            mapped_data = [{**m, "value": row[m["source"]]} 
                                        for m in st.session_state.mappings]
                            xml_outputs.append(generate_xml(
                                mapped_data, 
                                Config.XML_SCHEMAS[selected_api]
                            ))
                        
                        # Create ZIP
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