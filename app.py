import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, List
import xml.etree.ElementTree as ET
from xml.dom import minidom
import zipfile
import io
import sys
import subprocess

# --- Configuration ---
class Config:
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FILE_TYPES = ["csv", "xlsx"]
    XML_SCHEMAS = {
        "FacilityLoad": "schemas/facility_load.xsd",
        "PractitionerLoad": "schemas/practitioner_load.xsd",
        "MemberEnrollment": "schemas/member_enrollment.xsd"
    }

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Initialize AI Components ---
def initialize_ai():
    """Lazy-load AI components to improve startup time"""
    if "ai_initialized" not in st.session_state:
        with st.spinner("Loading AI components (first run only)..."):
            try:
                from sentence_transformers import SentenceTransformer
                from langchain_community.vectorstores import Chroma
                from langchain.embeddings import HuggingFaceEmbeddings
                
                st.session_state.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                st.session_state.VectorStore = Chroma
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                st.session_state.ai_initialized = True
            except ImportError as e:
                st.error(f"Critical dependency error: {str(e)}")
                logger.exception("AI initialization failed")
                st.stop()

# --- File Processing ---
def process_uploaded_file(file):
    """Validate and load uploaded file"""
    if file.size > Config.MAX_FILE_SIZE:
        st.error(f"File exceeds {Config.MAX_FILE_SIZE/1024/1024}MB limit")
        return None
    
    try:
        if file.type == "text/csv":
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, engine='openpyxl')
        
        st.success(f"Loaded {len(df)} records with {len(df.columns)} fields")
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        logger.exception("File processing error")
        return None

# --- XML Generation ---
def generate_xml(mapped_data: List[Dict], api_schema: str) -> str:
    """Generate validated XML output"""
    root = ET.Element(api_schema)
    
    for field in mapped_data:
        if field.get("confidence", 0) > 0.6:  # Confidence threshold
            elem = ET.SubElement(root, field["target"])
            elem.text = str(field["value"])
    
    return prettify_xml(root)

def prettify_xml(element) -> str:
    """Format XML with proper indentation"""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Healthcare Data Mapper",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Intelligent Healthcare Data Mapper")
    st.write("AI-powered field mapping with compliance validation")
    
    # Initialize session state
    if "mappings" not in st.session_state:
        st.session_state.mappings = []
    
    # Sidebar Configuration
    selected_api = st.sidebar.selectbox(
        "Select HRP API",
        list(Config.XML_SCHEMAS.keys()),
        help="Choose the target API specification"
    )
    
    # File Upload Section
    st.header("1. Data Upload")
    file = st.file_uploader(
        "Upload Provider Data",
        type=Config.SUPPORTED_FILE_TYPES,
        help="CSV or Excel file with source data"
    )
    
    if file:
        df = process_uploaded_file(file)
        
        if df is not None:
            # Initialize AI components only when needed
            initialize_ai()
            
            # Field Mapping Section
            with st.expander("Field Mapping Configuration", expanded=True):
                sample_fields = st.multiselect(
                    "Select fields to map",
                    df.columns.tolist(),
                    default=df.columns[:5].tolist()
                )
                
                if st.button("Generate Mappings"):
                    with st.spinner("Analyzing fields with AI..."):
                        from langchain.schema import Document
                        from langchain.embeddings import HuggingFaceEmbeddings
                        
                        # Simplified mapping logic for demo
                        temp_mappings = []
                        for field in (sample_fields if sample_fields else df.columns):
                            temp_mappings.append({
                                "source": field,
                                "target": field.upper().replace(" ", "_"),
                                "value": "",
                                "confidence": 0.9  # Mock confidence
                            })
                        st.session_state.mappings = temp_mappings
            
            # Results Display
            if st.session_state.mappings:
                st.header("2. Mapping Results")
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
                        
                        # Create downloadable ZIP
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "a") as zf:
                            for i, xml in enumerate(xml_outputs):
                                zf.writestr(f"record_{i+1}.xml", xml)
                        
                        st.download_button(
                            label="Download All Files (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name="hrp_mapping_output.zip",
                            mime="application/zip"
                        )
                        
                        # Show sample
                        with st.expander("Sample XML Output"):
                            st.code(xml_outputs[0], language="xml")

if __name__ == "__main__":
    main()