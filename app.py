# ✅ Refactored app.py with RAG-based mapping + HRP production-grade XML output

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import requests
import logging
from datetime import datetime
import io
import time
import xml.etree.ElementTree as ET

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom Styling ---
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; }
    .main-header p { font-size: 1.2rem; margin: 0; }
    .xml-box { background-color: #f5f5f5; padding: 1rem; border-radius: 10px; margin-top: 1rem; font-family: monospace; white-space: pre-wrap; }
    </style>
    """, unsafe_allow_html=True)

# --- XML Generator ---
def build_hrp_xml(row: pd.Series) -> str:
    provider = ET.Element("Provider")
    ET.SubElement(provider, "ProviderID").text = str(row.get("provider_id", ""))
    ET.SubElement(provider, "ProviderName").text = str(row.get("provider_name", ""))
    ET.SubElement(provider, "NPI").text = str(row.get("npi", ""))
    ET.SubElement(provider, "TaxID").text = str(row.get("tax_id", ""))
    address = ET.SubElement(provider, "Address")
    ET.SubElement(address, "Street").text = str(row.get("address_line1", ""))
    ET.SubElement(address, "City").text = str(row.get("city", ""))
    ET.SubElement(address, "State").text = str(row.get("state", ""))
    ET.SubElement(address, "ZipCode").text = str(row.get("zip", ""))
    contact = ET.SubElement(provider, "Contact")
    ET.SubElement(contact, "Phone").text = str(row.get("phone", ""))
    ET.SubElement(contact, "Email").text = str(row.get("email", ""))
    return ET.tostring(provider, encoding="unicode")

# --- Main App ---
def main():
    st.set_page_config(page_title="HRP Smart Data Mapper", layout="wide")
    load_custom_css()

    st.markdown("""
        <div class="main-header">
            <h1>🧠 HRP Smart Data Mapper</h1>
            <p>Upload RAG mapping + provider input file → get mapped XML</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📁 Step 1: Upload RAG Mapping File (CSV with 'fields' and 'xml field')")
    rag_file = st.file_uploader("RAG Mapping File", type=["csv"], key="rag")

    st.markdown("### 📄 Step 2: Upload Provider Input File")
    prov_file = st.file_uploader("Provider Input File", type=["csv", "xlsx"], key="provider")

    if not rag_file or not prov_file:
        st.info("Please upload both files to continue.")
        st.stop()

    try:
        rag_df = pd.read_csv(rag_file)
        if prov_file.name.endswith(".csv"):
            prov_df = pd.read_csv(prov_file)
        else:
            prov_df = pd.read_excel(prov_file)
    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.stop()

    st.success("✅ Files loaded successfully")
    st.markdown("---")
    st.markdown("### 🧠 Field Mapping Based on RAG")

    rag_df.columns = rag_df.columns.str.strip().str.lower()
    prov_df.columns = prov_df.columns.str.strip()

    field_map = {}
    for _, row in rag_df.iterrows():
        src = row.get("fields")
        tgt = row.get("xml field")
        if pd.notna(src) and pd.notna(tgt):
            field_map[str(src).strip()] = str(tgt).strip()

    mapped_data = []
    for _, row in prov_df.iterrows():
        xml_row = {}
        for col in prov_df.columns:
            if col in field_map:
                xml_row[field_map[col]] = row[col]
        mapped_data.append(xml_row)

    st.markdown("### 🔍 Sample Transformed XML Output")
    sample_df = pd.DataFrame(mapped_data)
    for i in range(min(3, len(sample_df))):
        xml = build_hrp_xml(sample_df.iloc[i])
        st.markdown(f"#### Provider {i+1}")
        st.markdown(f"<div class='xml-box'>{xml}</div>", unsafe_allow_html=True)

    full_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<Providers>\n' + '\n'.join([build_hrp_xml(pd.Series(r)) for r in mapped_data]) + '\n</Providers>'
    st.download_button("📥 Download Full Mapped XML", full_xml, file_name="mapped_providers.xml", mime="application/xml")

if __name__ == "__main__":
    main()
