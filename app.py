# ✅ Refactored app.py with XML generation and production-grade HRP support

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
    st.set_page_config(page_title="HRP Provider XML Generator", layout="wide")
    load_custom_css()

    st.markdown("""
        <div class="main-header">
            <h1>🩺 HRP Provider XML Generator</h1>
            <p>Generate production-ready XML for HealthRules Payor</p>
        </div>
    """, unsafe_allow_html=True)

    st.info("Upload provider data to generate standardized HRP XML output. Required fields: provider_id, provider_name, npi, tax_id, address_line1, city, state, zip, phone, email")

    uploaded = st.file_uploader("Upload Provider Excel File", type=["xlsx", "csv"])
    if not uploaded:
        st.stop()

    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.success(f"✅ Loaded {len(df)} records from {uploaded.name}")
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()

    # Preview
    st.dataframe(df.head())

    # Generate XML for first 3
    st.markdown("### 🧾 Sample XML Output")
    for i in range(min(3, len(df))):
        xml = build_hrp_xml(df.iloc[i])
        st.markdown(f"#### Provider {i+1}")
        st.markdown(f"<div class='xml-box'>{xml}</div>", unsafe_allow_html=True)

    # Full XML download
    full_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<Providers>\n' + '\n'.join([build_hrp_xml(row) for _, row in df.iterrows()]) + '\n</Providers>'
    st.download_button("📥 Download Full HRP XML", full_xml, file_name="hrp_providers.xml", mime="application/xml")

if __name__ == "__main__":
    main()
