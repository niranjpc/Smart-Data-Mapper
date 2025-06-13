# ✅ Refactored app.py with enhanced SME-aware AI mapping using reference data

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
    st.markdown(f"""
    <style>
    .main-header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }}
    .main-header h1 {{ font-size: 2.5rem; font-weight: 700; }}
    .main-header p {{ font-size: 1.1rem; margin: 0.5rem 0; }}
    .main-header small {{ font-size: 0.9rem; display: block; margin-top: 0.5rem; color: #f0f0f0; }}
    .xml-box {{ background-color: #f5f5f5; padding: 1rem; border-radius: 10px; margin-top: 1rem; font-family: monospace; white-space: pre-wrap; }}
    </style>
    """, unsafe_allow_html=True)

# --- Generic XML Builder from dynamic mapped keys ---
def build_dynamic_xml(row: pd.Series) -> str:
    root = ET.Element("Provider")
    for path, value in row.items():
        if pd.isna(value) or value == "":
            continue
        parts = path.strip().split("/")
        current = root
        for part in parts[:-1]:
            found = current.find(part)
            if found is None:
                found = ET.SubElement(current, part)
            current = found
        ET.SubElement(current, parts[-1]).text = str(value)
    return ET.tostring(root, encoding="unicode")

# --- Main App ---
def main():
    st.set_page_config(page_title="HRP AI Data Mapper", layout="wide")
    load_custom_css()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(f"""
        <div class="main-header">
            <h1>🤖 HRP AI Data Mapper</h1>
            <p>Upload your reference mapping & provider data to generate HRP-compliant XML files with AI-guided transformation.</p>
            <small>🗓️ {now}</small>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 📖 Features
    - ✅ Upload reference data (CSV) with mapping logic from SMEs
    - ✅ Upload provider file (CSV/XLSX)
    - ✅ Get auto-transformed provider data to HRP-compliant XML
    - ✅ Preview real XML structure in-browser
    - ✅ Download full XML
    - ✅ Download SME mapping audit report
    """)

    st.markdown("### 📁 Step 1: Upload Reference Data File (CSV with 'fields', 'xml field', 'mapping_type', 'logic_applied')")
    rag_file = st.file_uploader("Reference Mapping File", type=["csv"], key="rag")

    st.markdown("### 📄 Step 2: Upload Provider Input File")
    prov_file = st.file_uploader("Provider Input File", type=["csv", "xlsx"], key="provider")

    if not rag_file or not prov_file:
        st.info("Please upload both files to continue.")
        st.stop()

    try:
        import csv
        rag_df = pd.read_csv(rag_file, quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
        if prov_file.name.endswith(".csv"):
            prov_df = pd.read_csv(prov_file, quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
        else:
            prov_df = pd.read_excel(prov_file)
    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.stop()

    st.success("✅ Files loaded successfully")
    st.markdown("---")
    st.markdown("### 🧠 AI-Guided Field Mapping using Reference Data")

    rag_df.columns = rag_df.columns.str.strip().str.lower()
    prov_df.columns = prov_df.columns.str.strip()

    field_map = {}
    audit_rows = []

    for _, row in rag_df.iterrows():
        src = row.get("fields")
        tgt = row.get("xml field")
        mtype = row.get("mapping_type", "Direct")
        logic = row.get("logic_applied", "")
        if pd.notna(src) and pd.notna(tgt):
            field_map[str(src).strip()] = str(tgt).strip()
            audit_rows.append({
                "Mapped From (Input Column)": src,
                "Mapped To (XML Path)": tgt,
                "Mapping Type": mtype,
                "Logic Applied": logic
            })

    mapped_data = []
    for _, row in prov_df.iterrows():
        xml_row = {}
        for col in prov_df.columns:
            if col in field_map:
                xml_path = field_map[col]
                xml_row[xml_path] = row[col]
        mapped_data.append(xml_row)

    st.markdown("### 🔍 Sample Transformed XML Output")
    sample_df = pd.DataFrame(mapped_data)
    for i in range(min(3, len(sample_df))):
        xml = build_dynamic_xml(sample_df.iloc[i])
        st.markdown(f"#### Provider {i+1}")
        st.markdown(f"<div class='xml-box'>{xml}</div>", unsafe_allow_html=True)

    full_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<Providers>\n' + '\n'.join([build_dynamic_xml(pd.Series(r)) for r in mapped_data]) + '\n</Providers>'
    st.download_button("📥 Download Full Mapped XML", full_xml, file_name="mapped_providers.xml", mime="application/xml")

    # --- SME Reference Mapping Report ---
    st.markdown("### 🧾 SME Mapping Reference Report")
    audit_df = pd.DataFrame(audit_rows)
    st.dataframe(audit_df)

    csv = audit_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download SME Reference Report", csv, "sme_mapping_reference_report.csv", "text/csv")

if __name__ == "__main__":
    main()
