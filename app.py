# Intelligent HRP AI Data Mapper with Hugging Face Integration + Manual + Heuristic Fallback
# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import traceback
import logging
from datetime import datetime
from difflib import SequenceMatcher
import xml.etree.ElementTree as ET
from xml.dom import minidom
import requests
from typing import List, Dict, Tuple, Any

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    MAX_PREVIEW_RECORDS = 5
    XML_INDENT = "  "
    SUPPORTED_FORMATS = ["csv", "xlsx", "xls"]
    DEFAULT_ENCODING = "utf-8"
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    SIMILARITY_THRESHOLD = 0.6
    HF_API_KEY = "hf_your_key_here"  # Replace with your actual Hugging Face API key
    HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
    API_OPTIONS = ["FacilityLoad", "PractitionerLoad", "MemberLoad"]
    ENABLE_MANUAL_MAPPING = True
    ENABLE_LLM_MAPPING = True
    ENABLE_SIMILARITY_FALLBACK = True

# --- Utility Functions ---
def calculate_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# --- LLM Functions ---
def call_hf_llm(prompt: str, max_tokens: int = 200) -> str:
    url = f"https://api-inference.huggingface.co/models/{Config.HF_MODEL}"
    headers = {"Authorization": f"Bearer {Config.HF_API_KEY}"}
    payload = {"inputs": prompt, "max_new_tokens": max_tokens}
    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        return result[0].get("generated_text", "").strip()
    except Exception as e:
        logger.error(f"HF LLM Error: {e}")
        return ""

def llm_map_field_to_xml(source_field: str, xml_candidates: List[str]) -> str:
    prompt = f"""
You are a healthcare data mapping expert.

Field to map: "{source_field}"
Choose the best matching XML path from below:
{chr(10).join(f"- {x}" for x in xml_candidates)}

Only reply with the best matching XML path.
"""
    return call_hf_llm(prompt)

def llm_explain_mapping(source_field: str, target_field: str) -> str:
    prompt = f"""
Explain the mapping logic from "{source_field}" to XML path "{target_field}" in a healthcare context.
Include whether it is direct, conditional, lookup, formatting, or calculation based.
"""
    return call_hf_llm(prompt)

# --- XML Builder ---
def build_dynamic_xml(row: pd.Series, pretty_print: bool = False, root_name: str = "HealthcareData") -> str:
    try:
        root = ET.Element(root_name)
        for path, value in row.items():
            if pd.isna(value) or not path:
                continue
            parts = [part.strip() for part in path.split("/") if part.strip()]
            current = root
            for part in parts[:-1]:
                child = current.find(part)
                if child is None:
                    child = ET.SubElement(current, part)
                current = child
            ET.SubElement(current, parts[-1]).text = str(value)
        rough_string = ET.tostring(root, encoding='unicode')
        if pretty_print:
            return minidom.parseString(rough_string).toprettyxml(indent=Config.XML_INDENT)
        return rough_string
    except Exception as e:
        logger.error(f"XML build error: {e}")
        return "<Error>XML Generation Failed</Error>"

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="HRP LLM Mapper", layout="wide")
    st.markdown("""
        <style>
        .main-header h1 {font-size: 2.5em; margin-bottom: 0;}
        .main-header p {font-size: 1.1em; margin-top: 0; color: grey;}
        </style>
        <div class="main-header">
            <h1>🧠 Intelligent HRP AI Data Mapper</h1>
            <p>Smart field mapping with Hugging Face LLM ✨ + Manual override + Similarity fallback</p>
        </div>
    """, unsafe_allow_html=True)

    api_choice = st.selectbox("🔄 Choose Target API for XML Structure:", Config.API_OPTIONS)
    rag_file = st.file_uploader("📄 Upload Reference Mapping CSV", type="csv")
    data_file = st.file_uploader("📁 Upload Healthcare Input Data", type=Config.SUPPORTED_FORMATS)

    if not rag_file or not data_file:
        st.info("Upload both reference and input data files to begin.")
        return

    rag_df = pd.read_csv(rag_file)
    file_ext = data_file.name.split(".")[-1]
    data_df = pd.read_csv(data_file) if file_ext == "csv" else pd.read_excel(data_file)

    st.subheader("🔎 Mapping Setup")
    source_col = st.selectbox("Select source field column (input field names):", rag_df.columns)
    target_col = st.selectbox("Select target XML path column:", rag_df.columns)

    preview_count = st.slider("How many records to preview?", 1, 10, Config.MAX_PREVIEW_RECORDS)
    pretty_print = st.checkbox("Pretty print XML", True)
    manual_mode = st.checkbox("🛠️ Enable Manual Mapping Mode", value=False)

    reference_fields = rag_df[source_col].dropna().unique().tolist()
    xml_targets = rag_df[target_col].dropna().unique().tolist()
    field_map = {}
    logic_map = {}
    method_map = {}

    if manual_mode:
        st.subheader("🔧 Manual Field Mapping")
        for ref_field in reference_fields:
            selected = st.selectbox(f"Map '{ref_field}' to:", [""] + xml_targets, key=ref_field)
            if selected:
                field_map[ref_field] = selected
                logic_map[ref_field] = "Manual override"
                method_map[ref_field] = "Manual"
    else:
        with st.spinner("🔍 Mapping fields..."):
            for ref_field in reference_fields:
                xml_path = ""
                logic = ""
                method = ""
                if Config.ENABLE_LLM_MAPPING:
                    xml_path = llm_map_field_to_xml(ref_field, xml_targets)
                    logic = llm_explain_mapping(ref_field, xml_path)
                    method = "LLM"
                if not xml_path and Config.ENABLE_SIMILARITY_FALLBACK:
                    scores = [(t, calculate_similarity(ref_field, t)) for t in xml_targets]
                    best_match = max(scores, key=lambda x: x[1], default=("", 0))
                    if best_match[1] > Config.SIMILARITY_THRESHOLD:
                        xml_path = best_match[0]
                        logic = f"Similarity matched (score {best_match[1]:.2f})"
                        method = "Heuristic"
                field_map[ref_field] = xml_path
                logic_map[ref_field] = logic
                method_map[ref_field] = method

    st.success("✅ Mapping complete. Generating XML previews...")

    mapped_data = []
    for _, row in data_df.iterrows():
        xml_row = {}
        for ref_field, xml_path in field_map.items():
            if ref_field in row:
                xml_row[xml_path] = row[ref_field]
        mapped_data.append(xml_row)

    st.subheader("📄 XML Record Preview")
    for i in range(min(preview_count, len(mapped_data))):
        xml_content = build_dynamic_xml(pd.Series(mapped_data[i]), pretty_print=pretty_print, root_name=api_choice)
        st.code(xml_content, language='xml')

    if st.button("📥 Download All XML"):
        full_xml = '<?xml version="1.0"?>\n<' + api_choice + 'Records>\n' + '\n'.join(
            build_dynamic_xml(pd.Series(row), pretty_print=pretty_print, root_name=api_choice) for row in mapped_data
        ) + f'\n</{api_choice}Records>'
        st.download_button("Download XML File", data=full_xml, file_name=f"{api_choice}_data.xml", mime="application/xml")

    st.subheader("🧾 Field Mapping Audit")
    audit_df = pd.DataFrame({
        "Input Field": list(field_map.keys()),
        "Mapped XML Field": list(field_map.values()),
        "Logic Explanation": list(logic_map.values()),
        "Mapping Method": list(method_map.values())
    })
    st.dataframe(audit_df)
    st.download_button("📤 Download Mapping Audit", data=audit_df.to_csv(index=False), file_name="mapping_audit.csv", mime="text/csv")

if __name__ == "__main__":
    main()
