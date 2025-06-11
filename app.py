import streamlit as st
import pandas as pd
import requests
import json
from io import BytesIO
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from lxml import etree
import re
import os

# Load Hugging Face API Token from Streamlit Secrets
HF_TOKEN = st.secrets["HF_TOKEN"]
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# Set up embedder
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# Build FAISS index
def build_faiss_index(rag_df, field_col='fields'):
    rag_fields = rag_df[field_col].astype(str).tolist()
    rag_embeddings = embedder.encode(rag_fields)
    index = faiss.IndexFlatL2(rag_embeddings.shape[1])
    index.add(np.array(rag_embeddings).astype('float32'))
    return index, rag_embeddings

# Search RAG for best match
def retrieve_mapping(field_name, rag_df, index, top_n=1):
    field_emb = embedder.encode([field_name])
    D, I = index.search(np.array(field_emb).astype('float32'), top_n)
    results = [rag_df.iloc[i] for i in I[0]]
    return results[0]

# Hugging Face Inference
def generate_explanation(input_field, mapping_row, input_value):
    prompt = (
        f"Field: {input_field}\n"
        f"Sample Value: {input_value}\n"
        f"XML Field: {mapping_row['xml field']}\n"
        f"Logic: {mapping_row['logic']}\n"
        f"Comments: {mapping_row['comments']}\n"
        "Explain in simple terms how to map this field for an XML transformation."
    )

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100}
    }

    response = requests.post(HF_API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        output = response.json()
        return output[0]["generated_text"].split("Comments:")[-1].strip()
    else:
        return f"Error from LLM: {response.status_code}"

# Clean XML tag
def safe_xml_tag(tag):
    if pd.isnull(tag) or not tag:
        return None
    tag = str(tag).split('/')[-1]
    tag = re.sub(r'[^a-zA-Z0-9_]', '_', tag)
    if tag and tag[0].isdigit():
        tag = f"field_{tag}"
    return tag

# Streamlit App
st.title("🧠 AI-Powered Provider Data Mapper (RAG + Mistral)")
st.markdown("Upload your RAG Mapping and Provider Data files to begin.")

# Upload RAG
rag_file = st.file_uploader("📁 Upload RAG Mapping CSV", type="csv")
# Upload Provider Input
provider_file = st.file_uploader("📄 Upload Provider Data (CSV/Excel)", type=["csv", "xlsx"])

if st.button("🚀 Process Mapping"):
    if not rag_file or not provider_file:
        st.error("Please upload both files.")
    else:
        rag_df = pd.read_csv(rag_file)
        index, _ = build_faiss_index(rag_df)

        # Read Provider file
        if provider_file.name.endswith(".csv"):
            provider_df = pd.read_csv(provider_file)
        else:
            provider_df = pd.read_excel(provider_file)

        mapping_report = []
        xml_root = etree.Element("Providers")

        for idx, row in provider_df.iterrows():
            provider = etree.SubElement(xml_root, "Provider")
            for col in provider_df.columns:
                try:
                    mapping_row = retrieve_mapping(col, rag_df, index)
                    explanation = generate_explanation(col, mapping_row, row[col])
                    xml_field = safe_xml_tag(mapping_row["xml field"]) or col
                    etree.SubElement(provider, xml_field).text = str(row[col])
                    mapping_report.append({
                        "Row": idx + 1,
                        "Input Field": col,
                        "Input Value": row[col],
                        "XML Field": xml_field,
                        "Logic": mapping_row["logic"],
                        "LLM Explanation": explanation
                    })
                except Exception as e:
                    st.warning(f"Error processing column {col}: {str(e)}")

        # Show Report
        st.subheader("📊 Mapping Report")
        report_df = pd.DataFrame(mapping_report)
        st.dataframe(report_df)

        # Show XML
        st.subheader("🧾 Generated XML")
        xml_str = etree.tostring(xml_root, pretty_print=True).decode()
        st.code(xml_str, language="xml")

        # Download XML
        xml_bytes = BytesIO(xml_str.encode("utf-8"))
        st.download_button("📥 Download XML", data=xml_bytes, file_name="output.xml", mime="application/xml")
