import streamlit as st
import pandas as pd
import numpy as np
import requests
import faiss
from sentence_transformers import SentenceTransformer
from lxml import etree
from io import BytesIO
import re

# Load Hugging Face API token securely
HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ----------------- Utility Functions ------------------

def generate_explanation(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100},
    }
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    result = response.json()
    if isinstance(result, list):
        return result[0]['generated_text']
    return "Explanation generation failed."

def safe_xml_tag(tag):
    if pd.isnull(tag) or not tag:
        return None
    tag = str(tag).split("/")[-1]
    tag = re.sub(r"[^a-zA-Z0-9_]", "_", tag)
    if tag[0].isdigit():
        tag = f"field_{tag}"
    return tag

# ----------------- FAISS Setup ------------------

def build_faiss_index(rag_df, field_col='fields'):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    rag_fields = rag_df[field_col].astype(str).tolist()
    embeddings = embedder.encode(rag_fields)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return embedder, index

def retrieve_mapping(field_name, rag_df, embedder, index, field_col='fields'):
    field_emb = embedder.encode([field_name])
    D, I = index.search(np.array(field_emb).astype('float32'), 1)
    return rag_df.iloc[I[0][0]]

# ----------------- Streamlit UI ------------------

st.title("🧠 AI-Powered Provider Data Mapper (Streamlit + Mistral)")

with st.sidebar:
    st.header("📁 Upload Files")
    rag_file = st.file_uploader("RAG Mapping CSV", type=["csv"])
    provider_file = st.file_uploader("Provider Input CSV", type=["csv", "xlsx"])
    run_btn = st.button("🚀 Process Mapping")

if run_btn:
    if rag_file and provider_file:
        with st.spinner("🔍 Building index and processing data..."):
            rag_df = pd.read_csv(rag_file)
            embedder, index = build_faiss_index(rag_df)

            if provider_file.name.endswith(".csv"):
                df = pd.read_csv(provider_file)
            else:
                df = pd.read_excel(provider_file)

            mapping_report = []
            xml_root = etree.Element("Providers")

            for idx, row in df.iterrows():
                provider = etree.SubElement(xml_root, "Provider")
                for col in df.columns:
                    mapping_row = retrieve_mapping(col, rag_df, embedder, index)
                    value = row[col]
                    prompt = (
                        f"Field: {col}\n"
                        f"Sample Value: {value}\n"
                        f"XML Field: {mapping_row['xml field']}\n"
                        f"Logic: {mapping_row['logic']}\n"
                        f"Comments: {mapping_row['comments']}\n"
                        f"Explain in simple terms how to map this field for HRP XML."
                    )
                    explanation = generate_explanation(prompt)
                    xml_field = safe_xml_tag(mapping_row['xml field']) or col
                    child = etree.SubElement(provider, xml_field)
                    child.text = str(value)
                    mapping_report.append({
                        "Row": idx + 1,
                        "Input Field": col,
                        "Input Value": value,
                        "XML Field": xml_field,
                        "Logic": mapping_row['logic'],
                        "LLM Explanation": explanation
                    })

            report_df = pd.DataFrame(mapping_report)
            st.subheader("📄 Mapping Report")
            st.dataframe(report_df)

            xml_bytes = BytesIO(etree.tostring(xml_root, pretty_print=True))
            st.subheader("📥 Download XML")
            st.download_button("Download XML", xml_bytes, file_name="output.xml", mime="application/xml")
    else:
        st.warning("Please upload both RAG mapping file and provider input file.")
