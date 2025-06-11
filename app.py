import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import tempfile
import os

# Load lightweight model for demo
@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="tiiuae/falcon-rw-1b")
    return model, generator

embedder, generator = load_models()

st.title("🚀 Smart Provider Mapper (LLM + RAG Demo)")
st.markdown("Demo for auto-mapping provider input to XML using AI + mapping config")

# Upload RAG mapping file
st.header("📁 Step 1: Upload RAG Mapping File")
rag_file = st.file_uploader("Upload `sample_rag_mapping.csv`", type=["csv"])
if rag_file:
    rag_df = pd.read_csv(rag_file)
    st.success("RAG Mapping file uploaded!")
else:
    st.stop()

# Upload provider file
st.header("📄 Step 2: Upload Provider File")
prov_file = st.file_uploader("Upload `sample_provider_input.csv`", type=["csv", "xlsx"])
if prov_file:
    if prov_file.name.endswith(".csv"):
        prov_df = pd.read_csv(prov_file)
    else:
        prov_df = pd.read_excel(prov_file)
    st.success("Provider file uploaded!")
else:
    st.stop()

# Main logic
if st.button("🚀 Process Mapping"):
    st.info("Processing... please wait")

    # Build mapping index using embeddings
    rag_embeddings = embedder.encode(rag_df['fields'], convert_to_tensor=True)
    results = []

    for i, row in prov_df.iterrows():
        entry = {}
        for col in prov_df.columns:
            query_emb = embedder.encode(col, convert_to_tensor=True)
            sims = (rag_embeddings @ query_emb).cpu().numpy()
            best_match = rag_df.iloc[sims.argmax()]
            xml_path = best_match['xml field']
            value = row[col]
            entry[xml_path] = value
        results.append(entry)

    # Convert to XML
    def build_xml(provider_data):
        provider_el = ET.Element("provider")
        for path, val in provider_data.items():
            parts = path.split("/")
            current = provider_el
            for part in parts[:-1]:
                found = current.find(part)
                if found is None:
                    found = ET.SubElement(current, part)
                current = found
            ET.SubElement(current, parts[-1]).text = str(val)
        return ET.tostring(provider_el, encoding="unicode")

    st.subheader("📊 Output XMLs")
    xml_strings = []
    for data in results:
        xml_str = build_xml(data)
        xml_strings.append(xml_str)
        st.code(xml_str, language="xml")

    # Write to downloadable file
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".xml") as f:
        f.write("<providers>\n" + "\n".join(xml_strings) + "\n</providers>")
        st.download_button("⬇️ Download Full XML", f.read(), file_name="providers.xml", mime="application/xml")

st.markdown("---")
st.markdown("📌 **Sample Files:**")
st.markdown("- [sample_rag_mapping.csv](https://github.com/niranjpc/provider-mapper/blob/main/sample_rag_mapping.csv)")
st.markdown("- [sample_provider_input.csv](https://github.com/niranjpc/provider-mapper/blob/main/sample_provider_input.csv)")
