import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import requests
import tempfile

# Get Hugging Face API key from Streamlit secrets
hf_api_key = st.secrets["HUGGINGFACE_TOKEN"]

# Hugging Face API endpoints
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-small"

def get_embeddings(texts, api_key):
    """Get embeddings from Hugging Face Inference API."""
    API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": texts})
    response.raise_for_status()
    return response.json()

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def generate_text(prompt, api_key):
    """Generate text using Hugging Face Inference API."""
    API_URL = f"https://api-inference.huggingface.co/models/{GENERATION_MODEL}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 64}}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, dict) and 'error' in result:
            return f"[Error: {result['error']}]"
        else:
            return str(result)
    else:
        return f"[API Error: {response.status_code}]"

st.title("🧠 Provider Data Mapper (RAG + LLM, Hugging Face API Only)")
st.markdown("Upload your mapping and provider files. The app will auto-map fields and generate XML using only the Hugging Face Inference API (no local ML dependencies).")

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

if st.button("🚀 Process Mapping"):
    st.info("Processing... please wait (API calls may take a few seconds)")

    # Get embeddings for RAG fields and provider columns
    rag_fields = rag_df['fields'].astype(str).tolist()
    prov_columns = prov_df.columns.astype(str).tolist()

    # Get embeddings from Hugging Face API
    rag_embeddings = get_embeddings(rag_fields, hf_api_key)
    prov_embeddings = get_embeddings(prov_columns, hf_api_key)

    import numpy as np
    results = []
    mapping_explanations = []

    for i, row in prov_df.iterrows():
        entry = {}
        explain = {}
        for col_idx, col in enumerate(prov_df.columns):
            # Find best match in RAG fields using cosine similarity
            similarities = [
                cosine_similarity(prov_embeddings[col_idx], rag_emb)
                for rag_emb in rag_embeddings
            ]
            best_idx = int(np.argmax(similarities))
            best_match = rag_df.iloc[best_idx]
            xml_path = best_match['xml field']
            value = row[col]
            entry[xml_path] = value
            # Generate mapping logic explanation
            prompt = (
                f"Explain why the provider field '{col}' should be mapped to XML field '{xml_path}'. "
                f"Mapping logic: {best_match.get('logic', '')}. Comments: {best_match.get('comments', '')}."
            )
            logic_explanation = generate_text(prompt, hf_api_key)
            explain[col] = logic_explanation
        results.append(entry)
        mapping_explanations.append(explain)

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
    for idx, data in enumerate(results):
        xml_str = build_xml(data)
        xml_strings.append(xml_str)
        st.code(xml_str, language="xml")
        with st.expander(f"Mapping Explanations for Provider Row {idx+1}"):
            for field, explanation in mapping_explanations[idx].items():
                st.markdown(f"**{field}**: {explanation}")

    # Write to downloadable file
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".xml") as f:
        f.write("<providers>\n" + "\n".join(xml_strings) + "\n</providers>")
        f.flush()
        f.seek(0)
        st.download_button("⬇️ Download Full XML", f.read(), file_name="providers.xml", mime="application/xml")

st.markdown("---")
st.markdown("📌 **Sample Files:**")
st.markdown("- [sample_rag_mapping.csv](https://github.com/niranjpc/provider-mapper/blob/main/sample_rag_mapping.csv)")
st.markdown("- [sample_provider_input.csv](https://github.com/niranjpc/provider-mapper/blob/main/sample_provider_input.csv)")
