import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import requests
import time

# --- Hugging Face API Helpers ---
def get_embeddings(texts, api_key):
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
    if response.status_code != 200:
        st.error(f"Embedding API error {response.status_code}: {response.text}")
        st.stop()
    return response.json()

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def generate_text(prompt, api_key):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 80, "temperature": 0.3, "return_full_text": False}}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            return result[0]['generated_text'].strip()
        else:
            return f"[Error: Unexpected API response format: {result}]"
    elif response.status_code == 503:
        st.warning("Text generation model is loading. Retrying in a moment...")
        time.sleep(10)
        return generate_text(prompt, api_key)
    else:
        return f"[API Error: {response.status_code} - {response.text}]"

def build_xml(provider_data):
    provider_el = ET.Element("provider")
    for path, val in provider_data.items():
        if pd.isna(val):
            val = ""
        parts = path.split("/")
        current = provider_el
        for part in parts[:-1]:
            found = current.find(part)
            if found is None:
                found = ET.SubElement(current, part)
            current = found
        ET.SubElement(current, parts[-1]).text = str(val)
    return ET.tostring(provider_el, encoding="unicode")

# --- Streamlit UI ---
st.set_page_config(page_title="Smart Data Mapper (Hugging Face)", layout="wide")
st.title("🧠 Smart Data Mapper (Hugging Face-powered)")
st.caption("Upload reference data and a file to be mapped. The app will use semantic AI for mapping, logic, and explanations.")

with st.expander("ℹ️ How to use this tool", expanded=True):
    st.markdown("""
    1. **Upload one or more reference data files** (your mapping CSVs).
    2. **Upload the file to be mapped** (your provider data).
    3. **Click 'Process Mapping'** to see a smart preview, download a mapping report, and get your XML output.
    - The app uses Hugging Face's best free models for semantic mapping and logic.
    """)

# Get Hugging Face API key from secrets
api_key = st.secrets.get("HUGGINGFACE_TOKEN", "")
if not api_key:
    st.error("Please add your Hugging Face API key to Streamlit secrets as HUGGINGFACE_TOKEN.")
    st.stop()

st.divider()

# Upload reference data
st.header("📁 Upload Reference Data")
rag_files = st.file_uploader(
    "Upload one or more mapping CSVs (reference data)", 
    type=["csv"], 
    accept_multiple_files=True,
    help="These files define how your provider fields should be mapped to XML."
)
rag_dfs = {}
if rag_files:
    for rag_file in rag_files:
        rag_df = pd.read_csv(rag_file)
        rag_dfs[rag_file.name] = rag_df
    st.success(f"{len(rag_files)} reference data file(s) uploaded.")
    with st.expander("Preview Reference Data", expanded=False):
        for name, df in rag_dfs.items():
            st.markdown(f"**{name}:**")
            st.dataframe(df.head())
else:
    st.info("Please upload at least one reference data file to continue.")
    st.stop()

st.divider()

# Upload provider file
st.header("📄 Upload the File to be Mapped")
prov_file = st.file_uploader(
    "Upload your provider data (CSV or Excel)", 
    type=["csv", "xlsx"],
    help="This is the file whose fields you want to map to XML."
)
if prov_file:
    if prov_file.name.endswith(".csv"):
        prov_df = pd.read_csv(prov_file)
    else:
        prov_df = pd.read_excel(prov_file)
    st.success("File to be mapped uploaded.")
    with st.expander("Preview File to be Mapped", expanded=False):
        st.dataframe(prov_df.head())
else:
    st.info("Please upload the file to be mapped to continue.")
    st.stop()

st.divider()

# Process Mapping
if st.button("🚀 Process Mapping", use_container_width=True):
    st.info("Processing with Hugging Face models. This may take a moment...")

    try:
        prov_columns = prov_df.columns.astype(str).tolist()
        mapping_preview = []
        results = []
        mapping_explanations = []
        mapping_report_rows = []

        # Prepare all reference fields for embeddings
        reference_rows = []
        for rag_file, rag_df in rag_dfs.items():
            for _, row in rag_df.iterrows():
                reference_rows.append({
                    "fields": str(row['fields']),
                    "xml field": str(row['xml field']),
                    "logic": str(row.get('logic', '')),
                    "comments": str(row.get('comments', '')),
                    "rag_file": rag_file
                })
        reference_fields = [r["fields"] for r in reference_rows]

        # Get embeddings for reference fields (batch)
        st.info("Getting embeddings for reference fields...")
        ref_embeddings = get_embeddings(reference_fields, api_key)

        # Get embeddings for provider columns (batch)
        st.info("Getting embeddings for provider fields...")
        prov_embeddings = get_embeddings(prov_columns, api_key)

        # For each provider field, find best match in reference fields
        for idx, col in enumerate(prov_columns):
            prov_emb = prov_embeddings[idx]
            best_score = -1
            best_idx = -1
            for j, ref_emb in enumerate(ref_embeddings):
                score = cosine_similarity(prov_emb, ref_emb)
                if score > best_score:
                    best_score = score
                    best_idx = j
            best_ref = reference_rows[best_idx]
            xml_path = best_ref["xml field"]
            logic = best_ref["logic"]
            comments = best_ref["comments"]
            rag_file = best_ref["rag_file"]
            confidence = f"{best_score*100:.2f}%"
            # Generate explanation using text generation model
            prompt = (
                f"Explain why the provider field '{col}' should be mapped to XML field '{xml_path}'. "
                f"Mapping logic: {logic}. Comments: {comments}."
            )
            explanation = generate_text(prompt, api_key)
            mapping_preview.append({
                'Provider Field': col,
                'XML Field': xml_path,
                'Logic': logic,
                'Comments': comments,
                'Confidence': confidence,
                'Reference File': rag_file,
                'Explanation': explanation
            })

        st.subheader("📋 Field Mappings Preview")
        preview_df = pd.DataFrame(mapping_preview)
        st.dataframe(preview_df)

        progress_bar = st.progress(0)
        total_rows = len(prov_df)

        for i, row in prov_df.iterrows():
            entry = {}
            explain = {}
            for col in prov_df.columns:
                mapping = next((m for m in mapping_preview if m['Provider Field'] == col), None)
                xml_path = mapping['XML Field'] if mapping else col
                value = row[col]
                entry[xml_path] = value
                explanation = mapping['Explanation'] if mapping else ""
                explain[col] = explanation
                mapping_report_rows.append({
                    'Provider Row': i+1,
                    'Provider Field': col,
                    'Value': value,
                    'XML Field': xml_path,
                    'Logic': mapping['Logic'] if mapping else "",
                    'Comments': mapping['Comments'] if mapping else "",
                    'Confidence': mapping['Confidence'] if mapping else "",
                    'Reference File': mapping['Reference File'] if mapping else "",
                    'Explanation': explanation
                })
            results.append(entry)
            mapping_explanations.append(explain)
            progress_bar.progress((i + 1) / total_rows)

        st.divider()
        st.subheader("📊 Output XMLs")
        xml_strings = []
        for idx, data in enumerate(results):
            xml_str = build_xml(data)
            xml_strings.append(xml_str)
            with st.expander(f"Provider {idx+1} - XML Output", expanded=False):
                st.code(xml_str, language="xml")
            with st.expander(f"Provider {idx+1} - Mapping Explanations", expanded=False):
                for field, explanation in mapping_explanations[idx].items():
                    st.markdown(f"**{field}**: {explanation}")

        # Download XML
        full_xml = "<providers>\n" + "\n".join(xml_strings) + "\n</providers>"
        st.download_button(
            "⬇️ Download Full XML",
            full_xml,
            file_name="providers.xml",
            mime="application/xml"
        )

        # Download Mapping Report
        report_df = pd.DataFrame(mapping_report_rows)
        csv = report_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Mapping Report (CSV)",
            csv,
            file_name="mapping_report.csv",
            mime="text/csv"
        )

        st.success("✅ Processing completed using Hugging Face-powered mapping!")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your files and try again.")

st.markdown("---")
st.markdown("📌 **Sample Files:**")
st.markdown("- [sample_rag_mapping.csv](https://github.com/niranjpc/provider-mapper/blob/main/sample_rag_mapping.csv)")
st.markdown("- [sample_provider_input.csv](https://github.com/niranjpc/provider-mapper/blob/main/sample_provider_input.csv)")

st.markdown("---")
st.markdown("🔧 **How it works:**")
st.markdown("""
- Uses Hugging Face's best free models for semantic mapping and logic
- Supports multiple reference data files
- Lets you download a mapping report (CSV) and the generated XML
""")
