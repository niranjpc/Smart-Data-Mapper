import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import requests
import time

# --- Gemini API Helper ---
def gemini_generate(prompt, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(url, headers=headers, params=params, json=data)
    response.raise_for_status()
    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]

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
st.set_page_config(page_title="Smart Data Mapper (Gemini)", layout="wide")
st.title("🧠 Smart Data Mapper (Gemini-powered)")
st.caption("Upload reference data and a file to be mapped. Gemini will intelligently map, transform, and explain each field.")

with st.expander("ℹ️ How to use this tool", expanded=True):
    st.markdown("""
    1. **Upload one or more reference data files** (your mapping CSVs).
    2. **Upload the file to be mapped** (your provider data).
    3. **Click 'Process Mapping'** to see a smart preview, download a mapping report, and get your XML output.
    - Gemini will use your reference data, logic, and comments to make the best mapping.
    """)

# Get Gemini API key from secrets
api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.error("Please add your Gemini API key to Streamlit secrets as GEMINI_API_KEY.")
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
    st.info("Gemini is mapping and transforming your data. This may take a moment...")

    try:
        prov_columns = prov_df.columns.astype(str).tolist()
        mapping_preview = []
        results = []
        mapping_explanations = []
        mapping_report_rows = []

        # Prepare reference context for Gemini
        reference_context = []
        for rag_file, rag_df in rag_dfs.items():
            for _, row in rag_df.iterrows():
                reference_context.append(
                    f"Field: {row['fields']}, XML: {row['xml field']}, Logic: {row.get('logic','')}, Comments: {row.get('comments','')}, Source: {rag_file}"
                )
        reference_context_str = "\n".join(reference_context)

        # --- Gemini batching and caching ---
        gemini_cache = {}
        for col in prov_columns:
            if col in gemini_cache:
                gemini_response = gemini_cache[col]
            else:
                prompt = (
                    f"You are a US healthcare data mapping expert. "
                    f"Given the following reference data fields and mapping logic:\n{reference_context_str}\n\n"
                    f"Map the provider field '{col}' to the best XML field. "
                    f"Explain your reasoning, apply any transformation logic, and provide the XML path. "
                    f"Return your answer in this format:\n"
                    f"XML Field: <xml_path>\n"
                    f"Logic: <logic_applied>\n"
                    f"Comments: <comments>\n"
                    f"Confidence: <confidence 0-100%>\n"
                    f"Explanation: <explanation>"
                )
                gemini_response = gemini_generate(prompt, api_key)
                gemini_cache[col] = gemini_response
                time.sleep(1)  # Delay to avoid rate limit

            # Parse Gemini's response (simple parsing)
            xml_field = ""
            logic = ""
            comments = ""
            confidence = ""
            explanation = ""
            for line in gemini_response.splitlines():
                if line.lower().startswith("xml field:"):
                    xml_field = line.split(":",1)[1].strip()
                elif line.lower().startswith("logic:"):
                    logic = line.split(":",1)[1].strip()
                elif line.lower().startswith("comments:"):
                    comments = line.split(":",1)[1].strip()
                elif line.lower().startswith("confidence:"):
                    confidence = line.split(":",1)[1].strip()
                elif line.lower().startswith("explanation:"):
                    explanation = line.split(":",1)[1].strip()
            mapping_preview.append({
                'Provider Field': col,
                'XML Field': xml_field,
                'Logic': logic,
                'Comments': comments,
                'Confidence': confidence,
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

        st.success("✅ Processing completed using Gemini-powered mapping!")

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
- Uses Google Gemini 1.5 Pro for intelligent mapping, logic, and explanations
- Supports multiple reference data files
- Lets you download a mapping report (CSV) and the generated XML
""")
