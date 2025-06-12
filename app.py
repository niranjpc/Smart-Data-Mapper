import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

# --- Helper Functions ---

def text_similarity(text1, text2):
    return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()

def find_best_match(provider_column, rag_dfs):
    """Find the best matching RAG field for a provider column across multiple reference data files."""
    best_score = 0
    best_row = None
    best_rag_file = ""
    provider_str = str(provider_column) if pd.notna(provider_column) else ""
    def safe_words(val):
        try:
            return set(str(val).lower().split())
        except Exception:
            return set()
    for rag_file, rag_df in rag_dfs.items():
        for _, row in rag_df.iterrows():
            rag_field = row['fields']
            rag_str = str(rag_field) if pd.notna(rag_field) else ""
            score = text_similarity(provider_str, rag_str)
            provider_words = safe_words(provider_str)
            rag_words = safe_words(rag_str)
            common_words = provider_words.intersection(rag_words)
            if common_words:
                score += 0.05
            score = min(score, 1.0)
            if score > best_score:
                best_score = score
                best_row = row
                best_rag_file = rag_file
    return best_row, best_score, best_rag_file

def generate_simple_explanation(provider_field, xml_field, similarity_score, logic, comments, rag_file):
    base = (
        f"Field '{provider_field}' was mapped to XML field '{xml_field}' "
        f"from reference file '{rag_file}' with {similarity_score:.2%} confidence. "
        f"Logic: {logic}. Comments: {comments}"
    )
    if similarity_score < 0.5:
        base += " ⚠️ Low confidence: Please review this mapping."
    return base

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

# --- Sleek Streamlit UI ---

st.set_page_config(page_title="Smart Data Mapper", layout="wide")
st.markdown(
    """
    <style>
    .stButton>button {background-color:#4F8BF9;color:white;font-weight:bold;}
    .stProgress>div>div>div>div {background-color: #4F8BF9;}
    .confidence-high {color: #228B22; font-weight: bold;}
    .confidence-medium {color: #E6B800; font-weight: bold;}
    .confidence-low {color: #D7263D; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True
)

st.title("🧠 Smart Data Mapper")
st.caption("A modern, AI-inspired tool for mapping provider data to reference XML structures.")

with st.expander("ℹ️ How to use this tool", expanded=True):
    st.markdown("""
    1. **Upload one or more reference data files** (your mapping CSVs).
    2. **Upload the file to be mapped** (your provider data).
    3. **Click 'Process Mapping'** to see a smart preview, download a mapping report, and get your XML output.
    - Confidence scores are color-coded for quick review.
    - Low-confidence matches are flagged for your attention.
    """)

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
    st.info("Processing mappings using text similarity...")

    try:
        prov_columns = prov_df.columns.astype(str).tolist()
        mapping_preview = []
        results = []
        mapping_explanations = []
        mapping_report_rows = []

        # Preview and mapping logic
        for col in prov_columns:
            best_row, similarity, rag_file = find_best_match(col, rag_dfs)
            xml_path = best_row['xml field']
            logic = best_row.get('logic', '')
            comments = best_row.get('comments', '')
            # Color code confidence
            if similarity >= 0.85:
                conf_class = "confidence-high"
            elif similarity >= 0.65:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"
            mapping_preview.append({
                'Provider Field': col,
                'XML Field': xml_path,
                'Logic': logic,
                'Comments': comments,
                'Confidence': f"<span class='{conf_class}'>{similarity:.2%}</span>" + (" ⚠️" if similarity < 0.5 else ""),
                'Reference File': rag_file
            })

        st.subheader("📋 Field Mappings Preview")
        preview_df = pd.DataFrame(mapping_preview)
        st.write(preview_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        progress_bar = st.progress(0)
        total_rows = len(prov_df)

        for i, row in prov_df.iterrows():
            entry = {}
            explain = {}
            for col in prov_df.columns:
                best_row, similarity, rag_file = find_best_match(col, rag_dfs)
                xml_path = best_row['xml field']
                logic = best_row.get('logic', '')
                comments = best_row.get('comments', '')
                value = row[col]
                entry[xml_path] = value
                explanation = generate_simple_explanation(col, xml_path, similarity, logic, comments, rag_file)
                explain[col] = explanation
                mapping_report_rows.append({
                    'Provider Row': i+1,
                    'Provider Field': col,
                    'Value': value,
                    'XML Field': xml_path,
                    'Logic': logic,
                    'Comments': comments,
                    'Confidence': f"{similarity:.2%}" + (" ⚠️" if similarity < 0.5 else ""),
                    'Reference File': rag_file,
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

        st.success("✅ Processing completed using text similarity matching!")

        st.markdown("#### Example XML Output (for one provider):")
        st.code(
            '''<provider>
    <npi>1234567890</npi>
    <name>
        <first>John</first>
        <last>Doe</last>
    </name>
    <dob>1980-01-01</dob>
    <specialty>Cardiology</specialty>
    <license>
        <number>AB12345</number>
        <state>CA</state>
    </license>
</provider>''', language="xml"
        )

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
- Supports multiple reference data files
- Shows a detailed mapping preview with logic, comments, and confidence (color-coded)
- Lets you download a mapping report (CSV) and the generated XML
- Uses Python's built-in text similarity algorithms (no API required)
""")
