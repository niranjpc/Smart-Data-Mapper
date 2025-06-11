import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import requests
import tempfile
import time
from difflib import SequenceMatcher

# Simple text similarity approach - no API required
def text_similarity(text1, text2):
    """Calculate similarity between two texts using built-in difflib."""
    return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()

def find_best_match(provider_column, rag_fields_list):
    """Find the best matching RAG field for a provider column."""
    best_score = 0
    best_idx = 0
    
    for i, rag_field in enumerate(rag_fields_list):
        # Calculate similarity
        score = text_similarity(provider_column, rag_field)
        
        # Also check if keywords match
        provider_words = set(str(provider_column).lower().split())
        rag_words = set(str(rag_field).lower().split())
        
        # Boost score if there are common words
        common_words = provider_words.intersection(rag_words)
        if common_words:
            score += len(common_words) * 0.1
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return best_idx, best_score

def generate_simple_explanation(provider_field, xml_field, similarity_score):
    """Generate a simple explanation for the mapping."""
    explanations = [
        f"Field '{provider_field}' matches '{xml_field}' with {similarity_score:.2%} similarity.",
        f"The system mapped '{provider_field}' to '{xml_field}' based on text similarity analysis.",
        f"Provider field '{provider_field}' was automatically mapped to XML path '{xml_field}'.",
    ]
    return explanations[min(2, int(similarity_score * 3))]

st.title("🧠 Provider Data Mapper (Simple Text Matching)")
st.markdown("Upload your mapping and provider files. The app will auto-map fields using text similarity (no API required).")

# Upload RAG mapping file
st.header("📁 Step 1: Upload RAG Mapping File")
rag_file = st.file_uploader("Upload `sample_rag_mapping.csv`", type=["csv"])
if rag_file:
    rag_df = pd.read_csv(rag_file)
    st.success("RAG Mapping file uploaded!")
    st.dataframe(rag_df.head())
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
    st.dataframe(prov_df.head())
else:
    st.stop()

if st.button("🚀 Process Mapping"):
    st.info("Processing mappings using text similarity...")
    
    try:
        # Get field lists
        rag_fields = rag_df['fields'].astype(str).tolist()
        prov_columns = prov_df.columns.astype(str).tolist()

        results = []
        mapping_explanations = []

        # Show mapping preview
        st.subheader("📋 Field Mappings Preview")
        mapping_preview = []
        
        for col in prov_columns:
            best_idx, similarity = find_best_match(col, rag_fields)
            best_match = rag_df.iloc[best_idx]
            xml_path = best_match['xml field']
            
            mapping_preview.append({
                'Provider Field': col,
                'XML Field': xml_path,
                'RAG Field': best_match['fields'],
                'Similarity': f"{similarity:.2%}"
            })
        
        preview_df = pd.DataFrame(mapping_preview)
        st.dataframe(preview_df)

        progress_bar = st.progress(0)
        total_rows = len(prov_df)

        for i, row in prov_df.iterrows():
            entry = {}
            explain = {}
            
            for col in prov_df.columns:
                # Find best match using text similarity
                best_idx, similarity = find_best_match(col, rag_fields)
                best_match = rag_df.iloc[best_idx]
                xml_path = best_match['xml field']
                value = row[col]
                entry[xml_path] = value
                
                # Generate simple explanation
                explanation = generate_simple_explanation(col, xml_path, similarity)
                explain[col] = explanation
                
            results.append(entry)
            mapping_explanations.append(explain)
            
            # Update progress
            progress_bar.progress((i + 1) / total_rows)

        # Convert to XML
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

        st.subheader("📊 Output XMLs")
        xml_strings = []
        for idx, data in enumerate(results):
            xml_str = build_xml(data)
            xml_strings.append(xml_str)
            
            with st.expander(f"Provider {idx+1} - XML Output"):
                st.code(xml_str, language="xml")
                
            with st.expander(f"Provider {idx+1} - Mapping Explanations"):
                for field, explanation in mapping_explanations[idx].items():
                    st.markdown(f"**{field}**: {explanation}")

        # Create downloadable file
        full_xml = "<providers>\n" + "\n".join(xml_strings) + "\n</providers>"
        st.download_button(
            "⬇️ Download Full XML",
            full_xml,
            file_name="providers.xml",
            mime="application/xml"
        )
        
        st.success("✅ Processing completed using text similarity matching!")
        
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
- Uses Python's built-in text similarity algorithms
- No external API calls required
- Matches provider fields to XML fields based on text similarity
- Provides similarity scores for each mapping
- Works offline and is completely free
""")
