import streamlit as st
import pandas as pd
import requests
import os
import json
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- Settings: API Templates ---
API_SCHEMAS = {
    "PractitionerLoad": {
        "root": "Practitioners",
        "record": "Practitioner",
        "fields": [
            ("NPI", "Practitioner NPI"),
            ("FirstName", "Practitioner first name"),
            ("LastName", "Practitioner last name"),
            ("DOB", "Date of birth"),
            ("Gender", "Gender"),
            ("Degree", "Degree"),
            ("LicenseNumber", "License number"),
            ("LicenseState", "License state"),
            ("DEA", "DEA number"),
            ("PrimarySpecialty", "Primary specialty"),
            ("SecondarySpecialty", "Secondary specialty"),
            ("CAQH", "CAQH ID"),
            ("CredentialingStatus", "Credentialing status"),
            ("CredentialingDate", "Credentialing date"),
            ("MedicareEnrolled", "Medicare enrolled"),
            ("MedicaidEnrolled", "Medicaid enrolled"),
            ("PTAN", "PTAN number"),
            ("MedicaidProviderID", "Medicaid provider ID"),
            ("AcceptingNewPatients", "Accepting new patients"),
            ("TelemedicineEnabled", "Telemedicine enabled"),
            ("Address1", "Address line 1"),
            ("Address2", "Address line 2"),
            ("City", "City"),
            ("State", "State"),
            ("Zip", "Zip code"),
            ("County", "County"),
            ("Phone", "Phone number"),
            ("Fax", "Fax number"),
            ("Email", "Email"),
            ("Website", "Website"),
            ("ContractStatus", "Contract status"),
            ("ContractEffectiveDate", "Contract effective date"),
            ("ContractTerminationDate", "Contract termination date"),
            ("ParticipatingStatus", "Participating status"),
            ("HospitalAffiliation1", "Hospital affiliation 1"),
            ("HospitalAffiliation2", "Hospital affiliation 2"),
            ("MedicalSchool", "Medical school"),
            ("GraduationYear", "Graduation year"),
            ("ResidencyProgram", "Residency program"),
            ("FellowshipProgram", "Fellowship program"),
            ("LanguagesSpoken", "Languages spoken"),
            ("Race", "Race"),
            ("Ethnicity", "Ethnicity"),
            ("AuthorizedOfficialName", "Authorized official name"),
            ("AuthorizedOfficialTitle", "Authorized official title"),
            ("AuthorizedOfficialPhone", "Authorized official phone"),
            ("RecordStatus", "Record status"),
            ("LastUpdateDate", "Last update date"),
            ("DataSource", "Data source"),
        ]
    },
    "FacilityLoad": {
        "root": "Facilities",
        "record": "Facility",
        "fields": [
            ("FacilityID", "Facility unique ID"),
            ("FacilityName", "Facility name"),
            ("Address", "Address"),
            ("City", "City"),
            ("State", "State"),
            ("Zip", "Zip code"),
            ("Phone", "Phone number"),
            ("Type", "Facility type"),
        ]
    },
    "MemberLoad": {
        "root": "Members",
        "record": "Member",
        "fields": [
            ("MemberID", "Member unique ID"),
            ("FirstName", "Member first name"),
            ("LastName", "Member last name"),
            ("DOB", "Date of birth"),
            ("Gender", "Gender"),
            ("PlanID", "Plan ID"),
        ]
    }
}

def get_hf_token():
    return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

def hf_inference(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_new_tokens=1024, temperature=0.1):
    HF_TOKEN = get_hf_token()
    if not HF_TOKEN:
        st.error("Hugging Face API key not found. Please set HF_TOKEN in Streamlit secrets or environment.")
        st.stop()
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    st.write("HF API response:", response.text)  # Debug output
    if response.status_code != 200:
        st.error(f"HF API returned status {response.status_code}: {response.text}")
        st.stop()
    result = response.json()
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    if isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    if isinstance(result, dict) and "choices" in result:
        return result["choices"][0]["text"]
    return str(result)

def build_xml(records, api_schema, pretty_print=True):
    root = ET.Element(api_schema["root"])
    for record in records:
        rec_elem = ET.SubElement(root, api_schema["record"])
        for field, _ in api_schema["fields"]:
            value = record.get(field, "")
            ET.SubElement(rec_elem, field).text = str(value)
    xml_str = ET.tostring(root, encoding='unicode')
    if pretty_print:
        reparsed = minidom.parseString(xml_str)
        pretty_xml_str = reparsed.toprettyxml(indent="  ")
        lines = pretty_xml_str.split('\n')[1:]
        return '\n'.join(lines).strip()
    return xml_str

st.set_page_config(page_title="Intelligent healthcare data mapper", layout="wide")
st.markdown("""
    <div class="main-header">
        <h1>🧠 Intelligent healthcare data mapper</h1>
        <p>comprehensive data mapping solution</p>
        <small>🤖 LLM + RAG powered mapping (Hugging Face)</small>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    api_choice = st.selectbox("Select API", list(API_SCHEMAS.keys()))
    pretty_xml = st.checkbox("Pretty Print XML", value=True)
    xml_encoding = st.selectbox("XML Encoding", ["UTF-8", "ISO-8859-1"], index=0)

api_schema = API_SCHEMAS[api_choice]

st.markdown('<div class="step-header">📁 Step 1: Upload Reference Mapping File (RAG)</div>', unsafe_allow_html=True)
rag_file = st.file_uploader("Reference Mapping File (CSV/Excel)", type=["csv", "xlsx", "xls"], key="rag")
st.markdown('<div class="step-header">📄 Step 2: Upload Healthcare Data File</div>', unsafe_allow_html=True)
data_file = st.file_uploader("Healthcare Data File (CSV/Excel)", type=["csv", "xlsx", "xls"], key="healthcare_data")

if not rag_file or not data_file:
    st.warning("Please upload both reference mapping and healthcare data files to continue.")
    st.stop()

# --- Load files ---
rag_df = pd.read_csv(rag_file) if rag_file.name.endswith(".csv") else pd.read_excel(rag_file)
data_df = pd.read_csv(data_file) if data_file.name.endswith(".csv") else pd.read_excel(data_file)

# --- LLM+RAG Mapping ---
st.markdown('<div class="step-header">🔍 Step 3: AI Mapping (LLM + RAG)</div>', unsafe_allow_html=True)

def get_llm_mapping(rag_df, data_df, api_schema, api_choice, max_fields=5, max_examples=3):
    xml_fields = [f[0] for f in api_schema["fields"]][:max_fields]
    reference_examples = rag_df.head(max_examples).to_dict(orient="records")
    data_columns = list(data_df.columns)
    prompt = f"""
You are a data mapping expert. Given the following reference mapping table (as examples), a list of healthcare data columns, and a list of target XML fields for the {api_choice} API, map each XML field to the most appropriate healthcare data column. If no good match exists, return null. For each mapping, provide a short explanation of the logic (direct, calculated, rule-based, LLM fallback, etc).

Reference mapping examples (first {max_examples} rows):
{reference_examples}

Healthcare data columns:
{data_columns}

Target XML fields:
{xml_fields}

Return your answer as a valid JSON object, and nothing else, like this:
{{
  "XMLField1": {{"input_field": "healthcare_data_column", "logic": "Direct"}},
  "XMLField2": {{"input_field": null, "logic": "No good match found"}},
  ...
}}
If you can't find a good match, use null for input_field and explain in logic.
"""
    llm_output = hf_inference(prompt)
    return llm_output

if "llm_output" not in st.session_state or st.button("Retry Mapping"):
    with st.spinner("Using LLM to map your healthcare data columns to XML fields..."):
        llm_output = get_llm_mapping(rag_df, data_df, api_schema, api_choice)
        st.session_state["llm_output"] = llm_output
else:
    llm_output = st.session_state["llm_output"]

st.markdown("#### LLM Output (for debugging)")
st.code(llm_output, language="json")

# Try to extract JSON from the output
try:
    mapping = json.loads(llm_output[llm_output.find("{"):llm_output.rfind("}")+1])
except Exception:
    st.error("Could not parse mapping from LLM output. Please review the output above, reduce the number of fields/examples, and click 'Retry Mapping'.")
    st.stop()

# --- Review Mapping Screen ---
st.markdown('<div class="step-header">🧐 Step 4: Review & Edit Mapping</div>', unsafe_allow_html=True)
mapping_table = []
for xml_field, desc in api_schema["fields"]:
    map_info = mapping.get(xml_field, {})
    input_field = map_info.get("input_field") if isinstance(map_info, dict) else None
    logic = map_info.get("logic", "") if isinstance(map_info, dict) else ""
    status = "✅" if input_field else "❌"
    mapping_table.append({
        "XML Field": xml_field,
        "Description": desc,
        "Mapped Input Field": input_field if input_field else "",
        "Logic": logic,
        "Status": status
    })
mapping_df = pd.DataFrame(mapping_table)
edited_df = st.data_editor(mapping_df, use_container_width=True, num_rows="dynamic", key="mapping_editor")

if any(edited_df["Mapped Input Field"] == ""):
    st.warning("Some XML fields are not mapped. Please review and edit as needed.")

st.markdown('<div class="step-header">📦 Step 5: Download Results</div>', unsafe_allow_html=True)
if st.button("Generate & Download XML and Audit Report"):
    records = []
    for _, row in data_df.iterrows():
        rec = {}
        for _, map_row in edited_df.iterrows():
            xml_field = map_row["XML Field"]
            input_field = map_row["Mapped Input Field"]
            rec[xml_field] = row[input_field] if input_field in row else ""
        records.append(rec)
    xml_str = build_xml(records, api_schema, pretty_print=pretty_xml)
    xml_declaration = f'<?xml version="1.0" encoding="{xml_encoding.upper()}"?>\n'
    full_xml = xml_declaration + xml_str
    st.download_button(
        "📥 Download Complete XML File",
        full_xml,
        f"{api_choice.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml",
        "application/xml"
    )
    st.download_button(
        "📊 Download Audit Table (CSV)",
        edited_df.to_csv(index=False),
        f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )
