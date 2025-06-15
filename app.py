import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
import io
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
import traceback
from difflib import SequenceMatcher

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Config ---
class Config:
    MAX_PREVIEW_RECORDS = 5
    XML_INDENT = "  "
    SUPPORTED_FORMATS = ["csv", "xlsx", "xls"]
    DEFAULT_ENCODING = "utf-8"
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    SIMILARITY_THRESHOLD = 0.6

# --- API Schemas (Simulated) ---
API_SCHEMAS = {
    "FacilityLoad": {
        "root": "FacilityLoad",
        "fields": [
            "FacilityID", "FacilityName", "Address", "City", "State", "Zip"
        ]
    },
    "PractitionerLoad": {
        "root": "PractitionerLoad",
        "fields": [
            # 50 fields, a mix of direct, derived, and logical mappings
            "NPI", "FirstName", "LastName", "MiddleInitial", "Suffix", "Gender", "DOB", "Degree", "LicenseNumber", "LicenseState",
            "LicenseExpiration", "DEA", "DEAExpiration", "PrimarySpecialty", "SecondarySpecialty", "BoardCertified", "BoardEligible",
            "CAQHID", "CredentialingStatus", "CredentialingDate", "RecredentialingDue", "MedicareEnrolled", "MedicaidEnrolled",
            "PTAN", "MedicaidProviderID", "PECOSDate", "OptOutDate", "AcceptingNewPatients", "TelemedicineEnabled",
            "OfficeHours", "AgeRange", "Address1", "Address2", "City", "State", "Zip", "County", "Phone", "Fax", "Email",
            "Website", "ContractStatus", "ContractEffective", "ContractTermination", "ParticipatingStatus", "CapitationRate",
            "FeeSchedule", "CopayAmount", "DeductibleAmount", "ReimbursementRate", "RiskAdjustmentFactor"
        ]
    },
    "MemberLoad": {
        "root": "MemberLoad",
        "fields": [
            "MemberID", "FirstName", "LastName", "DOB", "Plan"
        ]
    }
}

# --- Field Derivation Logic for PractitionerLoad ---
def derive_practitioner_fields(row: pd.Series) -> Dict[str, Any]:
    # This is a demonstration. In real use, expand logic as needed.
    def safe(val, default=""):
        return str(val) if pd.notna(val) else default

    # Example of derived/complex logic:
    office_hours = "; ".join([
        f"Mon:{safe(row.get('office_hours_monday'))}",
        f"Tue:{safe(row.get('office_hours_tuesday'))}",
        f"Wed:{safe(row.get('office_hours_wednesday'))}",
        f"Thu:{safe(row.get('office_hours_thursday'))}",
        f"Fri:{safe(row.get('office_hours_friday'))}",
        f"Sat:{safe(row.get('office_hours_saturday'))}",
        f"Sun:{safe(row.get('office_hours_sunday'))}"
    ])
    age_range = f"{safe(row.get('age_range_min'))}-{safe(row.get('age_range_max'))}"

    return {
        "NPI": safe(row.get("npi")),
        "FirstName": safe(row.get("first_name")),
        "LastName": safe(row.get("last_name")),
        "MiddleInitial": safe(row.get("middle_initial")),
        "Suffix": safe(row.get("suffix")),
        "Gender": safe(row.get("gender")),
        "DOB": safe(row.get("date_of_birth")),
        "Degree": safe(row.get("degree")),
        "LicenseNumber": safe(row.get("license_number")),
        "LicenseState": safe(row.get("license_state")),
        "LicenseExpiration": safe(row.get("license_expiration")),
        "DEA": safe(row.get("dea_number")),
        "DEAExpiration": safe(row.get("dea_expiration")),
        "PrimarySpecialty": safe(row.get("primary_specialty")),
        "SecondarySpecialty": safe(row.get("secondary_specialty")),
        "BoardCertified": safe(row.get("board_certified")),
        "BoardEligible": safe(row.get("board_eligible")),
        "CAQHID": safe(row.get("caqh_id")),
        "CredentialingStatus": safe(row.get("credentialing_status")),
        "CredentialingDate": safe(row.get("credentialing_date")),
        "RecredentialingDue": safe(row.get("recredentialing_due")),
        "MedicareEnrolled": safe(row.get("medicare_enrolled")),
        "MedicaidEnrolled": safe(row.get("medicaid_enrolled")),
        "PTAN": safe(row.get("ptan_number")),
        "MedicaidProviderID": safe(row.get("medicaid_provider_id")),
        "PECOSDate": safe(row.get("pecos_enrollment_date")),
        "OptOutDate": safe(row.get("opt_out_date")),
        "AcceptingNewPatients": safe(row.get("accepting_new_patients")),
        "TelemedicineEnabled": safe(row.get("telemedicine_enabled")),
        "OfficeHours": office_hours,
        "AgeRange": age_range,
        "Address1": safe(row.get("address_line_1")),
        "Address2": safe(row.get("address_line_2")),
        "City": safe(row.get("city")),
        "State": safe(row.get("state")),
        "Zip": safe(row.get("zip_code")),
        "County": safe(row.get("county")),
        "Phone": safe(row.get("phone_number")),
        "Fax": safe(row.get("fax_number")),
        "Email": safe(row.get("email")),
        "Website": safe(row.get("website")),
        "ContractStatus": safe(row.get("contract_status")),
        "ContractEffective": safe(row.get("contract_effective_date")),
        "ContractTermination": safe(row.get("contract_termination_date")),
        "ParticipatingStatus": safe(row.get("participating_status")),
        "CapitationRate": safe(row.get("capitation_rate")),
        "FeeSchedule": safe(row.get("fee_schedule")),
        "CopayAmount": safe(row.get("copay_amount")),
        "DeductibleAmount": safe(row.get("deductible_amount")),
        "ReimbursementRate": safe(row.get("reimbursement_rate")),
        "RiskAdjustmentFactor": safe(row.get("risk_adjustment_factor")),
    }

def generate_api_xml(api_name: str, data: Dict[str, Any], pretty: bool = True) -> str:
    schema = API_SCHEMAS[api_name]
    root = ET.Element(schema["root"])
    for field in schema["fields"]:
        ET.SubElement(root, field).text = str(data.get(field, ""))
    rough_string = ET.tostring(root, encoding='unicode')
    if pretty:
        reparsed = minidom.parseString(rough_string)
        pretty_xml_str = reparsed.toprettyxml(indent=Config.XML_INDENT)
        lines = pretty_xml_str.split('\n')[1:]
        return '\n'.join(lines).strip()
    else:
        return rough_string

class ColumnDetector:
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    @classmethod
    def auto_map_fields(cls, api_fields: List[str], provider_columns: List[str]) -> Tuple[Dict[str, str], Dict[str, float]]:
        mapping = {}
        confidence = {}
        for api_field in api_fields:
            best_match = None
            best_score = 0
            for prov_col in provider_columns:
                score = cls.calculate_similarity(api_field, prov_col)
                if score > best_score:
                    best_score = score
                    best_match = prov_col
            if best_score >= Config.SIMILARITY_THRESHOLD:
                mapping[api_field] = best_match
                confidence[api_field] = best_score
            else:
                mapping[api_field] = None
                confidence[api_field] = best_score
        return mapping, confidence

def load_custom_css():
    try:
        st.markdown("""
        <style>
        .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .main-header h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .main-header p { font-size: 1.1rem; margin: 0.5rem 0; opacity: 0.95; }
        .main-header small { font-size: 0.9rem; display: block; margin-top: 1rem; color: #e8e8e8; opacity: 0.8; }
        .xml-preview { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border: 2px solid #dee2e6; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; font-family: 'Courier New', monospace; font-size: 0.9rem; white-space: pre-wrap; overflow-x: auto; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1); max-height: 400px; overflow-y: auto; }
        .stats-container { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center; }
        .warning-box { background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
        .success-box { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
        .error-box { background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
        .info-box { background: linear-gradient(135deg, #17a2b8 0%, #138496 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
        .feature-list { margin: 1rem 0 1.5rem 0; padding-left: 1.2rem; font-size: 1rem; }
        .feature-list li { margin-bottom: 0.3rem; }
        .step-header { background: linear-gradient(135deg, #6c757d 0%, #495057 100%); color: white; padding: 0.8rem 1.2rem; border-radius: 8px; margin: 1.5rem 0 1rem 0; font-weight: 600; }
        .detection-result { background: #f8f9fa; border-left: 4px solid #007bff; padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0; }
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error loading CSS: {e}")
        st.error("CSS loading failed, but the app will continue to work.")

def create_safe_download_button(label: str, data: Any, filename: str, mime_type: str):
    try:
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif hasattr(data, 'getvalue'):
            data_bytes = data.getvalue()
        else:
            data_bytes = data
        return st.download_button(
            label=label,
            data=data_bytes,
            file_name=filename,
            mime=mime_type,
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error creating download: {str(e)}")
        return False

def main():
    try:
        st.set_page_config(
            page_title="Intelligent healthcare data mapper", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        load_custom_css()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
            <div class="main-header">
                <h1>Intelligent healthcare data mapper</h1>
                <p>All purpose data mapping solution</p>
                <small>🗓️ Generated on {now}</small>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("### Features")
        st.markdown("""
        <ul class="feature-list">
            <li>Upload reference mapping (RAG) and provider data</li>
            <li>User selects API (FacilityLoad, PractitionerLoad, MemberLoad)</li>
            <li>Intelligent mapping from provider data to API schema</li>
            <li>Audit report with mapping, logic, and confidence</li>
            <li>Download API XML payload and audit report</li>
            <li>Modern UI and robust error handling</li>
        </ul>
        """, unsafe_allow_html=True)

        with st.sidebar:
            st.header("🛠️ API XML Generator")
            api_name = st.selectbox("Select API to Generate XML", list(API_SCHEMAS.keys()))
            st.header("⚙️ Configuration")
            max_preview = st.slider("Max Preview Records", 1, 10, Config.MAX_PREVIEW_RECORDS)
            pretty_xml = st.checkbox("Pretty Print XML", value=True)
            st.header("📊 Export Options")
            xml_encoding = st.selectbox("XML Encoding", ["UTF-8", "ISO-8859-1"], index=0)

        st.markdown('<div class="step-header">📁 Step 1: Upload Reference Mapping Data (RAG)</div>', unsafe_allow_html=True)
        rag_file = st.file_uploader(
            "Reference Mapping File (CSV)", 
            type=["csv"], 
            key="rag",
            help="Any CSV file with mapping rules"
        )
        st.markdown('<div class="step-header">📄 Step 2: Upload Provider Input Data</div>', unsafe_allow_html=True)
        prov_file = st.file_uploader(
            "Provider Input File", 
            type=Config.SUPPORTED_FORMATS, 
            key="provider",
            help="Provider data to be transformed"
        )
        if not rag_file or not prov_file:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Files Required</h4>
                <p>Please upload both reference mapping and provider data files to continue.</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Load files
        with st.spinner("🔄 Loading files..."):
            rag_df = load_file_with_encoding(rag_file, "csv")
            file_extension = prov_file.name.split('.')[-1].lower()
            prov_df = load_file_with_encoding(prov_file, file_extension)

        # --- Intelligent Mapping to API ---
        api_fields = API_SCHEMAS[api_name]["fields"]
        provider_columns = prov_df.columns.astype(str).tolist()
        field_map, confidence_map = ColumnDetector.auto_map_fields(api_fields, provider_columns)

        # --- Audit Table ---
        audit_rows = []
        for api_field in api_fields:
            mapped_col = field_map[api_field]
            conf = confidence_map[api_field]
            logic = f"Similarity: {conf:.2f}"
            status = "✅" if mapped_col else "❌"
            audit_rows.append({
                "API Field": api_field,
                "Provider Column": mapped_col if mapped_col else "",
                "Confidence": f"{conf:.2f}",
                "Logic Applied": logic,
                "Status": status
            })

        st.markdown('<div class="step-header">🧠 Step 3: Mapping Audit</div>', unsafe_allow_html=True)
        audit_df = pd.DataFrame(audit_rows)
        st.dataframe(audit_df, use_container_width=True)
        avg_conf = np.mean([v for v in confidence_map.values() if v is not None])
        st.info(f"Average mapping confidence: {avg_conf:.2f}")

        # --- XML Generation ---
        st.markdown('<div class="step-header">📦 Step 4: Generate & Download API XML</div>', unsafe_allow_html=True)
        mapped_data = []
        for idx, row in prov_df.iterrows():
            if api_name == "PractitionerLoad":
                api_row = derive_practitioner_fields(row)
            else:
                api_row = {}
                for api_field in api_fields:
                    prov_col = field_map[api_field]
                    api_row[api_field] = row[prov_col] if prov_col and prov_col in row else ""
            mapped_data.append(api_row)

        # Download full XML
        xml_declaration = f'<?xml version="1.0" encoding="{xml_encoding.upper()}"?>\n'
        xml_payloads = [generate_api_xml(api_name, row, pretty=pretty_xml) for row in mapped_data]
        full_xml = xml_declaration + f'<{API_SCHEMAS[api_name]["root"]}s>\n' + '\n'.join(xml_payloads) + f'\n</{API_SCHEMAS[api_name]["root"]}s>'
        create_safe_download_button(
            f"📥 Download {api_name} XML Payload",
            full_xml,
            f"{api_name}_payload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml",
            "application/xml"
        )

        # Download audit
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            audit_df.to_excel(writer, sheet_name='Mapping_Audit', index=False)
        create_safe_download_button(
            "📊 Download Audit Report (Excel)",
            output,
            f"{api_name}_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        logger.error(f"Critical error: {traceback.format_exc()}")
        st.markdown("""
        <div class="error-box">
            <h4>❌ Application Error</h4>
            <p>A critical error occurred. Please check your input files and try again.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
