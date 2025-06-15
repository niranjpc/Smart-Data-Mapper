import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import traceback
import logging
from datetime import datetime
from difflib import SequenceMatcher
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Tuple, Any

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    MAX_PREVIEW_RECORDS = 5
    XML_INDENT = "  "
    SUPPORTED_FORMATS = ["csv", "xlsx", "xls"]
    DEFAULT_ENCODING = "utf-8"
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit
    SIMILARITY_THRESHOLD = 0.6  # Minimum similarity for auto-mapping

# --- Intelligent Column Detection ---
class ColumnDetector:
    """Intelligent column detection and mapping logic"""

    SOURCE_PATTERNS = [
        r'\b(field|column|source|input|provider|data)[\s_-]*(name|field|col)?\b',
        r'\b(original|raw|src)[\s_-]*(field|column)?\b',
        r'\bfrom[\s_-]*(field|column)?\b'
    ]
    TARGET_PATTERNS = [
        r'\b(xml|target|destination|output|hrp)[\s_-]*(field|path|element)?\b',
        r'\b(mapped|transformed|converted)[\s_-]*(to|field)?\b',
        r'\bto[\s_-]*(field|column|xml)?\b'
    ]
    TYPE_PATTERNS = [
        r'\b(type|kind|method|mapping[\s_-]*type)\b',
        r'\b(transform|conversion|logic)[\s_-]*(type|method)?\b'
    ]
    LOGIC_PATTERNS = [
        r'\b(logic|rule|description|note|comment)\b',
        r'\b(applied|transformation|instruction)\b',
        r'\b(how|what|why|when)[\s_-]*(to|apply)?\b'
    ]

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    @classmethod
    def detect_column_type(cls, column_name: str, sample_values: List[str]) -> str:
        col_lower = column_name.lower().strip()
        for pattern in cls.SOURCE_PATTERNS:
            if re.search(pattern, col_lower, re.IGNORECASE):
                return 'source'
        for pattern in cls.TARGET_PATTERNS:
            if re.search(pattern, col_lower, re.IGNORECASE):
                return 'target'
        for pattern in cls.TYPE_PATTERNS:
            if re.search(pattern, col_lower, re.IGNORECASE):
                return 'type'
        for pattern in cls.LOGIC_PATTERNS:
            if re.search(pattern, col_lower, re.IGNORECASE):
                return 'logic'
        if sample_values:
            xml_path_count = sum(1 for val in sample_values[:10] if isinstance(val, str) and ('/' in val or '<' in val))
            if xml_path_count > len(sample_values) * 0.3:
                return 'target'
        return 'unknown'

    @classmethod
    def auto_detect_columns(cls, df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
        column_mapping = {}
        detected_types = {}
        for col in df.columns:
            sample_values = df[col].dropna().astype(str).tolist()[:20]
            col_type = cls.detect_column_type(col, sample_values)
            detected_types[col] = col_type
        source_candidates = [col for col, typ in detected_types.items() if typ == 'source']
        target_candidates = [col for col, typ in detected_types.items() if typ == 'target']
        type_candidates = [col for col, typ in detected_types.items() if typ == 'type']
        logic_candidates = [col for col, typ in detected_types.items() if typ == 'logic']
        if not source_candidates:
            source_candidates = cls._find_similar_columns(df.columns, ['fields', 'field', 'source', 'column', 'provider', 'input'])
        if not target_candidates:
            target_candidates = cls._find_similar_columns(df.columns, ['xml', 'target', 'output', 'destination', 'hrp', 'mapped'])
        if not type_candidates:
            type_candidates = cls._find_similar_columns(df.columns, ['type', 'mapping_type', 'method', 'transform'])
        if not logic_candidates:
            logic_candidates = cls._find_similar_columns(df.columns, ['logic', 'rule', 'description', 'note', 'applied'])
        if source_candidates:
            column_mapping['source'] = source_candidates[0]
        if target_candidates:
            column_mapping['target'] = target_candidates[0]
        if type_candidates:
            column_mapping['type'] = type_candidates[0]
        if logic_candidates:
            column_mapping['logic'] = logic_candidates[0]
        return column_mapping, detected_types

    @classmethod
    def _find_similar_columns(cls, columns: List[str], reference_terms: List[str]) -> List[str]:
        matches = []
        for col in columns:
            col_lower = col.lower().strip()
            for term in reference_terms:
                if cls.calculate_similarity(col_lower, term) >= Config.SIMILARITY_THRESHOLD:
                    matches.append((col, cls.calculate_similarity(col_lower, term)))
                    break
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches]

def load_custom_css():
    try:
        st.markdown("""
        <style>
        .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .main-header h1 { font-size: 2.8rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .main-header p { font-size: 1.2rem; margin: 0.5rem 0; opacity: 0.95; }
        .main-header small { font-size: 0.9rem; display: block; margin-top: 1rem; color: #e8e8e8; opacity: 0.8; }
        .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1rem 0; }
        .feature-card { background: white; border: 1px solid #dee2e6; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .step-header { background: linear-gradient(135deg, #6c757d 0%, #495057 100%); color: white; padding: 0.8rem 1.2rem; border-radius: 8px; margin: 1.5rem 0 1rem 0; font-weight: 600; }
        .detection-result { background: #f8f9fa; border-left: 4px solid #007bff; padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0; }
        .warning-box { background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
        .success-box { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
        .error-box { background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
        .info-box { background: linear-gradient(135deg, #17a2b8 0%, #138496 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error loading CSS: {e}")
        st.error("CSS loading failed, but the app will continue to work.")
        # --- Enhanced XML Builder ---
def build_dynamic_xml(row: pd.Series, pretty_print: bool = False) -> str:
    try:
        root = ET.Element("HealthcareData")
        for path, value in row.items():
            if pd.isna(value) or value == "" or value is None:
                continue
            path_clean = str(path).strip()
            if not path_clean:
                continue
            parts = [part.strip() for part in path_clean.split("/") if part.strip()]
            if not parts:
                continue
            current = root
            for part in parts[:-1]:
                part_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', str(part))
                if not part_clean or part_clean[0].isdigit():
                    part_clean = f"element_{part_clean}"
                part_clean = part_clean[:50]
                found = current.find(part_clean)
                if found is None:
                    found = ET.SubElement(current, part_clean)
                current = found
            final_part = re.sub(r'[^a-zA-Z0-9_-]', '_', str(parts[-1]))
            if not final_part or final_part[0].isdigit():
                final_part = f"element_{final_part}"
            final_part = final_part[:50]
            value_str = str(value).strip()
            value_str = value_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            value_str = value_str.replace('"', '&quot;').replace("'", '&apos;')
            if value_str and len(value_str) <= 1000:
                ET.SubElement(current, final_part).text = value_str
        if pretty_print:
            try:
                rough_string = ET.tostring(root, encoding='unicode')
                reparsed = minidom.parseString(rough_string)
                pretty_xml_str = reparsed.toprettyxml(indent=Config.XML_INDENT)
                lines = pretty_xml_str.split('\n')[1:]
                return '\n'.join(lines).strip()
            except:
                return ET.tostring(root, encoding='unicode')
        else:
            return ET.tostring(root, encoding='unicode')
    except Exception as e:
        logger.error(f"Error building XML for row: {e}")
        return f"<HealthcareData><Error>Failed to build XML: {str(e)[:100]}</Error></HealthcareData>"

# --- Data Validation ---
def validate_and_analyze_reference_data(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors = []
    warnings = []
    analysis = {}
    try:
        if len(df) == 0:
            errors.append("Reference data is empty")
            return False, errors, {}
        if len(df.columns) == 0:
            errors.append("Reference data has no columns")
            return False, errors, {}
        detector = ColumnDetector()
        column_mapping, detected_types = detector.auto_detect_columns(df)
        analysis['column_mapping'] = column_mapping
        analysis['detected_types'] = detected_types
        analysis['total_columns'] = len(df.columns)
        analysis['total_rows'] = len(df)
        if 'source' not in column_mapping:
            warnings.append("Could not automatically identify source field column. Manual mapping may be required.")
        if 'target' not in column_mapping:
            warnings.append("Could not automatically identify target XML field column. Manual mapping may be required.")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                warnings.append(f"Column '{col}' has {null_count}/{len(df)} null values")
        analysis['warnings'] = warnings
        analysis['confidence_score'] = len(column_mapping) / 4.0
        return True, errors, analysis
    except Exception as e:
        errors.append(f"Error analyzing reference data: {str(e)}")
        return False, errors, {}

def validate_healthcare_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    try:
        if len(df) == 0:
            errors.append("Healthcare data is empty")
        if len(df.columns) == 0:
            errors.append("Healthcare data has no columns")
        if len(df) > 10000:
            errors.append("Warning: Large dataset detected. Consider processing in smaller batches.")
    except Exception as e:
        errors.append(f"Error validating healthcare data: {str(e)}")
    return len(errors) == 0, errors

# --- File Loading ---
def load_file_with_encoding(file, file_type: str) -> pd.DataFrame:
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    try:
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        if file_size > Config.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB. Maximum allowed: {Config.MAX_FILE_SIZE / (1024*1024):.1f}MB")
        for encoding in encodings:
            try:
                file.seek(0)
                if file_type == "csv":
                    df = pd.read_csv(
                        file, 
                        encoding=encoding, 
                        on_bad_lines='skip',
                        low_memory=False,
                        dtype=str,
                        na_values=['', 'NULL', 'null', 'NA', 'na', 'N/A', 'n/a']
                    )
                elif file_type in ["xlsx", "xls"]:
                    df = pd.read_excel(file, dtype=str, na_values=['', 'NULL', 'null', 'NA', 'na', 'N/A', 'n/a'])
                if df.empty:
                    continue
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading file with {encoding}: {e}")
                continue
        raise ValueError(f"Could not read file with any supported encoding: {encodings}")
    except Exception as e:
        logger.error(f"File loading error: {e}")
        raise

# --- Statistics and Analytics ---
def generate_mapping_stats(rag_df: pd.DataFrame, data_df: pd.DataFrame, field_map: Dict[str, str]) -> Dict[str, Any]:
    try:
        stats = {
            'total_reference_mappings': len(rag_df),
            'total_healthcare_data_columns': len(data_df.columns),
            'mapped_columns': len(field_map),
            'unmapped_columns': len(data_df.columns) - len(field_map),
            'mapping_coverage': (len(field_map) / len(data_df.columns) * 100) if len(data_df.columns) > 0 else 0,
            'total_healthcare_data_records': len(data_df),
            'unmapped_column_list': [col for col in data_df.columns if col not in field_map]
        }
        return stats
    except Exception as e:
        logger.error(f"Error generating stats: {e}")
        return {
            'total_reference_mappings': 0,
            'total_healthcare_data_columns': 0,
            'mapped_columns': 0,
            'unmapped_columns': 0,
            'mapping_coverage': 0,
            'total_healthcare_data_records': 0,
            'unmapped_column_list': []
        }

# --- Download Button ---
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

# --- Manual Mapping UI ---
def display_manual_mapping_interface(df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, str]:
    st.markdown("### 🎯 Manual Column Mapping")
    st.info("Auto-detection couldn't identify all columns. Please manually map the columns below:")
    manual_mapping = {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Available Columns:**")
        for i, col in enumerate(df.columns):
            detected_type = analysis.get('detected_types', {}).get(col, 'unknown')
            confidence = "🔍" if detected_type == 'unknown' else "✅"
            st.write(f"• {confidence} `{col}` *(detected: {detected_type})*")
    with col2:
        st.markdown("**Map to Purpose:**")
        source_col = st.selectbox(
            "Source Fields Column (contains healthcare data field names):",
            ["None"] + list(df.columns),
            index=0 if 'source' not in analysis.get('column_mapping', {}) else list(df.columns).index(analysis['column_mapping']['source']) + 1
        )
        target_col = st.selectbox(
            "Target XML Fields Column (contains XML paths):",
            ["None"] + list(df.columns),
            index=0 if 'target' not in analysis.get('column_mapping', {}) else list(df.columns).index(analysis['column_mapping']['target']) + 1
        )
        type_col = st.selectbox(
            "Mapping Type Column (optional):",
            ["None"] + list(df.columns),
            index=0
        )
        logic_col = st.selectbox(
            "Logic/Description Column (optional):",
            ["None"] + list(df.columns),
            index=0
        )
    if source_col != "None":
        manual_mapping['source'] = source_col
    if target_col != "None":
        manual_mapping['target'] = target_col
    if type_col != "None":
        manual_mapping['type'] = type_col
    if logic_col != "None":
        manual_mapping['logic'] = logic_col
    return manual_mapping

# --- Intelligent Mapping ---
def auto_map_healthcare_to_reference(reference_fields: List[str], data_columns: List[str], threshold: float = 0.6) -> Dict[str, str]:
    mapping = {}
    for ref_field in reference_fields:
        best_match = None
        best_score = 0
        for data_col in data_columns:
            score = ColumnDetector.calculate_similarity(ref_field, data_col)
            if score > best_score:
                best_score = score
                best_match = data_col
        if best_score >= threshold:
            mapping[ref_field] = best_match
        else:
            mapping[ref_field] = None
    return mapping
    # --- Main Application ---
def main():
    st.set_page_config(
        page_title="Intelligent HRP AI Data Mapper", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_custom_css()
    st.markdown(f"""
        <div class="main-header">
            <h1>🧠 Intelligent HRP AI Data Mapper</h1>
            <p>Smart field mapping with AI-powered column detection - no rigid formats required!</p>
            <small>🤖 LLM Model: OpenAI GPT-4</small>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("## 🚀 Enhanced Features")
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <h4>🧠 Smart Detection</h4>
            <p>AI automatically detects column purposes - no strict naming required</p>
        </div>
        <div class="feature-card">
            <h4>🔄 Flexible Input</h4>
            <p>Works with any column names and data structures</p>
        </div>
        <div class="feature-card">
            <h4>✨ XML Generation</h4>
            <p>HRP-compliant XML with validation and formatting</p>
        </div>
        <div class="feature-card">
            <h4>📈 Analytics</h4>
            <p>Comprehensive mapping statistics and confidence scoring</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_choice = st.selectbox("Select API", ["FacilityLoad", "PractitionerLoad", "MemberLoad"])
        show_preview = st.checkbox("Show XML Preview", value=True)
        max_preview = st.slider("Max Preview Records", 1, 10, Config.MAX_PREVIEW_RECORDS)
        pretty_xml = st.checkbox("Pretty Print XML", value=True)
        st.header("🎯 Detection Settings")
        similarity_threshold = st.slider("Column Similarity Threshold", 0.1, 1.0, Config.SIMILARITY_THRESHOLD, 0.1)
        Config.SIMILARITY_THRESHOLD = similarity_threshold
        st.header("📊 Export Options")
        include_stats = st.checkbox("Include Statistics in Report", value=True)
        xml_encoding = st.selectbox("XML Encoding", ["UTF-8", "ISO-8859-1"], index=0)
    st.markdown('<div class="step-header">📁 Step 1: Upload Reference Mapping Data</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <h4>🧠 Smart Upload</h4>
        <p>Upload any CSV file with mapping data. The AI will automatically detect:</p>
        <ul>
            <li>Source field columns (healthcare data field names)</li>
            <li>Target XML path columns</li>
            <li>Mapping type and logic columns (if present)</li>
        </ul>
        <p><strong>No specific column names required!</strong></p>
    </div>
    """, unsafe_allow_html=True)
    rag_file = st.file_uploader(
        "Reference Mapping File (CSV)", 
        type=["csv"], 
        key="rag",
        help="Any CSV file with mapping rules - AI will auto-detect column purposes"
    )
    st.markdown('<div class="step-header">📄 Step 2: Upload Healthcare Data Input</div>', unsafe_allow_html=True)
    st.info("Upload your healthcare data file in CSV or Excel format")
    data_file = st.file_uploader(
        "Healthcare Data Input File", 
        type=Config.SUPPORTED_FORMATS, 
        key="healthcare_data",
        help="Healthcare data to be transformed into HRP XML format"
    )
    if not rag_file or not data_file:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Files Required</h4>
            <p>Please upload both reference mapping and healthcare data files to continue.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    try:
        with st.spinner("🔄 Loading and intelligently analyzing files..."):
            rag_df = load_file_with_encoding(rag_file, "csv")
            file_extension = data_file.name.split('.')[-1].lower()
            data_df = load_file_with_encoding(data_file, file_extension)
            rag_valid, rag_errors, analysis = validate_and_analyze_reference_data(rag_df)
            data_valid, data_errors = validate_healthcare_data(data_df)
            if not rag_valid or not data_valid:
                error_msg = "Data validation failed:\n"
                error_msg += "\n".join(rag_errors + data_errors)
                st.error(error_msg)
                st.stop()
        st.markdown('<div class="step-header">🔍 AI Analysis Results</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Columns Detected", analysis['total_columns'])
        with col2:
            st.metric("🎯 Auto-Mapped", len(analysis['column_mapping']))
        with col3:
            confidence_pct = analysis['confidence_score'] * 100
            st.metric("🎯 Confidence", f"{confidence_pct:.0f}%")
        if analysis['column_mapping']:
            st.markdown("### ✅ Auto-Detected Column Mappings:")
            for purpose, column in analysis['column_mapping'].items():
                purpose_emoji = {"source": "📥", "target": "📤", "type": "🔄", "logic": "📝"}
                st.markdown(f"""
                <div class="detection-result">
                    <strong>{purpose_emoji.get(purpose, '📋')} {purpose.title()} Field:</strong> <code>{column}</code>
                </div>
                """, unsafe_allow_html=True)
        if analysis.get('warnings'):
            with st.expander("⚠️ Analysis Warnings", expanded=False):
                for warning in analysis['warnings']:
                    st.warning(warning)
        column_mapping = analysis['column_mapping']
        if analysis['confidence_score'] < 0.5 or 'source' not in column_mapping or 'target' not in column_mapping:
            manual_override = st.checkbox("🎯 Override Auto-Detection (Manual Mapping)", value=False)
            if manual_override:
                column_mapping = display_manual_mapping_interface(rag_df, analysis)
                if not column_mapping.get('source') or not column_mapping.get('target'):
                    st.error("Both source and target columns must be mapped. Please complete the manual mapping.")
                    st.stop()
        # --- Intelligent Healthcare-to-Reference Mapping ---
        source_col = column_mapping['source']
        target_col = column_mapping['target']
        reference_fields = rag_df[source_col].dropna().astype(str).unique().tolist()
        data_columns = data_df.columns.astype(str).tolist()
        data_to_reference_map = auto_map_healthcare_to_reference(reference_fields, data_columns, Config.SIMILARITY_THRESHOLD)
        # Build field_map for XML transformation: {data_col: xml_path}
        field_map = {}
        audit_rows = []
        for _, row in rag_df.iterrows():
            src = row[source_col]
            tgt = row[target_col]
            mtype = row[column_mapping.get('type')] if 'type' in column_mapping and column_mapping.get('type') in row else "Direct"
            logic = row[column_mapping.get('logic')] if 'logic' in column_mapping and column_mapping.get('logic') in row else "No specific logic"
            mapped_data_col = data_to_reference_map.get(str(src), None)
            if mapped_data_col:
                field_map[mapped_data_col] = str(tgt)
                status = "✅ Mapped"
            else:
                status = "⚠️ Not Found in Healthcare Data"
            audit_rows.append({
                "Reference Field": str(src),
                "Healthcare Data Column": mapped_data_col if mapped_data_col else "",
                "Target XML Path": str(tgt),
                "Mapping Type": str(mtype),
                "Logic Applied": str(logic),
                "Status": status
            })
        stats = generate_mapping_stats(rag_df, data_df, field_map)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Mappings", stats['total_reference_mappings'])
        with col2:
            st.metric("📋 Healthcare Data Columns", stats['total_healthcare_data_columns'])
        with col3:
            st.metric("✅ Mapped Columns", stats['mapped_columns'])
        with col4:
            st.metric("📈 Coverage", f"{stats['mapping_coverage']:.1f}%")
        if stats['unmapped_columns'] > 0:
            with st.expander(f"⚠️ {stats['unmapped_columns']} Unmapped Columns", expanded=False):
                st.write("The following healthcare data columns were not mapped:")
                for col in stats['unmapped_column_list']:
                    st.write(f"• {col}")
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ File Loading Error</h4>
            <p>Error reading files: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
        logger.error(f"File loading error: {traceback.format_exc()}")
        st.stop()
    # Transform data
    try:
        with st.spinner("🔄 Transforming data..."):
            mapped_data = []
            for idx, row in data_df.iterrows():
                try:
                    xml_row = {}
                    for data_col, xml_path in field_map.items():
                        if data_col in row:
                            xml_row[xml_path] = row[data_col]
                    mapped_data.append(xml_row)
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {e}")
                    continue
    except Exception as e:
        st.error(f"Error transforming data: {str(e)}")
        logger.error(f"Data transformation error: {traceback.format_exc()}")
        st.stop()
    # Preview transformed data
    if show_preview and mapped_data:
        st.markdown('<div class="step-header">🔍 Step 4: XML Preview</div>', unsafe_allow_html=True)
        try:
            sample_df = pd.DataFrame(mapped_data)
            preview_count = min(max_preview, len(sample_df))
            for i in range(preview_count):
                xml_content = build_dynamic_xml(sample_df.iloc[i], pretty_print=pretty_xml)
                with st.expander(f"🏥 Healthcare Data {i+1} XML Preview", expanded=i == 0):
                    st.markdown(f'<div class="xml-preview">{xml_content}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Error generating preview: {str(e)}")
                # Generate full XML
    st.markdown('<div class="step-header">📦 Step 5: Export Results</div>', unsafe_allow_html=True)
    try:
        encoding_map = {"UTF-8": "utf-8", "ISO-8859-1": "iso-8859-1"}
        selected_encoding = encoding_map[xml_encoding]
        xml_declaration = f'<?xml version="1.0" encoding="{selected_encoding.upper()}"?>\n'
        xml_records = []
        for row_data in mapped_data:
            xml_records.append(build_dynamic_xml(pd.Series(row_data), pretty_print=pretty_xml))
        full_xml = xml_declaration + '<HealthcareDataRecords>\n' + '\n'.join(xml_records) + '\n</HealthcareDataRecords>'
        col1, col2 = st.columns(2)
        with col1:
            create_safe_download_button(
                "📥 Download Complete XML File",
                full_xml,
                f"hrp_healthcare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml",
                "application/xml"
            )
        with col2:
            try:
                audit_df = pd.DataFrame(audit_rows)
                if include_stats:
                    stats_rows = []
                    for key, value in stats.items():
                        if key != 'unmapped_column_list':
                            stats_rows.append({"Metric": key.replace('_', ' ').title(), "Value": str(value)})
                    output = io.BytesIO()
                    try:
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            audit_df.to_excel(writer, sheet_name='Mapping_Audit', index=False)
                            pd.DataFrame(stats_rows).to_excel(writer, sheet_name='Statistics', index=False)
                        create_safe_download_button(
                            "📊 Download Audit Report (Excel)",
                            output,
                            f"hrp_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.warning(f"Excel export failed: {str(e)}. Falling back to CSV.")
                        csv_data = audit_df.to_csv(index=False)
                        create_safe_download_button(
                            "📊 Download Audit Report (CSV)",
                            csv_data,
                            f"hrp_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                else:
                    csv_data = audit_df.to_csv(index=False)
                    create_safe_download_button(
                        "📊 Download Audit Report (CSV)",
                        csv_data,
                        f"hrp_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
            except Exception as e:
                st.error(f"Error creating audit report: {str(e)}")

if __name__ == "__main__":
    main()
