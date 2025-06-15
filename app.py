import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import logging
from datetime import datetime
import io
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
import traceback
from difflib import SequenceMatcher

# --- Enhanced Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    
    # Common patterns for source field identification
    SOURCE_PATTERNS = [
        r'\b(field|column|source|input|provider|data)[\s_-]*(name|field|col)?\b',
        r'\b(original|raw|src)[\s_-]*(field|column)?\b',
        r'\bfrom[\s_-]*(field|column)?\b'
    ]
    
    # Common patterns for target/XML field identification
    TARGET_PATTERNS = [
        r'\b(xml|target|destination|output|hrp)[\s_-]*(field|path|element)?\b',
        r'\b(mapped|transformed|converted)[\s_-]*(to|field)?\b',
        r'\bto[\s_-]*(field|column|xml)?\b'
    ]
    
    # Common patterns for mapping type identification
    TYPE_PATTERNS = [
        r'\b(type|kind|method|mapping[\s_-]*type)\b',
        r'\b(transform|conversion|logic)[\s_-]*(type|method)?\b'
    ]
    
    # Common patterns for logic/description identification
    LOGIC_PATTERNS = [
        r'\b(logic|rule|description|note|comment)\b',
        r'\b(applied|transformation|instruction)\b',
        r'\b(how|what|why|when)[\s_-]*(to|apply)?\b'
    ]
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    @classmethod
    def detect_column_type(cls, column_name: str, sample_values: List[str]) -> str:
        """Detect what type of column this is based on name and content"""
        col_lower = column_name.lower().strip()
        
        # Check against patterns
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
        
        # Analyze content for additional clues
        if sample_values:
            # Check if values look like XML paths
            xml_path_count = sum(1 for val in sample_values[:10] if isinstance(val, str) and ('/' in val or '<' in val))
            if xml_path_count > len(sample_values) * 0.3:  # 30% of samples look like XML paths
                return 'target'
        
        return 'unknown'
    
    @classmethod
    def auto_detect_columns(cls, df: pd.DataFrame) -> Dict[str, str]:
        """Automatically detect column purposes in the reference data"""
        column_mapping = {}
        detected_types = {}
        
        for col in df.columns:
            sample_values = df[col].dropna().astype(str).tolist()[:20]  # Get sample values
            col_type = cls.detect_column_type(col, sample_values)
            detected_types[col] = col_type
        
        # Find best matches for each type
        source_candidates = [col for col, typ in detected_types.items() if typ == 'source']
        target_candidates = [col for col, typ in detected_types.items() if typ == 'target']
        type_candidates = [col for col, typ in detected_types.items() if typ == 'type']
        logic_candidates = [col for col, typ in detected_types.items() if typ == 'logic']
        
        # If no clear matches, use similarity matching with common terms
        if not source_candidates:
            source_candidates = cls._find_similar_columns(df.columns, ['fields', 'field', 'source', 'column', 'provider', 'input'])
        
        if not target_candidates:
            target_candidates = cls._find_similar_columns(df.columns, ['xml', 'target', 'output', 'destination', 'hrp', 'mapped'])
        
        if not type_candidates:
            type_candidates = cls._find_similar_columns(df.columns, ['type', 'mapping_type', 'method', 'transform'])
        
        if not logic_candidates:
            logic_candidates = cls._find_similar_columns(df.columns, ['logic', 'rule', 'description', 'note', 'applied'])
        
        # Assign best matches
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
        """Find columns similar to reference terms"""
        matches = []
        
        for col in columns:
            col_lower = col.lower().strip()
            for term in reference_terms:
                if cls.calculate_similarity(col_lower, term) >= Config.SIMILARITY_THRESHOLD:
                    matches.append((col, cls.calculate_similarity(col_lower, term)))
                    break
        
        # Sort by similarity score
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches]

# --- Enhanced Custom Styling ---
def load_custom_css():
    """Load custom CSS with error handling"""
    try:
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-header h1 { 
            font-size: 2.8rem; 
            font-weight: 700; 
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .main-header p { 
            font-size: 1.2rem; 
            margin: 0.5rem 0; 
            opacity: 0.95;
        }
        .main-header small { 
            font-size: 0.9rem; 
            display: block; 
            margin-top: 1rem; 
            color: #e8e8e8; 
            opacity: 0.8;
        }
        .xml-preview {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            overflow-x: auto;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        .stats-container {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            text-align: center;
        }
        .warning-box {
            background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .success-box {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .error-box {
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .info-box {
            background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        .feature-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .step-header {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            color: white;
            padding: 0.8rem 1.2rem;
            border-radius: 8px;
            margin: 1.5rem 0 1rem 0;
            font-weight: 600;
        }
        .detection-result {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
        }
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error loading CSS: {e}")
        st.error("CSS loading failed, but the app will continue to work.")

# --- Enhanced XML Builder ---
def build_dynamic_xml(row: pd.Series, pretty_print: bool = False) -> str:
    """Build XML from mapped data with enhanced error handling and formatting."""
    try:
        root = ET.Element("Provider")
        
        for path, value in row.items():
            if pd.isna(value) or value == "" or value is None:
                continue
                
            # Clean and validate XML path
            path_clean = str(path).strip()
            if not path_clean:
                continue
                
            # Split path and create nested structure
            parts = [part.strip() for part in path_clean.split("/") if part.strip()]
            if not parts:
                continue
                
            current = root
            
            # Build nested structure
            for part in parts[:-1]:
                # Sanitize element names - more restrictive for XML compliance
                part_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', str(part))
                if not part_clean or part_clean[0].isdigit():
                    part_clean = f"element_{part_clean}"
                # Ensure valid XML name
                part_clean = part_clean[:50]  # Limit length
                    
                found = current.find(part_clean)
                if found is None:
                    found = ET.SubElement(current, part_clean)
                current = found
            
            # Add final element
            final_part = re.sub(r'[^a-zA-Z0-9_-]', '_', str(parts[-1]))
            if not final_part or final_part[0].isdigit():
                final_part = f"element_{final_part}"
            final_part = final_part[:50]  # Limit length
                
            # Convert value to string and clean it
            value_str = str(value).strip()
            # Escape XML special characters
            value_str = value_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            value_str = value_str.replace('"', '&quot;').replace("'", '&apos;')
            
            if value_str and len(value_str) <= 1000:  # Limit value length
                ET.SubElement(current, final_part).text = value_str
        
        # Format output
        if pretty_print:
            try:
                rough_string = ET.tostring(root, encoding='unicode')
                reparsed = minidom.parseString(rough_string)
                pretty_xml_str = reparsed.toprettyxml(indent=Config.XML_INDENT)
                # Remove the XML declaration line
                lines = pretty_xml_str.split('\n')[1:]
                return '\n'.join(lines).strip()
            except:
                # Fallback to non-pretty print
                return ET.tostring(root, encoding='unicode')
        else:
            return ET.tostring(root, encoding='unicode')
            
    except Exception as e:
        logger.error(f"Error building XML for row: {e}")
        return f"<Provider><Error>Failed to build XML: {str(e)[:100]}</Error></Provider>"

# --- Intelligent Data Validation ---
def validate_and_analyze_reference_data(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Intelligently validate and analyze reference mapping data structure."""
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
        
        # Auto-detect column purposes
        detector = ColumnDetector()
        column_mapping, detected_types = detector.auto_detect_columns(df)
        
        analysis['column_mapping'] = column_mapping
        analysis['detected_types'] = detected_types
        analysis['total_columns'] = len(df.columns)
        analysis['total_rows'] = len(df)
        
        # Check if we found essential columns
        if 'source' not in column_mapping:
            warnings.append("Could not automatically identify source field column. Manual mapping may be required.")
        
        if 'target' not in column_mapping:
            warnings.append("Could not automatically identify target XML field column. Manual mapping may be required.")
        
        # Analyze data quality
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:  # More than 50% null
                warnings.append(f"Column '{col}' has {null_count}/{len(df)} null values")
        
        analysis['warnings'] = warnings
        analysis['confidence_score'] = len(column_mapping) / 4.0  # Max 4 types (source, target, type, logic)
        
        return True, errors, analysis
        
    except Exception as e:
        errors.append(f"Error analyzing reference data: {str(e)}")
        return False, errors, {}

def validate_provider_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate provider data structure."""
    errors = []
    
    try:
        if len(df) == 0:
            errors.append("Provider data is empty")
        
        if len(df.columns) == 0:
            errors.append("Provider data has no columns")
            
        # Check for reasonable data size
        if len(df) > 10000:
            errors.append("Warning: Large dataset detected. Consider processing in smaller batches.")
            
    except Exception as e:
        errors.append(f"Error validating provider data: {str(e)}")
    
    return len(errors) == 0, errors

# --- Enhanced File Loading ---
def load_file_with_encoding(file, file_type: str) -> pd.DataFrame:
    """Load file with multiple encoding attempts and better error handling."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    try:
        # Check file size
        file.seek(0, 2)  # Go to end of file
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > Config.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB. Maximum allowed: {Config.MAX_FILE_SIZE / (1024*1024):.1f}MB")
        
        for encoding in encodings:
            try:
                file.seek(0)  # Reset file pointer
                if file_type == "csv":
                    # More robust CSV reading
                    df = pd.read_csv(
                        file, 
                        encoding=encoding, 
                        on_bad_lines='skip',
                        low_memory=False,
                        dtype=str,  # Read everything as string initially
                        na_values=['', 'NULL', 'null', 'NA', 'na', 'N/A', 'n/a']
                    )
                elif file_type in ["xlsx", "xls"]:
                    df = pd.read_excel(file, dtype=str, na_values=['', 'NULL', 'null', 'NA', 'na', 'N/A', 'n/a'])
                
                # Basic validation
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
def generate_mapping_stats(rag_df: pd.DataFrame, prov_df: pd.DataFrame, field_map: Dict[str, str]) -> Dict[str, Any]:
    """Generate comprehensive mapping statistics."""
    try:
        stats = {
            'total_reference_mappings': len(rag_df),
            'total_provider_columns': len(prov_df.columns),
            'mapped_columns': len(field_map),
            'unmapped_columns': len(prov_df.columns) - len(field_map),
            'mapping_coverage': (len(field_map) / len(prov_df.columns) * 100) if len(prov_df.columns) > 0 else 0,
            'total_provider_records': len(prov_df),
            'unmapped_column_list': [col for col in prov_df.columns if col not in field_map]
        }
        return stats
    except Exception as e:
        logger.error(f"Error generating stats: {e}")
        return {
            'total_reference_mappings': 0,
            'total_provider_columns': 0,
            'mapped_columns': 0,
            'unmapped_columns': 0,
            'mapping_coverage': 0,
            'total_provider_records': 0,
            'unmapped_column_list': []
        }

# --- Safe Download Functions ---
def create_safe_download_button(label: str, data: Any, filename: str, mime_type: str):
    """Create a download button with error handling"""
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

# --- Manual Column Mapping Interface ---
def display_manual_mapping_interface(df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, str]:
    """Display interface for manual column mapping when auto-detection fails"""
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
            "Source Fields Column (contains provider field names):",
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

# --- Main Application ---
def main():
    try:
        st.set_page_config(
            page_title="Intelligent HRP AI Data Mapper", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        load_custom_css()

        # Header
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
            <div class="main-header">
                <h1>🧠 Intelligent HRP AI Data Mapper</h1>
                <p>Smart field mapping with AI-powered column detection - no rigid formats required!</p>
                <small>🗓️ Generated on {now}</small>
            </div>
        """, unsafe_allow_html=True)

        # Features Overview
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

        # Sidebar Configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            show_preview = st.checkbox("Show XML Preview", value=True)
            max_preview = st.slider("Max Preview Records", 1, 10, Config.MAX_PREVIEW_RECORDS)
            pretty_xml = st.checkbox("Pretty Print XML", value=True)
            
            st.header("🎯 Detection Settings")
            similarity_threshold = st.slider("Column Similarity Threshold", 0.1, 1.0, Config.SIMILARITY_THRESHOLD, 0.1)
            Config.SIMILARITY_THRESHOLD = similarity_threshold
            
            st.header("📊 Export Options")
            include_stats = st.checkbox("Include Statistics in Report", value=True)
            xml_encoding = st.selectbox("XML Encoding", ["UTF-8", "ISO-8859-1"], index=0)

        # Step 1: Reference Data Upload
        st.markdown('<div class="step-header">📁 Step 1: Upload Reference Mapping Data</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <h4>🧠 Smart Upload</h4>
            <p>Upload any CSV file with mapping data. The AI will automatically detect:</p>
            <ul>
                <li>Source field columns (provider field names)</li>
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

        # Step 2: Provider Data Upload
        st.markdown('<div class="step-header">📄 Step 2: Upload Provider Input Data</div>', unsafe_allow_html=True)
        st.info("Upload your provider data file in CSV or Excel format")
        
        prov_file = st.file_uploader(
            "Provider Input File", 
            type=Config.SUPPORTED_FORMATS, 
            key="provider",
            help="Provider data to be transformed into HRP XML format"
        )

        # Early exit if files not uploaded
        if not rag_file or not prov_file:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Files Required</h4>
                <p>Please upload both reference mapping and provider data files to continue.</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Load and validate files
        try:
            with st.spinner("🔄 Loading and intelligently analyzing files..."):
                # Load reference data
                rag_df = load_file_with_encoding(rag_file, "csv")
                
                # Load provider data
                file_extension = prov_file.name.split('.')[-1].lower()
                prov_df = load_file_with_encoding(prov_file, file_extension)
                
                # Intelligent validation and analysis
                rag_valid, rag_errors, analysis = validate_and_analyze_reference_data(rag_df)
                prov_valid, prov_errors = validate_provider_data(prov_df)
                
                if not rag_valid or not prov_valid:
                    error_msg = "Data validation failed:\n"
                    error_msg += "\n".join(rag_errors + prov_errors)
                    st.error(error_msg)
                    st.stop()

            # Display analysis results
            st.markdown('<div class="step-header">🔍 AI Analysis Results</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Columns Detected", analysis['total_columns'])
            with col2:
                st.metric("🎯 Auto-Mapped", len(analysis['column_mapping']))
            with col3:
                confidence_pct = analysis['confidence_score'] * 100
                st.metric("🎯 Confidence", f"{confidence_pct:.0f}%")
            
            # Show detection results
            if analysis['column_mapping']:
                st.markdown("### ✅ Auto-Detected Column Mappings:")
                for purpose, column in analysis['column_mapping'].items():
                    purpose_emoji = {"source": "📥", "target": "📤", "type": "🔄", "logic": "📝"}
                    st.markdown(f"""
                    <div class="detection-result">
                        <strong>{purpose_emoji.get(purpose, '📋')} {purpose.title()} Field:</strong> <code>{column}</code>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show warnings if any
            if analysis.get('warnings'):
                with st.expander("⚠️ Analysis Warnings", expanded=False):
                    for warning in analysis['warnings']:
                        st.warning(warning)
            
            # Manual mapping interface if confidence is low
            column_mapping = analysis['column_mapping']
            if analysis['confidence_score'] < 0.5 or 'source' not in column_mapping or 'target' not in column_mapping:
                manual_override = st.checkbox("🎯 Override Auto-Detection (Manual Mapping)", value=False)
                if manual_override:
                    column_mapping = display_manual_mapping_interface(rag_df, analysis)
                    if not column_mapping.get('source') or not column_mapping.get('
