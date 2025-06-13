import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import requests
import time
import json
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom CSS for Professional UI ---
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .success-banner {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .error-banner {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        padding: 0 1rem;
    }
    
    .step {
        flex: 1;
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 0 0.5rem;
        position: relative;
    }
    
    .step.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .step.completed {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
    }
    
    .download-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #0c5460;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Enhanced API Functions with Better Error Handling ---
class HuggingFaceAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-inference.huggingface.co"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
    def get_embeddings(self, texts: List[str], max_retries: int = 3) -> Optional[List[List[float]]]:
        """Get embeddings with proper error handling and retries"""
        if not texts:
            return []
            
        # Fixed API endpoint - use models/ prefix
        url = f"{self.base_url}/models/sentence-transformers/all-MiniLM-L6-v2"
        
        for attempt in range(max_retries):
            try:
                # Batch processing for efficiency
                batch_size = 10
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    payload = {
                        "inputs": batch,
                        "options": {"wait_for_model": True}
                    }
                    
                    response = requests.post(url, headers=self.headers, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list):
                            all_embeddings.extend(result)
                        else:
                            logger.error(f"Unexpected API response format: {result}")
                            return None
                    elif response.status_code == 503:
                        st.warning(f"Model loading... Attempt {attempt + 1}/{max_retries}")
                        time.sleep(10 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        logger.error(f"API Error {response.status_code}: {response.text}")
                        if attempt == max_retries - 1:
                            st.error(f"Embedding API error after {max_retries} attempts: {response.status_code}")
                            return None
                        time.sleep(5)
                        continue
                        
                return all_embeddings
                
            except requests.exceptions.Timeout:
                st.warning(f"Request timeout. Attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Unexpected error in get_embeddings: {str(e)}")
                if attempt == max_retries - 1:
                    st.error(f"Failed to get embeddings: {str(e)}")
                    return None
                time.sleep(5)
                
        return None

    def generate_text(self, prompt: str, max_retries: int = 3) -> str:
        """Generate text with better error handling"""
        url = f"{self.base_url}/models/google/flan-t5-base"  # Use base model for better results
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 100,
                        "temperature": 0.3,
                        "return_full_text": False,
                        "do_sample": True
                    }
                }
                
                response = requests.post(url, headers=self.headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        if 'generated_text' in result[0]:
                            return result[0]['generated_text'].strip()
                        else:
                            return str(result[0]).strip()
                    return "Generated explanation available"
                elif response.status_code == 503:
                    if attempt < max_retries - 1:
                        st.warning(f"Text generation model loading... Attempt {attempt + 1}")
                        time.sleep(10)
                        continue
                    return "Explanation generation temporarily unavailable"
                else:
                    logger.error(f"Text generation error: {response.status_code}")
                    return "Unable to generate explanation"
                    
            except Exception as e:
                logger.error(f"Error in generate_text: {str(e)}")
                if attempt == max_retries - 1:
                    return "Explanation generation failed"
                time.sleep(2)
                
        return "Explanation generation failed"

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return 0.0

def build_xml(provider_data: Dict[str, any]) -> str:
    """Build XML from provider data with better formatting"""
    try:
        provider_el = ET.Element("provider")
        
        for path, val in provider_data.items():
            if pd.isna(val) or val == "":
                val = ""
            
            # Handle nested XML paths
            parts = str(path).split("/")
            current = provider_el
            
            for part in parts[:-1]:
                found = current.find(part)
                if found is None:
                    found = ET.SubElement(current, part)
                current = found
                
            # Clean the part name for XML
            final_part = parts[-1].replace(" ", "_").replace("-", "_")
            ET.SubElement(current, final_part).text = str(val)
        
        # Pretty print XML
        from xml.dom import minidom
        rough_string = ET.tostring(provider_el, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ").split('\n', 1)[1]  # Remove first line
        
    except Exception as e:
        logger.error(f"Error building XML: {str(e)}")
        return f"<provider><error>XML generation failed: {str(e)}</error></provider>"

def validate_file(file, file_type: str) -> Tuple[bool, str]:
    """Validate uploaded files with enhanced error handling"""
    try:
        if file_type == "reference":
            if not file.name.endswith('.csv'):
                return False, "Reference files must be CSV format"
            
            # Reset file pointer to beginning
            file.seek(0)
            
            # Try to read the file with different encodings and parameters
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    file.seek(0)
                    # Try with different separators and handle various CSV issues
                    df = pd.read_csv(
                        file, 
                        encoding=encoding,
                        sep=None,  # Auto-detect separator
                        engine='python',  # More flexible parser
                        skipinitialspace=True,
                        na_values=['', 'NA', 'N/A', 'null', 'NULL'],
                        keep_default_na=True,
                        skip_blank_lines=True
                    )
                    break
                except (UnicodeDecodeError, pd.errors.EmptyDataError):
                    continue
                except Exception as e:
                    if encoding == encodings_to_try[-1]:  # Last encoding attempt
                        return False, f"Unable to parse CSV: {str(e)}"
                    continue
            
            if df is None or df.empty:
                return False, "File is empty or cannot be parsed"
            
            # Clean column names - remove extra whitespace and convert to lowercase for comparison
            df.columns = df.columns.str.strip()
            column_names_lower = [col.lower() for col in df.columns]
            
            # Check for required columns (flexible matching)
            required_columns = ['fields', 'xml field']
            required_lower = [col.lower() for col in required_columns]
            
            missing_columns = []
            for req_col, req_lower in zip(required_columns, required_lower):
                if req_lower not in column_names_lower:
                    # Try alternative column names
                    alternatives = {
                        'fields': ['field', 'field_name', 'source_field', 'provider_field'],
                        'xml field': ['xml_field', 'target_field', 'destination_field', 'xml path', 'xml_path']
                    }
                    
                    found = False
                    if req_lower in alternatives:
                        for alt in alternatives[req_lower]:
                            if alt.lower() in column_names_lower:
                                found = True
                                break
                    
                    if not found:
                        missing_columns.append(req_col)
            
            if missing_columns:
                available_cols = ', '.join(df.columns.tolist())
                return False, f"Missing required columns: {missing_columns}. Available columns: [{available_cols}]. Required: {required_columns}"
            
            # Check if there are any non-empty rows
            if len(df.dropna(how='all')) == 0:
                return False, "File contains no valid data rows"
            
            # Additional validation - check for reasonable data
            fields_col = None
            xml_col = None
            
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ['fields', 'field', 'field_name', 'source_field', 'provider_field']:
                    fields_col = col
                elif col_lower in ['xml field', 'xml_field', 'target_field', 'destination_field', 'xml path', 'xml_path']:
                    xml_col = col
            
            if fields_col and xml_col:
                non_empty_rows = df[(df[fields_col].notna()) & (df[xml_col].notna()) & 
                                   (df[fields_col] != '') & (df[xml_col] != '')]
                if len(non_empty_rows) == 0:
                    return False, "No rows contain valid data in both required columns"
                
        elif file_type == "provider":
            if not (file.name.endswith('.csv') or file.name.endswith('.xlsx')):
                return False, "Provider file must be CSV or Excel format"
            
            # Reset file pointer
            file.seek(0)
            
            # Try to read provider file
            try:
                if file.name.endswith('.csv'):
                    # Try different encodings for CSV
                    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    df = None
                    
                    for encoding in encodings_to_try:
                        try:
                            file.seek(0)
                            df = pd.read_csv(
                                file,
                                encoding=encoding,
                                sep=None,
                                engine='python',
                                skipinitialspace=True,
                                na_values=['', 'NA', 'N/A', 'null', 'NULL'],
                                keep_default_na=True
                            )
                            break
                        except (UnicodeDecodeError, pd.errors.EmptyDataError):
                            continue
                else:
                    df = pd.read_excel(file, engine='openpyxl')
                
                if df is None or df.empty:
                    return False, "Provider file is empty or cannot be parsed"
                    
            except Exception as e:
                return False, f"Error reading provider file: {str(e)}"
                
        return True, "File is valid"
        
    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        return False, f"File validation error: {str(e)}"

# --- Main Streamlit App ---
def main():
    st.set_page_config(
        page_title="Smart Data Mapper Pro",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Smart Data Mapper Pro</h1>
        <p>AI-Powered Data Mapping & XML Generation Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings and info
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key management
        if "HUGGINGFACE_TOKEN" in st.secrets and st.secrets["HUGGINGFACE_TOKEN"]:
            api_key = st.secrets["HUGGINGFACE_TOKEN"]
            st.success("✅ API Token loaded from secrets")
        else:
            api_key = st.text_input(
                "Hugging Face API Token",
                type="password",
                placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                help="Get your free token from huggingface.co/settings/tokens"
            )
        
        if not api_key:
            st.error("API token required!")
            st.markdown("[Get your free token here](https://huggingface.co/settings/tokens)")
            st.stop()
        
        st.divider()
        
        # Configuration options
        st.subheader("🔧 Configuration")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum confidence for automatic mapping"
        )
        
        batch_size = st.selectbox(
            "Processing Batch Size",
            options=[5, 10, 20, 50],
            index=1,
            help="Number of records to process at once"
        )
        
        st.divider()
        
        # Help section
        with st.expander("📚 How to Use", expanded=False):
            st.markdown("""
            1. **Upload Reference Data**: CSV files with mapping rules
            2. **Upload Provider Data**: File to be mapped
            3. **Review Mappings**: Check AI suggestions
            4. **Download Results**: Get XML and reports
            
            **Required columns in reference CSV:**
            - `fields`: Source field names
            - `xml field`: Target XML paths
            - `logic`: Mapping logic (optional)
            - `comments`: Additional notes (optional)
            """)
        
        # Statistics
        if 'stats' in st.session_state:
            st.subheader("📊 Session Stats")
            stats = st.session_state.stats
            st.metric("Files Processed", stats.get('files_processed', 0))
            st.metric("Mappings Created", stats.get('mappings_created', 0))
            st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
    
    # Initialize session state
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'files_processed': 0,
            'mappings_created': 0,
            'success_rate': 0.0
        }
    
    # Step indicator
    st.markdown("""
    <div class="step-indicator">
        <div class="step active">
            <h4>1. Upload Reference</h4>
            <p>Mapping rules</p>
        </div>
        <div class="step">
            <h4>2. Upload Provider</h4>
            <p>Data to map</p>
        </div>
        <div class="step">
            <h4>3. Process</h4>
            <p>AI mapping</p>
        </div>
        <div class="step">
            <h4>4. Download</h4>
            <p>Results</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize API client
    hf_api = HuggingFaceAPI(api_key)
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Reference Data Files")
        st.markdown('<div class="info-box">Upload CSV files containing your mapping rules and logic.</div>', unsafe_allow_html=True)
        
        rag_files = st.file_uploader(
            "Upload Reference CSV Files",
            type=["csv"],
            accept_multiple_files=True,
            help="Files containing field mappings and transformation rules"
        )
        
        if rag_files:
            rag_dfs = {}
            valid_files = 0
            error_details = []
            
            for rag_file in rag_files:
                # Reset file pointer
                rag_file.seek(0)
                
                is_valid, message = validate_file(rag_file, "reference")
                if is_valid:
                    try:
                        # Reset file pointer again before reading
                        rag_file.seek(0)
                        
                        # Enhanced CSV reading with multiple fallback options
                        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                        rag_df = None
                        
                        for encoding in encodings_to_try:
                            try:
                                rag_file.seek(0)
                                rag_df = pd.read_csv(
                                    rag_file,
                                    encoding=encoding,
                                    sep=None,  # Auto-detect separator
                                    engine='python',  # More flexible parser
                                    skipinitialspace=True,
                                    na_values=['', 'NA', 'N/A', 'null', 'NULL'],
                                    keep_default_na=True,
                                    skip_blank_lines=True
                                )
                                break
                            except Exception as read_error:
                                if encoding == encodings_to_try[-1]:  # Last attempt
                                    error_details.append(f"{rag_file.name}: {str(read_error)}")
                                continue
                        
                        if rag_df is not None and not rag_df.empty:
                            # Clean column names
                            rag_df.columns = rag_df.columns.str.strip()
                            
                            # Standardize column names to match expected format
                            column_mapping = {}
                            for col in rag_df.columns:
                                col_lower = col.lower().strip()
                                if col_lower in ['field', 'field_name', 'source_field', 'provider_field']:
                                    column_mapping[col] = 'fields'
                                elif col_lower in ['xml_field', 'target_field', 'destination_field', 'xml path', 'xml_path']:
                                    column_mapping[col] = 'xml field'
                            
                            # Rename columns if needed
                            if column_mapping:
                                rag_df = rag_df.rename(columns=column_mapping)
                            
                            # Remove completely empty rows
                            rag_df = rag_df.dropna(how='all')
                            
                            if not rag_df.empty:
                                rag_dfs[rag_file.name] = rag_df
                                valid_files += 1
                            else:
                                error_details.append(f"{rag_file.name}: File contains no valid data after cleaning")
                        else:
                            error_details.append(f"{rag_file.name}: Unable to read file content")
                            
                    except Exception as e:
                        error_details.append(f"{rag_file.name}: {str(e)}")
                else:
                    error_details.append(f"{rag_file.name}: {message}")
            
            # Display results
            if valid_files > 0:
                st.markdown(f'<div class="success-banner">✅ {valid_files} reference file(s) loaded successfully!</div>', unsafe_allow_html=True)
                
                # Show file details
                for name, df in rag_dfs.items():
                    st.success(f"✅ {name}: {len(df)} records, {len(df.columns)} columns")
                
                # Preview section
                with st.expander("👀 Preview Reference Data", expanded=False):
                    for name, df in rag_dfs.items():
                        st.markdown(f"**{name}** ({len(df)} records)")
                        st.dataframe(df.head(3), use_container_width=True)
                        
                        # Show column info
                        st.markdown("**Columns:** " + ", ".join(df.columns.tolist()))
            
            # Show errors if any
            if error_details:
                st.markdown('<div class="warning-box">⚠️ Some files could not be processed:</div>', unsafe_allow_html=True)
                for error in error_details:
                    st.error(f"❌ {error}")
                
                # Helpful tips
                with st.expander("💡 Troubleshooting Tips", expanded=False):
                    st.markdown("""
                    **Common issues and solutions:**
                    
                    1. **Empty file**: Make sure your CSV has data rows, not just headers
                    2. **Wrong encoding**: Try saving your CSV as UTF-8 in Excel/Google Sheets
                    3. **Missing columns**: Your CSV must have 'fields' and 'xml field' columns (case-insensitive)
                    4. **Wrong separator**: Make sure you're using commas (,) as separators
                    5. **Special characters**: Avoid special characters in column names
                    
                    **Required CSV format:**
                    ```
                    fields,xml field,logic,comments
                    provider_name,provider/name,Direct mapping,Provider company name
                    provider_id,provider/id,Direct mapping,Unique identifier
                    ```
                    """)
            
            if valid_files == 0:
                st.error("❌ No valid reference files could be loaded!")
                st.stop()
        else:
            st.info("👆 Please upload at least one reference CSV file")
            st.stop()
    
    with col2:
        st.subheader("📄 Provider Data File")
        st.markdown('<div class="info-box">Upload the file containing data to be mapped to XML format.</div>', unsafe_allow_html=True)
        
        prov_file = st.file_uploader(
            "Upload Provider Data File",
            type=["csv", "xlsx"],
            help="CSV or Excel file with provider data to be mapped"
        )
        
        if prov_file:
            # Reset file pointer
            prov_file.seek(0)
            
            is_valid, message = validate_file(prov_file, "provider")
            if not is_valid:
                st.error(f"❌ Invalid provider file: {message}")
                st.stop()
            
            try:
                # Reset file pointer again
                prov_file.seek(0)
                
                if prov_file.name.endswith(".csv"):
                    # Enhanced CSV reading for provider files
                    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    prov_df = None
                    
                    for encoding in encodings_to_try:
                        try:
                            prov_file.seek(0)
                            prov_df = pd.read_csv(
                                prov_file,
                                encoding=encoding,
                                sep=None,
                                engine='python',
                                skipinitialspace=True,
                                na_values=['', 'NA', 'N/A', 'null', 'NULL'],
                                keep_default_na=True
                            )
                            break
                        except Exception:
                            continue
                    
                    if prov_df is None:
                        st.error("❌ Unable to read CSV file. Please check the file format.")
                        st.stop()
                else:
                    prov_df = pd.read_excel(prov_file, engine='openpyxl')
                
                # Clean column names
                prov_df.columns = prov_df.columns.str.strip()
                
                # Remove completely empty rows
                prov_df = prov_df.dropna(how='all')
                
                if prov_df.empty:
                    st.error("❌ Provider file contains no valid data rows.")
                    st.stop()
                
                st.markdown('<div class="success-banner">✅ Provider file loaded successfully!</div>', unsafe_allow_html=True)
                
                # File statistics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Records", len(prov_df))
                with col_b:
                    st.metric("Fields", len(prov_df.columns))
                with col_c:
                    st.metric("File Size", f"{prov_file.size / 1024:.1f} KB")
                
                # Show column information
                st.markdown("**Detected Columns:** " + ", ".join(prov_df.columns.tolist()))
                
                # Preview
                with st.expander("👀 Preview Provider Data", expanded=False):
                    st.dataframe(prov_df.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error reading provider file: {str(e)}")
                
                # Enhanced error help
                with st.expander("💡 File Reading Help", expanded=True):
                    st.markdown("""
                    **Try these solutions:**
                    
                    1. **For CSV files:**
                       - Save as UTF-8 encoding in Excel
                       - Use comma (,) as separator
                       - Remove any special characters from headers
                    
                    2. **For Excel files:**
                       - Save as .xlsx format (not .xls)
                       - Ensure data starts from row 1
                       - Remove any merged cells in headers
                    
                    3. **General tips:**
                       - File should not be password protected
                       - Avoid very large files (>50MB)
                       - Check that file is not corrupted
                    """)
                st.stop()
        else:
            st.info("👆 Please upload a provider data file")
            st.stop()
    
    st.divider()
    
    # Processing section
    if st.button("🚀 Start AI-Powered Mapping", type="primary", use_container_width=True):
        
        # Update step indicator
        st.markdown("""
        <div class="step-indicator">
            <div class="step completed">
                <h4>1. Upload Reference</h4>
                <p>✓ Complete</p>
            </div>
            <div class="step completed">
                <h4>2. Upload Provider</h4>
                <p>✓ Complete</p>
            </div>
            <div class="step active">
                <h4>3. Process</h4>
                <p>🔄 Processing</p>
            </div>
            <div class="step">
                <h4>4. Download</h4>
                <p>Pending</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            with st.spinner("🧠 AI is analyzing your data..."):
                
                # Prepare data
                prov_columns = prov_df.columns.astype(str).tolist()
                
                # Progress tracking
                progress_container = st.container()
                status_placeholder = st.empty()
                
                # Prepare reference data
                reference_rows = []
                for rag_file, rag_df in rag_dfs.items():
                    for _, row in rag_df.iterrows():
                        reference_rows.append({
                            "fields": str(row['fields']),
                            "xml_field": str(row['xml field']),
                            "logic": str(row.get('logic', '')),
                            "comments": str(row.get('comments', '')),
                            "rag_file": rag_file
                        })
                
                reference_fields = [r["fields"] for r in reference_rows]
                
                # Get embeddings
                status_placeholder.info("🔍 Analyzing reference field semantics...")
                ref_embeddings = hf_api.get_embeddings(reference_fields)
                
                if ref_embeddings is None:
                    st.error("Failed to get reference embeddings. Please check your API token and try again.")
                    st.stop()
                
                status_placeholder.info("🔍 Analyzing provider field semantics...")
                prov_embeddings = hf_api.get_embeddings(prov_columns)
                
                if prov_embeddings is None:
                    st.error("Failed to get provider embeddings. Please check your API token and try again.")
                    st.stop()
                
                # Create mappings
                status_placeholder.info("🤖 Creating intelligent field mappings...")
                mapping_preview = []
                
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
                    
                    # Generate explanation
                    explanation_prompt = (
                        f"Explain why '{col}' maps to '{best_ref['xml_field']}'. "
                        f"Logic: {best_ref['logic']}. Notes: {best_ref['comments']}"
                    )
                    explanation = hf_api.generate_text(explanation_prompt)
                    
                    mapping_preview.append({
                        'Provider Field': col,
                        'XML Field': best_ref["xml_field"],
                        'Logic': best_ref["logic"],
                        'Comments': best_ref["comments"],
                        'Confidence': f"{best_score*100:.1f}%",
                        'Confidence_Score': best_score,
                        'Reference File': best_ref["rag_file"],
                        'AI Explanation': explanation,
                        'Status': '✅ High' if best_score >= confidence_threshold else '⚠️ Low'
                    })
                
                status_placeholder.success("✅ Field mappings completed!")
                
                # Display mapping results
                st.subheader("🎯 AI-Generated Field Mappings")
                
                # Summary metrics
                high_confidence = sum(1 for m in mapping_preview if m['Confidence_Score'] >= confidence_threshold)
                low_confidence = len(mapping_preview) - high_confidence
                avg_confidence = np.mean([m['Confidence_Score'] for m in mapping_preview]) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Mappings", len(mapping_preview))
                with col2:
                    st.metric("High Confidence", high_confidence, delta=f"{high_confidence/len(mapping_preview)*100:.1f}%")
                with col3:
                    st.metric("Low Confidence", low_confidence, delta=f"{low_confidence/len(mapping_preview)*100:.1f}%")
                with col4:
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                # Mapping table
                preview_df = pd.DataFrame(mapping_preview)
                
                # Color coding for confidence
                def highlight_confidence(row):
                    if row['Confidence_Score'] >= confidence_threshold:
                        return ['background-color: #d4edda'] * len(row)
                    else:
                        return ['background-color: #f8d7da'] * len(row)
                
                st.dataframe(
                    preview_df.drop('Confidence_Score', axis=1).style.apply(highlight_confidence, axis=1),
                    use_container_width=True
                )
                
                # Show low confidence mappings
                if low_confidence > 0:
                    st.markdown('<div class="warning-box">⚠️ Some mappings have low confidence. Please review them carefully.</div>', unsafe_allow_html=True)
                    
                    low_conf_df = preview_df[preview_df['Confidence_Score'] < confidence_threshold]
                    with st.expander(f"⚠️ Review {low_confidence} Low Confidence Mappings", expanded=True):
                        st.dataframe(low_conf_df.drop('Confidence_Score', axis=1), use_container_width=True)
                
                # Process data transformation
                status_placeholder.info("🔄 Transforming data to XML format...")
                
                results = []
                mapping_explanations = []
                mapping_report_rows = []
                
                # Batch processing
                progress_bar = st.progress(0)
                total_rows = len(prov_df)
                
                for i in range(0, total_rows, batch_size):
                    batch_end = min(i + batch_size, total_rows)
                    batch_df = prov_df.iloc[i:batch_end]
                    
                    for idx, (_, row) in enumerate(batch_df.iterrows()):
                        row_idx = i + idx
                        entry = {}
                        explain = {}
                        
                        for col in prov_df.columns:
                            mapping = next((m for m in mapping_preview if m['Provider Field'] == col), None)
                            xml_path = mapping['XML Field'] if mapping else col
                            value = row[col]
                            entry[xml_path] = value
                            explanation = mapping['AI Explanation'] if mapping else "Direct mapping"
                            explain[col] = explanation
                            
                            mapping_report_rows.append({
                                'Provider Row': row_idx + 1,
                                'Provider Field': col,
                                'Value': str(value)[:100] + ('...' if len(str(value)) > 100 else ''),
                                'XML Field': xml_path,
                                'Logic': mapping['Logic'] if mapping else "",
                                'Comments': mapping['Comments'] if mapping else "",
                                'Confidence': mapping['Confidence'] if mapping else "N/A",
                                'Reference File': mapping['Reference File'] if mapping else "",
                                'AI Explanation': explanation,
                                'Status': mapping['Status'] if mapping else "✅ Direct"
                            })
                        
                        results.append(entry)
                        mapping_explanations.append(explain)
                    
                    progress_bar.progress((batch_end) / total_rows)
                
                status_placeholder.success("✅ Data transformation completed!")
                
                # Update session stats
                st.session_state.stats['files_processed'] += 1
                st.session_state.stats['mappings_created'] += len(mapping_preview)
                st.session_state.stats['success_rate'] = (high_confidence / len(mapping_preview)) * 100
                
                # XML Generation
                st.subheader("📄 Generated XML Output")
                
                xml_strings = []
                xml_preview_container = st.container()
                
                with xml_preview_container:
                    # Show sample XMLs
                    sample_size = min(3, len(results))
                    for idx in range(sample_size):
                        xml_str = build_xml(results[idx])
                        xml_strings.append(xml_str)
                        
                        with st.expander(f"🔍 Sample XML {idx+1}", expanded=idx==0):
                            st.code(xml_str, language="xml")
                    
                    if len(results) > sample_size:
                        st.info(f"Showing {sample_size} sample XMLs. Full output available in download.")
                
                # Generate all XMLs
                for idx in range(sample_size, len(results)):
                    xml_str = build_xml(results[idx])
                    xml_strings.append(xml_str)
                
                # Download section
                st.markdown("""
                <div class="download-section">
                    <h3>📥 Download Your Results</h3>
                    <p>Your data has been successfully processed and is ready for download!</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Full XML download
                    full_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<providers>\n' + '\n'.join(xml_strings) + '\n</providers>'
                    st.download_button(
                        "📄 Download Complete XML",
                        full_xml,
                        file_name=f"providers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml",
                        mime="application/xml",
                        type="primary",
                        use_container_width=True
                    )
                
                with col2:
                    # Mapping report download
                    report_df = pd.DataFrame(mapping_report_rows)
                    csv_buffer = io.StringIO()
                    report_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "📊 Download Mapping Report",
                        csv_buffer.getvalue(),
                        file_name=f"mapping_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col3:
                    # Summary statistics download
                    summary_stats = {
                        'Processing Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Provider File': prov_file.name,
                        'Reference Files': ', '.join(rag_dfs.keys()),
                        'Total Records': len(prov_df),
                        'Total Fields': len(prov_df.columns),
                        'Total Mappings': len(mapping_preview),
                        'High Confidence Mappings': high_confidence,
                        'Low Confidence Mappings': low_confidence,
                        'Average Confidence': f"{avg_confidence:.2f}%",
                        'Success Rate': f"{(high_confidence/len(mapping_preview)*100):.1f}%"
                    }
                    
                    summary_json = json.dumps(summary_stats, indent=2)
                    st.download_button(
                        "📈 Download Summary Stats",
                        summary_json,
                        file_name=f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                # Update final step indicator
                st.markdown("""
                <div class="step-indicator">
                    <div class="step completed">
                        <h4>1. Upload Reference</h4>
                        <p>✓ Complete</p>
                    </div>
                    <div class="step completed">
                        <h4>2. Upload Provider</h4>
                        <p>✓ Complete</p>
                    </div>
                    <div class="step completed">
                        <h4>3. Process</h4>
                        <p>✓ Complete</p>
                    </div>
                    <div class="step completed">
                        <h4>4. Download</h4>
                        <p>✓ Ready</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Success message
                st.balloons()
                st.markdown(f"""
                <div class="success-banner">
                    🎉 Processing Complete! 
                    <br>
                    ✅ {len(results)} records processed
                    <br>
                    ✅ {len(mapping_preview)} field mappings created
                    <br>
                    ✅ {high_confidence} high-confidence mappings ({(high_confidence/len(mapping_preview)*100):.1f}%)
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed explanations section
                with st.expander("🤖 AI Mapping Explanations", expanded=False):
                    for i, mapping in enumerate(mapping_preview):
                        st.markdown(f"""
                        **{mapping['Provider Field']}** → **{mapping['XML Field']}**
                        - *Confidence: {mapping['Confidence']}*
                        - *Logic: {mapping['Logic']}*
                        - *AI Explanation: {mapping['AI Explanation']}*
                        """)
                        if i < len(mapping_preview) - 1:
                            st.divider()
                
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            st.markdown(f"""
            <div class="error-banner">
                ❌ Processing failed: {str(e)}
                <br>
                Please check your files and try again. If the problem persists, contact support.
            </div>
            """, unsafe_allow_html=True)
            
            # Error details for debugging
            with st.expander("🔍 Error Details", expanded=False):
                st.code(str(e))
                st.write("**Troubleshooting Tips:**")
                st.write("- Check your Hugging Face API token")
                st.write("- Ensure your CSV files have the required columns")
                st.write("- Try with smaller files first")
                st.write("- Check your internet connection")
    
    # Footer with additional resources
    st.divider()
    
    # Help and resources section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📚 Resources
        - [Sample Reference CSV](https://github.com/niranjpc/provider-mapper/blob/main/sample_rag_mapping.csv)
        - [Sample Provider CSV](https://github.com/niranjpc/provider-mapper/blob/main/sample_provider_input.csv)
        - [User Guide](https://docs.example.com/smart-mapper)
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Technical Info
        - **AI Models**: Hugging Face Transformers
        - **Embedding Model**: all-MiniLM-L6-v2
        - **Text Generation**: flan-t5-base
        - **Similarity**: Cosine Similarity
        """)
    
    with col3:
        st.markdown("""
        ### 🚀 Features
        - ✅ Semantic field mapping
        - ✅ Batch processing
        - ✅ Confidence scoring
        - ✅ AI explanations
        - ✅ XML generation
        - ✅ Comprehensive reporting
        """)
    
    # Version and credits
    st.markdown("""
    ---
    **Smart Data Mapper Pro v2.0** | Powered by Hugging Face AI | Built with ❤️ using Streamlit
    
    *For support, feature requests, or bug reports, please contact our team.*
    """)

if __name__ == "__main__":
    main()
