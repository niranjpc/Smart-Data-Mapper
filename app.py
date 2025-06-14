import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import io
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
import traceback

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

# --- Data Validation ---
def validate_reference_data(df: pd.DataFrame) -> tuple[bool, List[str]]:
    """Validate reference mapping data structure."""
    errors = []
    
    try:
        required_cols = ['fields', 'xml field']
        
        # Check required columns
        df_cols_lower = [str(col).lower().strip() for col in df.columns]
        for req_col in required_cols:
            if req_col.lower() not in df_cols_lower:
                errors.append(f"Missing required column: '{req_col}'")
        
        # Check for empty mappings
        if len(df) == 0:
            errors.append("Reference data is empty")
        
        # Check data types and content
        if len(errors) == 0:
            non_null_count = df.dropna(subset=[col for col in df.columns if col.lower() in ['fields', 'xml field']]).shape[0]
            if non_null_count == 0:
                errors.append("No valid mapping entries found")
                
    except Exception as e:
        errors.append(f"Error validating reference data: {str(e)}")
    
    return len(errors) == 0, errors

def validate_provider_data(df: pd.DataFrame) -> tuple[bool, List[str]]:
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

# --- Main Application ---
def main():
    try:
        st.set_page_config(
            page_title="HRP AI Data Mapper", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        load_custom_css()

        # Header
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
            <div class="main-header">
                <h1>🤖 HRP AI Data Mapper</h1>
                <p>Transform your provider data into HRP-compliant XML with AI-guided field mapping</p>
                <small>🗓️ Generated on {now}</small>
            </div>
        """, unsafe_allow_html=True)

        # Features Overview
        st.markdown("## 🚀 Key Features")
        st.markdown("""
        <div class="feature-grid">
            <div class="feature-card">
                <h4>📊 Smart Mapping</h4>
                <p>AI-guided field mapping using SME reference data</p>
            </div>
            <div class="feature-card">
                <h4>🔄 Multi-Format Support</h4>
                <p>CSV, XLSX file processing with encoding detection</p>
            </div>
            <div class="feature-card">
                <h4>✨ XML Generation</h4>
                <p>HRP-compliant XML with validation and formatting</p>
            </div>
            <div class="feature-card">
                <h4>📈 Analytics</h4>
                <p>Comprehensive mapping statistics and audit reports</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar Configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            show_preview = st.checkbox("Show XML Preview", value=True)
            max_preview = st.slider("Max Preview Records", 1, 10, Config.MAX_PREVIEW_RECORDS)
            pretty_xml = st.checkbox("Pretty Print XML", value=True)
            
            st.header("📊 Export Options")
            include_stats = st.checkbox("Include Statistics in Report", value=True)
            xml_encoding = st.selectbox("XML Encoding", ["UTF-8", "ISO-8859-1"], index=0)

        # Step 1: Reference Data Upload
        st.markdown('<div class="step-header">📁 Step 1: Upload Reference Mapping Data</div>', unsafe_allow_html=True)
        st.info("Upload a CSV file containing columns: 'fields', 'xml field', 'mapping_type' (optional), 'logic_applied' (optional)")
        
        rag_file = st.file_uploader(
            "Reference Mapping File (CSV)", 
            type=["csv"], 
            key="rag",
            help="CSV file with mapping rules from subject matter experts"
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
            with st.spinner("🔄 Loading and validating files..."):
                # Load reference data
                rag_df = load_file_with_encoding(rag_file, "csv")
                
                # Load provider data
                file_extension = prov_file.name.split('.')[-1].lower()
                prov_df = load_file_with_encoding(prov_file, file_extension)
                
                # Validate data
                rag_valid, rag_errors = validate_reference_data(rag_df)
                prov_valid, prov_errors = validate_provider_data(prov_df)
                
                if not rag_valid or not prov_valid:
                    error_msg = "Data validation failed:\n"
                    error_msg += "\n".join(rag_errors + prov_errors)
                    st.error(error_msg)
                    st.stop()

            st.markdown("""
            <div class="success-box">
                <h4>✅ Files Loaded Successfully</h4>
                <p>Both files have been loaded and validated.</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ File Loading Error</h4>
                <p>Error reading files: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
            logger.error(f"File loading error: {traceback.format_exc()}")
            st.stop()

        # Data Processing
        st.markdown('<div class="step-header">🧠 Step 3: AI-Guided Field Mapping</div>', unsafe_allow_html=True)
        
        try:
            # Normalize column names
            rag_df.columns = rag_df.columns.str.strip().str.lower()
            prov_df.columns = prov_df.columns.str.strip()

            # Build field mapping
            field_map = {}
            audit_rows = []

            for _, row in rag_df.iterrows():
                try:
                    src = row.get("fields")
                    tgt = row.get("xml field") 
                    mtype = row.get("mapping_type", "Direct")
                    logic = row.get("logic_applied", "No specific logic")
                    
                    if pd.notna(src) and pd.notna(tgt):
                        src_clean = str(src).strip()
                        tgt_clean = str(tgt).strip()
                        
                        if src_clean and tgt_clean:
                            field_map[src_clean] = tgt_clean
                            audit_rows.append({
                                "Source Column": src_clean,
                                "Target XML Path": tgt_clean,
                                "Mapping Type": str(mtype),
                                "Logic Applied": str(logic),
                                "Status": "✅ Mapped" if src_clean in prov_df.columns else "⚠️ Not Found in Provider Data"
                            })
                except Exception as e:
                    logger.error(f"Error processing mapping row: {e}")
                    continue

            # Generate statistics
            stats = generate_mapping_stats(rag_df, prov_df, field_map)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Mappings", stats['total_reference_mappings'])
            with col2:
                st.metric("📋 Provider Columns", stats['total_provider_columns'])
            with col3:
                st.metric("✅ Mapped Columns", stats['mapped_columns'])
            with col4:
                st.metric("📈 Coverage", f"{stats['mapping_coverage']:.1f}%")

            # Warnings for unmapped columns
            if stats['unmapped_columns'] > 0:
                with st.expander(f"⚠️ {stats['unmapped_columns']} Unmapped Columns", expanded=False):
                    st.write("The following provider columns were not found in the reference mapping:")
                    for col in stats['unmapped_column_list']:
                        st.write(f"• {col}")

        except Exception as e:
            st.error(f"Error in field mapping: {str(e)}")
            logger.error(f"Field mapping error: {traceback.format_exc()}")
            st.stop()

        # Transform data
        try:
            with st.spinner("🔄 Transforming data..."):
                mapped_data = []
                for idx, row in prov_df.iterrows():
                    try:
                        xml_row = {}
                        for col in prov_df.columns:
                            if col in field_map:
                                xml_path = field_map[col]
                                xml_row[xml_path] = row[col]
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
                    
                    with st.expander(f"🏥 Provider {i+1} XML Preview", expanded=i == 0):
                        st.markdown(f'<div class="xml-preview">{xml_content}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Error generating preview: {str(e)}")

        # Generate full XML
        st.markdown('<div class="step-header">📦 Step 5: Export Results</div>', unsafe_allow_html=True)
        
        try:
            # XML Export
            encoding_map = {"UTF-8": "utf-8", "ISO-8859-1": "iso-8859-1"}
            selected_encoding = encoding_map[xml_encoding]
            
            xml_declaration = f'<?xml version="1.0" encoding="{selected_encoding.upper()}"?>\n'
            xml_providers = []
            
            for row_data in mapped_data:
                xml_providers.append(build_dynamic_xml(pd.Series(row_data), pretty_print=pretty_xml))
            
            full_xml = xml_declaration + '<Providers>\n' + '\n'.join(xml_providers) + '\n</Providers>'
            
            col1, col2 = st.columns(2)
            
            with col1:
                create_safe_download_button(
                    "📥 Download Complete XML File",
                    full_xml,
                    f"hrp_providers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml",
                    "application/xml"
                )
            
            # Audit Report Export
            with col2:
                try:
                    audit_df = pd.DataFrame(audit_rows)
                    
                    if include_stats:
                        # Add statistics sheet
                        stats_rows = []
                        for key, value in stats.items():
                            if key != 'unmapped_column_list':
                                stats_rows.append({"Metric": key.replace('_', ' ').title(), "Value": str(value)})
                        
                        # Create Excel with multiple sheets
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

        except Exception as e:
            st.error(f"Error generating exports: {str(e)}")
            logger.error(f"Export generation error: {traceback.format_exc()}")

        # Detailed Audit Report Display
        try:
            st.markdown('<div class="step-header">📋 Mapping Audit Report</div>', unsafe_allow_html=True)
            
            if audit_rows:
                audit_df = pd.DataFrame(audit_rows)
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    status_filter = st.selectbox(
                        "Filter by Status:",
                        ["All", "✅ Mapped", "⚠️ Not Found in Provider Data"],
                        index=0
                    )
                with col2:
                    mapping_types = list(audit_df["Mapping Type"].unique()) if not audit_df.empty else []
                    mapping_type_filter = st.selectbox(
                        "Filter by Mapping Type:",
                        ["All"] + mapping_types,
                        index=0
                    )
                
                # Apply filters
                display_df = audit_df.copy()
                if status_filter != "All":
                    display_df = display_df[display_df["Status"] == status_filter]
                if mapping_type_filter != "All":
                    display_df = display_df[display_df["Mapping Type"] == mapping_type_filter]
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No audit data available.")
            
            # Summary
            st.markdown(f"""
            <div class="stats-container">
                <h4>📊 Processing Summary</h4>
                <p><strong>{len(prov_df)}</strong> provider records processed | 
                <strong>{stats['mapped_columns']}</strong> fields mapped | 
                <strong>{stats['mapping_coverage']:.1f}%</strong> coverage achieved</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error displaying audit report: {str(e)}")
            logger.error(f"Audit report display error: {traceback.format_exc()}")

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