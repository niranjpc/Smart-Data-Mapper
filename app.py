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

# ... (rest of the code as in the previous message, due to message length limits) ...

if __name__ == "__main__":
    main()
