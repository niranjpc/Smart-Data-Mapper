# ✅ Refactored app.py with improved UX and performance
# This integrates the updated main() function into the full Streamlit app.

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import requests
import logging
from datetime import datetime
import io

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom Styling ---
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
    .main-header h1 { font-size: 2.5rem; font-weight: 700; }
    .main-header p { font-size: 1.2rem; margin: 0; }
    </style>
    """, unsafe_allow_html=True)

# --- API Handler ---
class HuggingFaceAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-inference.huggingface.co"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get_embeddings(self, texts: List[str], max_retries: int = 3) -> Optional[List[List[float]]]:
        if not texts: return []
        url = f"{self.base_url}/models/sentence-transformers/all-MiniLM-L6-v2"
        all_embeddings = []
        for i in range(0, len(texts), 10):
            payload = {"inputs": texts[i:i+10], "options": {"wait_for_model": True}}
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=30)
                if response.status_code == 200:
                    all_embeddings.extend(response.json())
                elif response.status_code == 503:
                    time.sleep(5)
                else:
                    return None
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                return None
        return all_embeddings

    def generate_text(self, prompt: str, max_retries: int = 3) -> str:
        url = f"{self.base_url}/models/google/flan-t5-base"
        try:
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 100}}
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result[0].get("generated_text", "")
        except Exception as e:
            logger.error(f"Text gen error: {e}")
        return "Explanation unavailable"

# --- Cosine Similarity ---
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1, v2 = np.array(vec1), np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# --- File Validation ---
def validate_file(file, file_type: str) -> (bool, str):
    try:
        if file_type == "reference":
            df = pd.read_csv(file)
            if 'fields' not in df.columns or 'xml field' not in df.columns:
                return False, "Missing required columns: 'fields' and 'xml field'"
        else:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            if df.empty:
                return False, "Empty provider file"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {str(e)}"

# --- Main App Logic ---
def main():
    st.set_page_config(page_title="Smart Data Mapper Pro", page_icon="🧠", layout="wide")
    load_custom_css()

    st.markdown("""
        <div class="main-header">
            <h1>🧠 Smart Data Mapper Pro</h1>
            <p>AI-Powered Data Mapping & XML Generation</p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configure")
        if "HUGGINGFACE_TOKEN" in st.secrets:
            api_key = st.secrets["HUGGINGFACE_TOKEN"]
        else:
            api_key = st.text_input("Hugging Face API Token", type="password")

        if not api_key:
            st.error("API Token required")
            st.stop()

        with st.expander("Advanced Settings"):
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
            batch_size = st.selectbox("Batch Size", [5, 10, 20, 50], index=1)

    st.markdown("<h4>Step 1: Upload Reference CSV</h4>", unsafe_allow_html=True)
    ref_file = st.file_uploader("Reference CSV", type="csv")
    if not ref_file: st.stop()

    st.markdown("<h4>Step 2: Upload Provider File</h4>", unsafe_allow_html=True)
    prov_file = st.file_uploader("Provider CSV/XLSX", type=["csv", "xlsx"])
    if not prov_file: st.stop()

    valid_ref, msg_ref = validate_file(ref_file, "reference")
    valid_prov, msg_prov = validate_file(prov_file, "provider")
    if not valid_ref: st.error(msg_ref); st.stop()
    if not valid_prov: st.error(msg_prov); st.stop()

    ref_file.seek(0); prov_file.seek(0)
    ref_df = pd.read_csv(ref_file)
    prov_df = pd.read_csv(prov_file) if prov_file.name.endswith(".csv") else pd.read_excel(prov_file)

    if st.button("🚀 Start Mapping"):
        hf_api = HuggingFaceAPI(api_key)

        with st.spinner("Embedding reference fields..."):
            ref_texts = ref_df['fields'].dropna().astype(str).tolist()
            ref_embeddings = hf_api.get_embeddings(ref_texts)
            if not ref_embeddings:
                st.error("Failed to embed reference fields")
                st.stop()

        with st.spinner("Embedding provider fields..."):
            prov_fields = prov_df.columns.astype(str).tolist()
            prov_embeddings = hf_api.get_embeddings(prov_fields)
            if not prov_embeddings:
                st.error("Failed to embed provider fields")
                st.stop()

        mappings = []
        with st.spinner("Calculating similarities..."):
            for i, p_vec in enumerate(prov_embeddings):
                best_score, best_idx = -1, -1
                for j, r_vec in enumerate(ref_embeddings):
                    score = cosine_similarity(p_vec, r_vec)
                    if score > best_score:
                        best_score, best_idx = score, j
                mappings.append({
                    "Provider Field": prov_fields[i],
                    "XML Field": ref_df.iloc[best_idx]['xml field'],
                    "Confidence": f"{best_score*100:.1f}%",
                    "Status": "✅ High" if best_score >= confidence_threshold else "⚠️ Low"
                })

        st.success("✅ Mapping Complete")
        st.dataframe(pd.DataFrame(mappings))

        if st.checkbox("🔍 Generate AI Explanations (slower)"):
            with st.spinner("Generating explanations..."):
                for m in mappings:
                    prompt = f"Explain why '{m['Provider Field']}' maps to '{m['XML Field']}'."
                    m["AI Explanation"] = hf_api.generate_text(prompt)
            st.dataframe(pd.DataFrame(mappings))

if __name__ == "__main__":
    main()
