import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import requests
import tempfile
import time

# Get Hugging Face API key from Streamlit secrets
hf_api_key = st.secrets["HUGGINGFACE_TOKEN"]

# Hugging Face API endpoints
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Changed to a more reliable generation model for Inference API
GENERATION_MODEL = "microsoft/DialoGPT-medium"  # Alternative: "gpt2" or "facebook/blenderbot-400M-distill"

def get_embeddings(texts, api_key):
    """Get embeddings from Hugging Face Inference API."""
    # Correct URL format for Hugging Face Inference API
    API_URL = f"https://api-inference.huggingface.co/models/{EMBEDDING_MODEL}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Add retry logic for model loading
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Send the texts as inputs
            payload = {"inputs": texts}
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 503:
                # Model is loading, wait and retry
                wait_time = 20 * (attempt + 1)  # Exponential backoff
                st.warning(f"Model loading, waiting {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif response.status_code == 404:
                st.error(f"Model {EMBEDDING_MODEL} not found. Trying alternative...")
                # Try alternative embedding model
                return get_embeddings_fallback(texts, api_key)
                
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list):
                return result
            else:
                st.error(f"Unexpected response format: {type(result)}")
                return get_embeddings_fallback(texts, api_key)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Embedding API error (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                return get_embeddings_fallback(texts, api_key)
            time.sleep(5)
    
    return get_embeddings_fallback(texts, api_key)

def get_embeddings_fallback(texts, api_key):
    """Fallback embedding method using alternative models."""
    fallback_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]
    
    for model in fallback_models:
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {"inputs": texts}
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list):
                    st.success(f"Using fallback embedding model: {model}")
                    return result
            elif response.status_code == 503:
                st.warning(f"Model {model} is loading, trying next...")
                continue
                
        except Exception as e:
            st.warning(f"Fallback model {model} failed: {str(e)}")
            continue
    
    # If all embedding models fail, create simple fallback embeddings
    st.warning("All embedding models failed. Using simple text-based similarity.")
    return create_simple_embeddings(texts)

def create_simple_embeddings(texts):
    """Create simple embeddings based on text similarity when API fails."""
    import hashlib
    
    # Simple word-based embeddings as fallback
    embeddings = []
    for text in texts:
        # Create a simple vector based on character frequencies and word lengths
        text_lower = str(text).lower()
        
        # Create a 384-dimensional vector (matching typical embedding size)
        embedding = [0.0] * 384
        
        # Fill with simple features
        for i, char in enumerate(text_lower[:384]):
            embedding[i] = ord(char) / 128.0  # Normalize ASCII values
        
        # Add some word-based features
        words = text_lower.split()
        if words:
            avg_word_len = sum(len(word) for word in words) / len(words)
            embedding[0] = avg_word_len / 10.0  # Normalize
            embedding[1] = len(words) / 10.0    # Word count
            embedding[2] = len(text_lower) / 100.0  # Character count
        
        embeddings.append(embedding)
    
    return embeddings

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)

def generate_text(prompt, api_key):
    """Generate text using Hugging Face Inference API."""
    # Try multiple models as fallback
    models_to_try = [
        "microsoft/DialoGPT-medium",
        "gpt2",
        "facebook/blenderbot-400M-distill",
        "google/flan-t5-small"
    ]
    
    for model in models_to_try:
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # Adjust parameters based on model type
            if "flan-t5" in model:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 64,
                        "temperature": 0.7,
                        "do_sample": True
                    }
                }
            else:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": len(prompt.split()) + 64,
                        "temperature": 0.7,
                        "do_sample": True,
                        "pad_token_id": 50256  # For GPT-2 based models
                    }
                }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 503:
                st.warning(f"Model {model} is loading, trying next model...")
                continue
                
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        generated = result[0]['generated_text']
                        # Remove the original prompt from the response
                        if generated.startswith(prompt):
                            generated = generated[len(prompt):].strip()
                        return generated if generated else f"[Generated with {model}]"
                    elif isinstance(result[0], str):
                        return result[0]
                elif isinstance(result, dict):
                    if 'error' in result:
                        st.warning(f"Model {model} error: {result['error']}, trying next model...")
                        continue
                    else:
                        return str(result)
                else:
                    return f"[Response from {model}: {str(result)[:100]}...]"
            else:
                st.warning(f"Model {model} returned status {response.status_code}, trying next model...")
                continue
                
        except Exception as e:
            st.warning(f"Error with model {model}: {str(e)}, trying next model...")
            continue
    
    # If all models fail, return a fallback explanation
    return "[Unable to generate explanation - API models unavailable]"

st.title("🧠 Provider Data Mapper (RAG + LLM, Hugging Face API Only)")
st.markdown("Upload your mapping and provider files. The app will auto-map fields and generate XML using only the Hugging Face Inference API (no local ML dependencies).")

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
    st.info("Processing... please wait (API calls may take a few seconds)")
    
    try:
        # Get embeddings for RAG fields and provider columns
        rag_fields = rag_df['fields'].astype(str).tolist()
        prov_columns = prov_df.columns.astype(str).tolist()

        st.write("Getting embeddings for RAG fields...")
        rag_embeddings = get_embeddings(rag_fields, hf_api_key)
        
        st.write("Getting embeddings for provider columns...")
        prov_embeddings = get_embeddings(prov_columns, hf_api_key)

        import numpy as np
        results = []
        mapping_explanations = []

        progress_bar = st.progress(0)
        total_rows = len(prov_df)

        for i, row in prov_df.iterrows():
            entry = {}
            explain = {}
            
            for col_idx, col in enumerate(prov_df.columns):
                # Find best match in RAG fields using cosine similarity
                similarities = [
                    cosine_similarity(prov_embeddings[col_idx], rag_emb)
                    for rag_emb in rag_embeddings
                ]
                best_idx = int(np.argmax(similarities))
                best_match = rag_df.iloc[best_idx]
                xml_path = best_match['xml field']
                value = row[col]
                entry[xml_path] = value
                
                # Generate mapping logic explanation
                prompt = (
                    f"Explain why provider field '{col}' maps to XML field '{xml_path}'. "
                    f"Logic: {best_match.get('logic', 'N/A')}. "
                    f"Comments: {best_match.get('comments', 'N/A')}."
                )
                logic_explanation = generate_text(prompt, hf_api_key)
                explain[col] = logic_explanation
                
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
        
        st.success("✅ Processing completed!")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your API key and try again. If the problem persists, the Hugging Face models might be temporarily unavailable.")

st.markdown("---")
st.markdown("📌 **Sample Files:**")
st.markdown("- [sample_rag_mapping.csv](https://github.com/niranjpc/provider-mapper/blob/main/sample_rag_mapping.csv)")
st.markdown("- [sample_provider_input.csv](https://github.com/niranjpc/provider-mapper/blob/main/sample_provider_input.csv)")

st.markdown("---")
st.markdown("🔧 **Troubleshooting Tips:**")
st.markdown("""
- Ensure your Hugging Face API token is valid and set in Streamlit secrets
- The embedding model may take time to load on first use
- If generation fails, the app will try multiple models as fallbacks
- Check the Hugging Face status page if all models are unavailable
""")
