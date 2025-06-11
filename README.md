# 🧠 Provider Data Mapper (AI-Powered Demo)

This is a working demo of a **smart provider field mapping tool** using **RAG + LLM**, powered by Hugging Face's Mistral 7B model.

Built with ❤️ using **Streamlit** for a beautiful web UI.

---

## ✅ What It Does

- Upload a **provider roster** (CSV/Excel)
- Upload one or more **RAG mapping files** (CSV)
- Automatically maps fields to XML using:
  - Embeddings
  - FAISS search
  - Hugging Face LLM for logic explanations
- Produces:
  - A **detailed mapping report**
  - A **generated XML output**
  - Downloadable files for both

---

## 🚀 How to Use (Client Instructions)

1. Click "📁 Upload RAG Mapping File(s)"  
2. Upload `sample_rag_mapping.csv`
3. Click "📄 Upload Provider Data"  
4. Upload `sample_provider_input.csv`
5. Click **"🚀 Process Mapping"**
6. View the **AI-generated mapping logic**
7. Download XML and report

---

## 📁 File Formats

### 🔹 sample_rag_mapping.csv
```csv
fields,xml field,logic,comments
NPI,provider/npi,Direct mapping,National Provider Identifier
...
