import streamlit as st
import requests
import os

# ================= CONFIG =================
BACKEND_URL = "http://localhost:8000"   # FastAPI base URL
st.set_page_config(page_title="PDF RAG Form Filler", layout="wide")

# ================= SESSION =================
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ================= UI =================
st.title("📄 PDF RAG Form Filler")
st.caption("Auto-fill PDF forms using MongoDB Vector Search + Perplexity AI")

tab1, tab2, tab3 = st.tabs([
    "📥 Fill PDF with RAG",
    "🧠 Learn from PDF / DOCX",
    "🔍 Extract from Static PDF"
])

# =====================================================
# TAB 1: FILL PDF WITH RAG
# =====================================================
with tab1:
    st.subheader("📥 Upload PDF to Auto-Fill")

    pdf_file = st.file_uploader(
        "Upload Fillable PDF",
        type=["pdf"],
        key="fill_pdf"
    )

    if pdf_file and st.button("🚀 Fill PDF"):
        with st.spinner("Filling PDF using RAG..."):
            files = {"file": pdf_file}
            resp = requests.post(
                f"{BACKEND_URL}/api/fill-pdf-with-rag/",
                files=files
            )

        if resp.status_code == 200:
            data = resp.json()

            st.success(data["message"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Filled PDF")
                st.markdown(
                    f"[Download Filled PDF]({BACKEND_URL}/{data['output']})"
                )

            with col2:
                st.markdown("### 🔴 Highlighted PDF")
                st.markdown(
                    f"[Download Highlighted PDF]({BACKEND_URL}/{data['highlightedOutput']})"
                )

            st.markdown("### 📊 Stats")
            st.json(data["stats"])

            st.markdown("### 🧾 Sample Logs")
            st.json(data["logs"])
        else:
            st.error(resp.text)

# =====================================================
# TAB 2: LEARN FROM PDF / DOCX
# =====================================================
with tab2:
    st.subheader("🧠 Teach the System (Learning Mode)")

    category = st.text_input("Category (optional)", placeholder="e.g. HR Forms")

    learn_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
        key="learn_pdf"
    )

    if learn_file and st.button("📚 Learn from Document"):
        with st.spinner("Learning from document..."):
            files = {"file": learn_file}
            data = {"category": category}

            resp = requests.post(
                f"{BACKEND_URL}/api/learn-from-pdf/",
                files=files,
                data=data
            )

        if resp.status_code == 200:
            st.success("Learning completed successfully")
            st.json(resp.json())
        else:
            st.error(resp.text)

# =====================================================
# TAB 3: EXTRACT FROM STATIC PDF
# =====================================================
with tab3:
    st.subheader("🔍 Extract Data from Static PDF")

    static_pdf = st.file_uploader(
        "Upload Static (Non-fillable) PDF",
        type=["pdf"],
        key="static_pdf"
    )

    if static_pdf and st.button("📤 Extract Fields"):
        with st.spinner("Extracting data from static PDF..."):
            files = {"file": static_pdf}

            resp = requests.post(
                f"{BACKEND_URL}/api/extract-data-from-static-pdf/",
                files=files
            )

        if resp.status_code == 200:
            data = resp.json()
            st.success("Extraction successful")

            st.markdown("### 📄 Extracted Data")
            st.json(data["data"])

            st.markdown(f"**Total Fields Detected:** {data['fields_detected']}")
        else:
            st.error(resp.text)

# ================= FOOTER =================
st.markdown("---")
st.caption("Built using FastAPI • MongoDB Vector Search • Perplexity AI • Streamlit")
