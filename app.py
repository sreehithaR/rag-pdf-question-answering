import os
import tempfile
import streamlit as st
from utils.rag import process_docs, ask_question

st.title("📄 Knowledge Base Search Engine")

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    # Write to a temp directory (works on read-only filesystems)
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "uploaded.pdf")

    with open(tmp_path, "wb") as f:
        f.write(file.read())

    # Cache the vector DB in session_state so it doesn't rebuild on every interaction
    if "db" not in st.session_state or st.session_state.get("file_name") != file.name:
        with st.spinner("Processing PDF..."):
            st.session_state.db = process_docs(tmp_path)
            st.session_state.file_name = file.name

    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Thinking..."):
            answer = ask_question(st.session_state.db, query)
        st.write(answer)
