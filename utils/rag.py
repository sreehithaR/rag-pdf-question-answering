from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
import streamlit as st


@st.cache_resource
def load_embeddings():
    """Load embedding model once and cache it."""
    return HuggingFaceEmbeddings()


@st.cache_resource
def load_llm():
    """Load the LLM pipeline once and cache it."""
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=256,
    )
    return HuggingFacePipeline(pipeline=pipe)


def process_docs(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    embeddings = load_embeddings()
    db = FAISS.from_documents(texts, embeddings)

    return db


def ask_question(db, query):
    llm = load_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
    )

    return qa.run(query)
