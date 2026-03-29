from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


def process_docs(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    # ✅ FREE embeddings (no OpenAI)
    embeddings = HuggingFaceEmbeddings()

    db = FAISS.from_documents(texts, embeddings)

    return db


def ask_question(db, query):
    docs = db.similarity_search(query)

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    chain = load_qa_chain(llm, chain_type="stuff")

    return chain.run(input_documents=docs, question=query)