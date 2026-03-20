import os
import pickle
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

DOCUMENT_DIR = "src/documents"
VECTOR_DIR = "vector_store"

os.makedirs(DOCUMENT_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


def load_document(path):
    ext = path.lower()

    if ext.endswith(".pdf"):
        return PyPDFLoader(path).load()
    if ext.endswith(".txt") or ext.endswith(".md"):
        return TextLoader(path).load()
    if ext.endswith(".docx"):
        return Docx2txtLoader(path).load()

    raise ValueError(f"Unsupported file type: {path}")


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def save_vectorstore(vectorstore):
    faiss.write_index(
        vectorstore.index,
        f"{VECTOR_DIR}/index.faiss",
    )

    data = {
        "index": vectorstore.index,
        "docstore": vectorstore.docstore,
        "index_to_docstore_id": vectorstore.index_to_docstore_id,
    }

    with open(f"{VECTOR_DIR}/index.pkl", "wb") as f:
        pickle.dump(data, f)


def rebuild_vectorstore_from_all_documents():
    all_docs = []

    for filename in os.listdir(DOCUMENT_DIR):
        path = os.path.join(DOCUMENT_DIR, filename)
        docs = load_document(path)
        all_docs.extend(docs)

    if not all_docs:
        print("No documents found in", DOCUMENT_DIR)
        return None

    chunks = chunk_documents(all_docs)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    save_vectorstore(vectorstore)
    return vectorstore


def process_uploaded_file(uploaded_file):
    save_path = os.path.join(DOCUMENT_DIR, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    rebuild_vectorstore_from_all_documents()


def ingest_all():
    vectorstore = rebuild_vectorstore_from_all_documents()
    if vectorstore:
        print("Ingestion complete. Vector store updated.")
    else:
        print("No documents to ingest.")


if __name__ == "__main__":
    ingest_all()