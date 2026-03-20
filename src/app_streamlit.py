import streamlit as st
import os
import pickle
from rag_pipeline import retrieve_chunks, build_prompt, generate_answer
from ingest import process_uploaded_file, ingest_all
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_vectorstore():
    index_path = "vector_store/index.faiss"
    pkl_path = "vector_store/index.pkl"

    if not os.path.exists(index_path) or not os.path.exists(pkl_path):
        return None

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    index = data["index"]
    docstore = data["docstore"]
    index_to_docstore_id = data["index_to_docstore_id"]

    vectorstore = FAISS(
        embedding_function=get_embeddings(),
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    return vectorstore


st.title("QA Assistant")

st.sidebar.header("Upload Documents")
uploaded = st.sidebar.file_uploader("Upload a file", type=["pdf", "txt", "md", "docx"])

if uploaded:
    process_uploaded_file(uploaded)
    st.sidebar.success("Document uploaded and vector store rebuilt.")
st.sidebar.markdown("### Available Documents")

supported = [".pdf", ".txt", ".md", ".docx"]
docs = [
    f for f in os.listdir("src/documents")
    if os.path.splitext(f)[1].lower() in supported
]

if docs:
    for d in docs:
        st.sidebar.write(f"- {d}")
else:
    st.sidebar.write("No documents uploaded yet.")

st.sidebar.header("Maintenance")
if st.sidebar.button("Rebuild Vector Store (Full Ingest)"):
    ingest_all()
    st.sidebar.success("Vector store rebuilt successfully.")

vectorstore = load_vectorstore()

if vectorstore is None:
    st.warning("No vector store found. Upload documents or rebuild.")
else:
    user_input = st.text_input("Ask a question")

    if user_input:
        retrieved = retrieve_chunks(vectorstore, user_input, top_k=4)
        answer = generate_answer(user_input, retrieved)
        st.write(answer)

        with st.expander("Retrieved Chunks"):
            for c in retrieved:
                st.write(c)