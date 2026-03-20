import os
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama


VECTOR_DIR = "vector_store"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_vectorstore():
    index_path = os.path.join(VECTOR_DIR, "index.faiss")
    pkl_path = os.path.join(VECTOR_DIR, "index.pkl")

    if not os.path.exists(index_path) or not os.path.exists(pkl_path):
        raise FileNotFoundError("Vector store not found. Run ingest.py first.")

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


def retrieve_chunks(vectorstore, query, top_k=4):
    docs = vectorstore.similarity_search(query, k=top_k)

    results = []
    for d in docs:
        results.append(
            {
                "text": d.page_content,
                "source": d.metadata.get("source", "unknown"),
            }
        )
    return results


def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(
        [f"Source: {c['source']}\n{c['text']}" for c in retrieved_chunks]
    )

    return f"""
Use ONLY the context below to answer.

Context:
{context}

Question: {question}

Answer:
"""


def generate_answer(question, retrieved_chunks, model_name="llama3"):
    prompt = build_prompt(question, retrieved_chunks)
    llm = Ollama(model=model_name)
    return llm.invoke(prompt)