# RAG QA Assistant

This folder contains a unified version of the Retrieval-Augmented Generation (RAG) QA assistant.  
Both the CLI and Streamlit applications share the same vector store and document ingestion pipeline.

## Folder Structure

RAG-QA-ASSISTANT/
 ├── src/
 │    ├── ingest.py
 │    ├── rag_pipeline.py
 │    ├── app_cli.py
 │    └── app_streamlit.py
 ├── data/
 │    └── documents/
 │         └── placeholder.txt
 ├── requirements.txt
 └── README.md