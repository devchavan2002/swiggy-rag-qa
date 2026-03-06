# Swiggy Annual Report RAG QA System

A Retrieval-Augmented Generation (RAG) based question answering system built using the Swiggy Annual Report.

## Features

- Document processing from PDF
- Semantic chunking and embeddings
- FAISS vector database
- Local LLM inference using Mistral
- Streamlit UI for user queries
- Context-grounded answers with source citations

## Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama (Mistral)
- Streamlit

## Architecture

User Query
    │
    ▼
Streamlit UI
    │
    ▼
Retriever (FAISS Vector DB)
    │
    ▼
Relevant Document Chunks
    │
    ▼
LLM (Mistral via Ollama)
    │
    ▼
Generated Answer + Source Pages


## Installation

```bash
pip install -r requirements.txt

## Run the Project

python build_index.py
streamlit run app.py
