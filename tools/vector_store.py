"""
FAISS Vector Store Tool for RAG Applications
--------------------------------------------

This file provides a complete pipeline to:
1. Convert raw documents → embeddings
2. Build and persist a FAISS vector database
3. Load existing vector database
4. Retrieve semantically relevant chunks for LLM context

Designed for LangChain / AI Agent workflows.
"""

# =========================
# Imports
# =========================

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Union
import os


# =========================
# Global Embedding Model
# =========================
# WHY HuggingFace instead of OpenAI?
# Runs locally — no API key, no cost, no network call for embeddings.
# all-MiniLM-L6-v2 is fast, small, and accurate for semantic search.

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# Text Splitter (CRITICAL)
# =========================
# WHY:
# LLMs and embeddings work best with small semantic chunks.
# Chunk overlap preserves context between chunks.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # optimal chunk size for most LLMs
    chunk_overlap=150    # prevents loss of context at boundaries
)


# =========================
# Build Vector Store
# =========================

def build_store(
    docs: List[Union[str, Document]],
    save_path: str = "faiss_store"
) -> FAISS:
    """
    Build a FAISS vector store from documents and save locally.

    Parameters
    ----------
    docs : list[str] OR list[Document]
        Raw text documents or LangChain Document objects.

    save_path : str
        Directory where FAISS index will be stored.

    Returns
    -------
    FAISS vector store
    """

    # ---- Guardrail ----
    # Prevent silent failures if empty docs passed
    if not docs:
        raise ValueError("Document list is empty.")

    # ---- Convert strings → Document objects ----
    # WHY:
    # LangChain stores metadata inside Document objects.
    if isinstance(docs[0], str):
        docs = [Document(page_content=d) for d in docs]

    # ---- Split documents into chunks ----
    # WHY:
    # Embeddings on large text are noisy and inefficient.
    split_docs = text_splitter.split_documents(docs)

    print(f"Created {len(split_docs)} text chunks.")

    # ---- Create FAISS vector store ----
    # This step:
    # 1. Generates embeddings
    # 2. Builds similarity index
    vector_store = FAISS.from_documents(split_docs, embeddings)

    # ---- Persist to disk ----
    # WHY:
    # Without saving, embeddings must be rebuilt every run.
    vector_store.save_local(save_path)
    print(f"Vector store saved at '{save_path}'")

    return vector_store


# =========================
# Load Existing Store
# =========================

def load_store(path: str = "faiss_store") -> FAISS:
    """
    Load a previously saved FAISS vector store.

    Returns None if no store has been built yet — the researcher
    will fall back to web search only in that case.
    """
    if not os.path.exists(path):
        print(f"[VectorStore] No store found at '{path}' — running without RAG.")
        return None

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================
# Retrieval Function
# =========================

def retrieve(query: str, store: FAISS, k: int = 4) -> str:
    """
    Retrieve top-k relevant chunks for a query.

    Parameters
    ----------
    query : str
        User question

    store : FAISS
        Loaded vector store

    k : int
        Number of chunks to retrieve

    Returns
    -------
    Formatted string ready for LLM context
    """

    # ---- Guardrails ----
    if not query or not query.strip():
        return "Empty query provided."

    # ---- Semantic search ----
    docs = store.similarity_search(query, k=k)

    if not docs:
        return "No relevant documents found."

    # ---- Format for LLM ----
    # WHY:
    # Structured context reduces hallucinations.
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(
            f"[Source {i}]\n{doc.page_content}"
        )

    return "\n\n".join(results)


# =========================
# Example Usage (CLI demo)
# =========================
# Run this file directly to test the pipeline.

if __name__ == "__main__":

    sample_docs = [
        "LangChain is a framework for building LLM powered apps.",
        "FAISS is a vector database developed by Facebook AI.",
        "Embeddings convert text into numerical vectors.",
        "RAG stands for Retrieval Augmented Generation."
    ]

    print("\n--- Building Vector Store ---")
    store = build_store(sample_docs)

    print("\n--- Loading Vector Store ---")
    store = load_store()

    print("\n--- Retrieval Demo ---")
    question = "What is FAISS?"
    context = retrieve(question, store)

    print("\nRetrieved Context:\n")
    print(context)