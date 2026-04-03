"""
knowledge_base/loader.py — Load, split, embed KB markdown files into FAISS.

Usage:
    from knowledge_base.loader import build_vector_store
    vectorstore = build_vector_store()
"""
import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = Path(__file__).parent / "data"
FAISS_INDEX_PATH = Path(__file__).parent / "faiss_index"

# Free local embeddings — no API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return the HuggingFace embedding model (cached after first load)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store() -> FAISS:
    """Load all markdown files, split into chunks, embed and save FAISS index."""
    docs = []
    for md_file in DATA_DIR.glob("*.md"):
        loader = TextLoader(str(md_file), encoding="utf-8")
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    print(f"✅ FAISS index built with {len(chunks)} chunks → {FAISS_INDEX_PATH}")
    return vectorstore


def load_vector_store() -> FAISS:
    """Load FAISS index from disk; build it first if it doesn't exist."""
    embeddings = _get_embeddings()
    if FAISS_INDEX_PATH.exists():
        return FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return build_vector_store()
