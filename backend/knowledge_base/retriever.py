"""
knowledge_base/retriever.py — Expose a LangChain retriever backed by FAISS.
"""
from functools import lru_cache
from langchain_core.vectorstores import VectorStoreRetriever

from knowledge_base.loader import load_vector_store


@lru_cache(maxsize=1)
def get_retriever(k: int = 4) -> VectorStoreRetriever:
    """Return a cached retriever that fetches the top-k relevant chunks."""
    vectorstore = load_vector_store()
    return vectorstore.as_retriever(search_kwargs={"k": k})
