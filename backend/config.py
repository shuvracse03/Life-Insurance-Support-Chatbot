"""
config.py — Centralised configuration loaded from .env
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

_llm = None


def get_llm() -> ChatGroq:
    """Return a cached Groq ChatGroq LLM instance."""
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment / .env file")
        # llama-3.3-70b-versatile is the recommended replacement for the deprecated
        # llama3-groq-70b-8192-tool-use-preview and supports tool/function calling.
        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=api_key,
            temperature=0,
        )
    return _llm
