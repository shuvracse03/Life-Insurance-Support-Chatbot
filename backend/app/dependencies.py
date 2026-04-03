"""
app/dependencies.py — Shared FastAPI dependencies (compiled agent graph).

The agent is built once at startup and stored as a module-level singleton,
avoiding lru_cache issues with hot-reloads.
"""
from agent.graph import build_graph
from memory.store import get_checkpointer

# Module-level singleton — built once when the app starts
_agent = None


def get_agent():
    """Return the compiled LangGraph agent, building it on first call."""
    global _agent
    if _agent is None:
        checkpointer = get_checkpointer()
        _agent = build_graph(checkpointer)
    return _agent
