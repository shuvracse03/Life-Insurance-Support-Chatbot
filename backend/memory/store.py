"""
memory/store.py — SQLite-backed LangGraph checkpointer for conversation persistence.
"""
import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = Path(__file__).parent.parent / "data" / "conversations.db"


def get_checkpointer() -> SqliteSaver:
    """Return a SqliteSaver checkpointer using a persistent connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    return SqliteSaver(conn)
