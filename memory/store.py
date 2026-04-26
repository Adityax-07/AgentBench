"""
MEMORY STORE (Two-Tier Persistence)
===================================

Purpose:
Provides long-term memory for the agent system.

Why this exists:
LLM agents are stateless by default.  
This module lets the system remember:
• past sessions
• previous queries
• generated reports

We use SQLite via SqliteDict for simplicity + persistence.
No external DB required.
"""

# =========================
# Imports
# =========================
from sqlitedict import SqliteDict
from datetime import datetime


# =========================
# Memory Store Class
# =========================
class MemoryStore:
    """
    Lightweight persistent memory layer.

    Uses SqliteDict:
    • behaves like a Python dictionary
    • stored on disk using SQLite
    • auto-persists data between runs
    """

    def __init__(self, db_path="memory/sessions.db"):
        # autocommit=True ensures writes are instantly saved.
        # Prevents data loss if the program crashes.
        self.db = SqliteDict(db_path, autocommit=True)


    # =========================
    # Save a completed session
    # =========================
    def save_session(self, session_id: str, query: str, report: str):
        """
        Store final output of an agent run.

        session_id → unique identifier (user/session/thread)
        query → original user request
        report → final generated answer
        """

        self.db[session_id] = {
            "query": query,
            "report": report,

            # Timestamp helps with ordering + analytics later
            "timestamp": datetime.now().isoformat()
        }


    # =========================
    # Retrieve session history
    # =========================
    def get_history(self, session_id: str) -> dict:
        """
        Fetch stored data for a specific session.
        Returns empty dict if session not found.
        """

        return self.db.get(session_id, {})


    # =========================
    # Get all stored sessions
    # =========================
    def get_all_sessions(self) -> dict:
        """
        Useful for:
        • dashboards
        • analytics
        • debugging
        """

        return dict(self.db)