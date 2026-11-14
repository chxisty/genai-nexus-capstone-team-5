
"""Persistent session management for the Railway Passenger Help Desk.

Member 4: Prasad Tribhuvan
Compatible with helpdesk.memory_adapter.MemoryBackendProtocol.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import uuid
from typing import Any


class SessionManager:
    """Store independent conversation sessions in SQLite."""

    def __init__(self, db_path: str | Path = "data/sessions.db") -> None:
        self.db_path = str(db_path)

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    FOREIGN KEY (session_id)
                        REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id)
                """
            )

    def create_session(self, session_id: str | None = None) -> str:
        """Create a session or return an existing session ID."""
        sid = (session_id or "").strip()

        if not sid:
            sid = f"sess_{uuid.uuid4().hex}"

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id)
                VALUES (?)
                """,
                (sid,),
            )

        return sid

    def has_session(self, session_id: str) -> bool:
        """Check whether a session exists."""
        if not session_id:
            return False

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        return row is not None

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return messages in chronological order."""
        if not session_id:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT message_json
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return [json.loads(row[0]) for row in rows]

    def add_message(
        self,
        session_id: str,
        message: dict[str, Any],
    ) -> None:
        """Append a message to an existing session."""
        if not self.has_session(session_id):
            raise ValueError(f"Unknown session: {session_id}")

        if not isinstance(message, dict):
            raise TypeError("Message must be a dictionary.")

        serialized = json.dumps(
            deepcopy(message),
            ensure_ascii=False,
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (session_id, message_json)
                VALUES (?, ?)
                """,
                (session_id, serialized),
            )

    def clear_session(self, session_id: str) -> None:
        """Delete the session and its stored messages."""
        if not session_id:
            return

        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            )
