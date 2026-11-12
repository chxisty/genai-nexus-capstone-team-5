"""
helpdesk/pipeline.py
--------------------
Central integration pipeline wiring all team modules into a unified Railway Passenger Help Desk.

Components wired:
- Member 1 (Christy): AIAssistant, LLMClient, AnswerGenerator, MemoryAdapter
- Member 2 (Krishna): SQLite Timetable DB (data/rail_timetable.db), tools.database_tools
- Member 3 (Neha): ToolRegistry, TOOL_SCHEMAS (tools.schemas)
- Member 4 (Prasad): SessionManager (SQLite), ContextHandler, ReliableHelpDesk
- Member 5 (Ram): Pipeline orchestration, unified ask() entry point, terminal interface
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from helpdesk.ai_assistant import AIAssistant
from helpdesk.context_handler import ContextHandler
from helpdesk.llm import LLMClient
from helpdesk.memory_adapter import MemoryAdapter
from helpdesk.reliability import ReliableHelpDesk
from helpdesk.session_manager import SessionManager
from tools.database_tools import register_database_tools
from tools.registry import ToolRegistry
from tools.schemas import TOOL_SCHEMAS

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "rail_timetable.db"
DEFAULT_SESSION_DB = Path(__file__).resolve().parents[1] / "data" / "sessions.db"


class RailwayHelpDeskPipeline:
    """Integrated pipeline managing tools, persistent sessions, LLM, and reliability."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        db_path: str | Path = DEFAULT_DB_PATH,
        session_db_path: str | Path = DEFAULT_SESSION_DB,
        max_tool_steps: int = 5,
        extra_system_instructions: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.session_db_path = Path(session_db_path)

        # 1. Initialize ToolRegistry and register real SQLite database tools (Krishna & Neha)
        self.registry = ToolRegistry()
        if self.db_path.exists():
            register_database_tools(self.registry)
            logger.info("Registered real database tools from %s", self.db_path)
        else:
            logger.warning("Database file %s not found. Database tools unregistered.", self.db_path)

        # 2. Initialize persistent SQLite session storage & memory adapter (Prasad & Christy)
        self.session_manager = SessionManager(db_path=self.session_db_path)
        self.memory_adapter = MemoryAdapter(backend=self.session_manager)

        # 3. Initialize LLM client & AI assistant (Christy)
        self.llm_client = llm_client or LLMClient()
        self.assistant = AIAssistant(
            llm_client=self.llm_client,
            tool_executor=self.registry,
            tool_schemas=TOOL_SCHEMAS,
            memory_adapter=self.memory_adapter,
            max_tool_steps=max_tool_steps,
            extra_system_instructions=extra_system_instructions,
        )

        # 4. Context handler and response reliability validation wrapper (Prasad)
        self.context_handler = ContextHandler()
        self.reliable_helpdesk = ReliableHelpDesk(
            assistant=self.assistant,
            context_handler=self.context_handler,
        )

    def ask(self, question: str, *, session_id: str | None = None) -> dict[str, Any]:
        """Ask a passenger question and return the validated 4-field response dictionary."""
        return self.reliable_helpdesk.ask(question, session_id=session_id)

    def get_history(self, session_id: str | None) -> list[dict[str, Any]]:
        """Retrieve full conversation history for a given session."""
        if not session_id:
            return []
        return self.session_manager.get_history(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self.session_manager.clear_session(session_id)


_default_pipeline: RailwayHelpDeskPipeline | None = None


def get_pipeline(
    *,
    llm_client: LLMClient | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    session_db_path: str | Path = DEFAULT_SESSION_DB,
    force_new: bool = False,
) -> RailwayHelpDeskPipeline:
    """Get or create the singleton pipeline instance."""
    global _default_pipeline
    if _default_pipeline is None or force_new:
        _default_pipeline = RailwayHelpDeskPipeline(
            llm_client=llm_client,
            db_path=db_path,
            session_db_path=session_db_path,
        )
    return _default_pipeline


def ask(question: str, *, session_id: str | None = None) -> dict[str, Any]:
    """Public helper function to ask a question through the integrated help desk."""
    pipeline = get_pipeline()
    return pipeline.ask(question, session_id=session_id)
