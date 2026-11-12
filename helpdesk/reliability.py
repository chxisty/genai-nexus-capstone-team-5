
"""Context-aware response reliability integration.

Member 4: Context Management and Response Reliability.
"""

from __future__ import annotations

from typing import Any, Mapping

from helpdesk.ai_assistant import AIAssistant
from helpdesk.context_handler import ContextHandler
from helpdesk.memory_adapter import MemoryAdapter


class ReliableHelpDesk:
    """Wrap the existing assistant with context and response checks."""

    def __init__(
        self,
        assistant: AIAssistant,
        context_handler: ContextHandler | None = None,
    ) -> None:
        self.assistant = assistant
        self.context_handler = context_handler or ContextHandler()

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a question with clarification and output validation."""

        memory: MemoryAdapter = self.assistant.memory_adapter

        sid = memory.ensure_session(session_id)
        history = memory.get_context_messages(sid)

        analysis = self.context_handler.analyze(question, history)

        if analysis.needs_clarification:
            result = {
                "answer": analysis.clarification_question
                or "Please clarify your railway question.",
                "sources": [],
                "sql": None,
                "dont_know": True,
            }

            memory.record_user_message(sid, question)
            memory.record_assistant_message(
                sid,
                result["answer"],
                metadata={
                    "sources": [],
                    "sql": None,
                    "dont_know": True,
                },
            )

            return result

        result = self.assistant.ask(question, session_id=sid)

        return self.validate_response(result)

    @staticmethod
    def validate_response(
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Check the required four-field answer contract."""

        if not isinstance(response, Mapping):
            raise ValueError("Response must be a dictionary.")

        required = {"answer", "sources", "sql", "dont_know"}

        if set(response.keys()) != required:
            raise ValueError(
                "Response must contain exactly: "
                "answer, sources, sql, dont_know."
            )

        answer = response["answer"]
        sources = response["sources"]
        sql = response["sql"]
        dont_know = response["dont_know"]

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer must be a non-empty string.")

        if not isinstance(sources, list):
            raise ValueError("Sources must be a list.")

        for source in sources:
            if not isinstance(source, Mapping):
                raise ValueError("Each source must be a dictionary.")

            if set(source.keys()) != {"doc", "page"}:
                raise ValueError(
                    "Each source must contain doc and page."
                )

            if not isinstance(source["doc"], str):
                raise ValueError("Source doc must be a string.")

            page = source["page"]

            if page is not None and (
                isinstance(page, bool)
                or not isinstance(page, int)
                or page < 1
            ):
                raise ValueError(
                    "Source page must be a positive integer or None."
                )

        if sql is not None and not isinstance(sql, str):
            raise ValueError("SQL must be a string or None.")

        if not isinstance(dont_know, bool):
            raise ValueError("dont_know must be a boolean.")

        return dict(response)
