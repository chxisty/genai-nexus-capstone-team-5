
"""Tests for Member 4 response reliability module."""

import pytest

from helpdesk.reliability import ReliableHelpDesk


def test_valid_response():
    """Accept a response following the required schema."""

    response = {
        "answer": "Train information is available.",
        "sources": [
            {"doc": "railway_rules.pdf", "page": 5}
        ],
        "sql": None,
        "dont_know": False,
    }

    result = ReliableHelpDesk.validate_response(response)

    assert result == response


def test_valid_unknown_response():
    """Accept a safe refusal when evidence is unavailable."""

    response = {
        "answer": "I don't know based on the available evidence.",
        "sources": [],
        "sql": None,
        "dont_know": True,
    }

    result = ReliableHelpDesk.validate_response(response)

    assert result["dont_know"] is True


def test_missing_required_field():
    """Reject responses missing a required field."""

    response = {
        "answer": "Train information.",
        "sources": [],
        "sql": None,
    }

    with pytest.raises(ValueError):
        ReliableHelpDesk.validate_response(response)


def test_invalid_answer_type():
    """Reject answers that are not strings."""

    response = {
        "answer": 123,
        "sources": [],
        "sql": None,
        "dont_know": False,
    }

    with pytest.raises(ValueError):
        ReliableHelpDesk.validate_response(response)


def test_invalid_sources_type():
    """Reject sources that are not lists."""

    response = {
        "answer": "Train information.",
        "sources": "railway_rules.pdf",
        "sql": None,
        "dont_know": False,
    }

    with pytest.raises(ValueError):
        ReliableHelpDesk.validate_response(response)


def test_invalid_source_page():
    """Reject invalid citation page numbers."""

    response = {
        "answer": "Train information.",
        "sources": [
            {"doc": "railway_rules.pdf", "page": -2}
        ],
        "sql": None,
        "dont_know": False,
    }

    with pytest.raises(ValueError):
        ReliableHelpDesk.validate_response(response)


def test_invalid_sql_type():
    """Reject SQL values that are not strings or None."""

    response = {
        "answer": "Train information.",
        "sources": [],
        "sql": 123,
        "dont_know": False,
    }

    with pytest.raises(ValueError):
        ReliableHelpDesk.validate_response(response)


def test_invalid_dont_know_type():
    """Reject non-boolean dont_know values."""

    response = {
        "answer": "Train information.",
        "sources": [],
        "sql": None,
        "dont_know": "False",
    }

    with pytest.raises(ValueError):
        ReliableHelpDesk.validate_response(response)


def test_reliable_helpdesk_integration(tmp_path):
    """Test clarification using real session and memory components."""

    from helpdesk.ai_assistant import AIAssistant
    from helpdesk.context_handler import ContextHandler
    from helpdesk.memory_adapter import MemoryAdapter
    from helpdesk.session_manager import SessionManager

    # Use a temporary SQLite database for testing.
    session_store = SessionManager(
        db_path=str(tmp_path / "test_sessions.db")
    )

    memory = MemoryAdapter(backend=session_store)

    # No live LLM call is needed for the clarification path.
    assistant = AIAssistant(
        llm_client=None,
        memory_adapter=memory,
    )

    reliable = ReliableHelpDesk(
        assistant=assistant,
        context_handler=ContextHandler(),
    )

    session_id = memory.ensure_session("integration_test")

    memory.record_user_message(
        session_id,
        "Tell me about trains 12002 and 12951.",
    )

    memory.record_assistant_message(
        session_id,
        "Which train would you like to know about?",
    )

    result = reliable.ask(
        "What time does that train depart?",
        session_id=session_id,
    )

    assert result["dont_know"] is True
    assert result["sources"] == []
    assert result["sql"] is None
    assert "clarify" in result["answer"].lower() or (
        "provide more details" in result["answer"].lower()
    )

    # Confirm the conversation was saved in SQLite.
    history = memory.get_raw_history(session_id)

    assert len(history) == 4
    assert history[-2]["role"] == "user"
    assert history[-1]["role"] == "assistant"


def test_reliable_helpdesk_normal_answer_path(tmp_path):
    """A clear question should reach the assistant and be validated."""

    from helpdesk.memory_adapter import MemoryAdapter
    from helpdesk.session_manager import SessionManager

    class FakeAssistant:
        def __init__(self, memory_adapter):
            self.memory_adapter = memory_adapter
            self.received_question = None
            self.received_session_id = None

        def ask(self, question, *, session_id=None):
            self.received_question = question
            self.received_session_id = session_id

            return {
                "answer": "The timetable lists the requested train.",
                "sources": [],
                "sql": "SELECT train_number FROM trains LIMIT 1",
                "dont_know": False,
            }

    session_store = SessionManager(
        db_path=str(tmp_path / "normal_path.db")
    )
    memory = MemoryAdapter(backend=session_store)

    assistant = FakeAssistant(memory)
    reliable = ReliableHelpDesk(assistant=assistant)

    result = reliable.ask(
        "Show me the available train information.",
        session_id="normal_test",
    )

    assert result["answer"] == (
        "The timetable lists the requested train."
    )
    assert result["dont_know"] is False
    assert result["sql"] == (
        "SELECT train_number FROM trains LIMIT 1"
    )
    assert assistant.received_question == (
        "Show me the available train information."
    )
    assert assistant.received_session_id == "normal_test"
