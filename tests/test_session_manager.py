
"""Tests for Prasad's persistent session manager."""

import pytest


from helpdesk.session_manager import SessionManager
from helpdesk.memory_adapter import MemoryAdapter
from helpdesk.ai_assistant import AIAssistant


@pytest.fixture
def manager(tmp_path):
    """Create a fresh temporary database for every test."""
    return SessionManager(tmp_path / "test_sessions.db")


def test_create_session(manager):
    session_id = manager.create_session()

    assert session_id.startswith("sess_")
    assert manager.has_session(session_id)


def test_save_and_retrieve_history(manager):
    session_id = manager.create_session()

    manager.add_message(
        session_id,
        {
            "role": "user",
            "content": "Which trains go to Bengaluru?"
        },
    )

    history = manager.get_history(session_id)

    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == (
        "Which trains go to Bengaluru?"
    )


def test_multiple_sessions_are_isolated(manager):
    session_1 = manager.create_session()
    session_2 = manager.create_session()

    manager.add_message(
        session_1,
        {"role": "user", "content": "Chennai"},
    )

    manager.add_message(
        session_2,
        {"role": "user", "content": "Mumbai"},
    )

    assert manager.get_history(session_1)[0]["content"] == "Chennai"
    assert manager.get_history(session_2)[0]["content"] == "Mumbai"


def test_clear_session(manager):
    session_id = manager.create_session()

    manager.add_message(
        session_id,
        {"role": "user", "content": "Hello"},
    )

    manager.clear_session(session_id)

    assert not manager.has_session(session_id)
    assert manager.get_history(session_id) == []


def test_unknown_session_rejected(manager):
    with pytest.raises(ValueError):
        manager.add_message(
            "unknown-session",
            {"role": "user", "content": "Hello"},
        )


def test_history_persists_after_reopening(tmp_path):
    database = tmp_path / "persistent.db"

    first_manager = SessionManager(database)
    session_id = first_manager.create_session()

    first_manager.add_message(
        session_id,
        {"role": "user", "content": "Tatkal booking"},
    )

    second_manager = SessionManager(database)

    history = second_manager.get_history(session_id)

    assert len(history) == 1
    assert history[0]["content"] == "Tatkal booking"


def test_memory_adapter_integration(manager):
    """Verify compatibility with Christy's MemoryAdapter."""
    memory = MemoryAdapter(backend=manager)

    session_id = memory.ensure_session()

    memory.record_user_message(
        session_id,
        "Which trains run from Chennai?",
    )

    memory.record_assistant_message(
        session_id,
        "I need the destination station.",
    )

    history = memory.get_context_messages(session_id)

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[0]["content"] == (
        "Which trains run from Chennai?"
    )


def test_ai_assistant_accepts_persistent_memory(manager):
    """Verify AIAssistant can use our SQLite session backend."""

    memory = MemoryAdapter(backend=manager)

    # A real LLM is not needed for this constructor test.
    assistant = AIAssistant(
        llm_client=None,
        memory_adapter=memory,
    )

    assert assistant.memory_adapter is memory

    session_id = assistant.memory_adapter.ensure_session()

    assistant.memory_adapter.record_user_message(
        session_id,
        "What is the train schedule?",
    )

    # Simulate reopening the application.
    reopened_manager = SessionManager(manager.db_path)
    reopened_memory = MemoryAdapter(backend=reopened_manager)

    history = reopened_memory.get_context_messages(session_id)

    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is the train schedule?"


def test_multi_turn_conversation_persistence(manager):
    """Verify multiple conversation turns persist in order."""

    memory = MemoryAdapter(backend=manager)
    session_id = memory.ensure_session()

    # First conversation turn
    memory.record_user_message(
        session_id,
        "I want to travel from Pune to Mumbai.",
    )
    memory.record_assistant_message(
        session_id,
        "What is your travel date?",
    )

    # Second conversation turn
    memory.record_user_message(
        session_id,
        "Tomorrow.",
    )
    memory.record_assistant_message(
        session_id,
        "I will need to check the timetable.",
    )

    # Reopen the database to simulate an application restart.
    reopened_manager = SessionManager(manager.db_path)
    reopened_memory = MemoryAdapter(backend=reopened_manager)

    history = reopened_memory.get_context_messages(session_id)

    assert len(history) == 4

    assert [message["role"] for message in history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    assert history[0]["content"] == (
        "I want to travel from Pune to Mumbai."
    )
    assert history[1]["content"] == (
        "What is your travel date?"
    )
    assert history[2]["content"] == "Tomorrow."
    assert history[3]["content"] == (
        "I will need to check the timetable."
    )
