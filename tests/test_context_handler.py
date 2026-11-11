
"""Unit tests for Railway Passenger Help Desk context handling."""

import pytest

from helpdesk.context_handler import ContextHandler


@pytest.fixture
def handler():
    return ContextHandler()


def test_independent_question(handler):
    result = handler.analyze(
        "What is the departure time of train 12002?"
    )

    assert result.is_follow_up is False
    assert result.needs_clarification is False
    assert result.relevant_history == []


def test_follow_up_with_history(handler):
    history = [
        {
            "role": "user",
            "content": "I want to travel from Pune to Mumbai.",
        },
        {
            "role": "assistant",
            "content": "What is your travel date?",
        },
    ]

    result = handler.analyze("Tomorrow", history)

    assert result.is_follow_up is True
    assert result.needs_clarification is False
    assert len(result.relevant_history) == 2


def test_follow_up_without_history(handler):
    result = handler.analyze("What about tomorrow?")

    assert result.is_follow_up is True
    assert result.needs_clarification is True
    assert result.clarification_question is not None


def test_empty_question(handler):
    result = handler.analyze("   ")

    assert result.needs_clarification is True
    assert result.question == ""


def test_history_limit():
    handler = ContextHandler(max_history_messages=2)

    history = [
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Message 2"},
        {"role": "user", "content": "Message 3"},
        {"role": "assistant", "content": "Message 4"},
    ]

    result = handler.analyze("What about tomorrow?", history)

    assert len(result.relevant_history) == 2
    assert result.relevant_history[0]["content"] == "Message 3"
    assert result.relevant_history[1]["content"] == "Message 4"


def test_tool_messages_are_excluded(handler):
    history = [
        {"role": "user", "content": "Find my train"},
        {"role": "tool", "content": "Database result"},
        {"role": "assistant", "content": "Which train number?"},
    ]

    result = handler.analyze("That one", history)

    assert result.is_follow_up is True
    assert len(result.relevant_history) == 2
    assert all(
        message["role"] != "tool"
        for message in result.relevant_history
    )


def test_no_history_when_limit_zero():
    handler = ContextHandler(max_history_messages=0)

    history = [
        {"role": "user", "content": "Pune to Mumbai"},
    ]

    result = handler.analyze("Tomorrow", history)

    assert result.relevant_history == []
    assert result.needs_clarification is True


def test_invalid_history_limit():
    with pytest.raises(ValueError):
        ContextHandler(max_history_messages=-1)


def test_ambiguous_reference_with_irrelevant_history(handler):
    """Do not treat unrelated history as sufficient context."""

    history = [
        {
            "role": "user",
            "content": "Hello!",
        },
        {
            "role": "assistant",
            "content": "How can I help you?",
        },
    ]

    result = handler.analyze(
        "What about that train?",
        history,
    )

    assert result.is_follow_up is True
    assert result.needs_clarification is True
    assert result.clarification_question is not None



def test_ambiguous_train_without_specific_identity(handler):
    """A general train discussion must not identify a specific train."""

    history = [
        {
            "role": "user",
            "content": "Tell me about trains from Pune to Mumbai.",
        },
        {
            "role": "assistant",
            "content": "There are multiple trains on this route.",
        },
    ]

    result = handler.analyze(
        "What time does that train depart?",
        history,
    )

    assert result.is_follow_up is True
    assert result.needs_clarification is True
    assert result.clarification_question is not None


def test_ambiguous_reference_with_multiple_trains(handler):
    """Ask for clarification when multiple trains are mentioned."""

    history = [
        {
            "role": "user",
            "content": "Tell me about trains 12002 and 12951.",
        },
        {
            "role": "assistant",
            "content": "Which train would you like to know more about?",
        },
    ]

    result = handler.analyze(
        "What time does that train depart?",
        history,
    )

    assert result.is_follow_up is True
    assert result.needs_clarification is True
    assert result.clarification_question is not None


def test_explicit_train_selection_resolves_ambiguity(handler):
    """A later explicit train selection should resolve ambiguity."""

    history = [
        {
            "role": "user",
            "content": "Tell me about trains 12002 and 12951.",
        },
        {
            "role": "assistant",
            "content": "Which train would you like to know about?",
        },
        {
            "role": "user",
            "content": "Train 12002.",
        },
        {
            "role": "assistant",
            "content": "What would you like to know about train 12002?",
        },
    ]

    result = handler.analyze(
        "What time does that train depart?",
        history,
    )

    assert result.is_follow_up is True
    assert result.needs_clarification is False
    assert result.clarification_question is None
