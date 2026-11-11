"""Unit tests for helpdesk.prompts."""

from __future__ import annotations

from helpdesk.prompts import (
    build_answer_system_prompt,
    build_assistant_system_prompt,
    build_grounded_answer_messages,
    format_conversation_context,
    format_database_evidence,
    format_document_evidence,
)


def test_assistant_system_prompt_contains_core_guardrails():
    prompt = build_assistant_system_prompt(include_tools=True)
    lower = prompt.lower()

    assert "indian railways passenger help desk" in lower
    assert "newer" in lower and "disagree" in lower
    assert "2016" in lower
    assert "no ticket fares" in lower
    assert "i don't know" in lower
    assert "never invent" in lower
    assert "tool usage guidelines" in lower


def test_assistant_system_prompt_without_tools_and_extra_instructions():
    prompt = build_assistant_system_prompt(
        include_tools=False,
        extra_instructions="Keep answers under 3 sentences.",
    )
    assert "TOOL USAGE GUIDELINES" not in prompt
    assert "Keep answers under 3 sentences." in prompt


def test_answer_system_prompt_includes_json_schema_without_hardcoded_answers():
    prompt = build_answer_system_prompt(strict_json=True)
    assert '"answer"' in prompt
    assert '"sources"' in prompt
    assert '"dont_know"' in prompt
    # Ensure no sample question answers are hardcoded in prompts
    assert "G.S.R. 41(E)" not in prompt
    assert "12430" not in prompt
    assert "12965" not in prompt


def test_format_document_and_database_evidence_and_context():
    doc_block = format_document_evidence(
        [
            {"doc": "rb_cc10_2024_advance_reservation.pdf", "page": 1, "date": "2024-10-16", "text": "ARP is 60 days."},
            {"doc": "pib_2026-03-24_reform_express.html", "page": None, "date": "2026-03-24", "text": "PIB note."},
        ]
    )
    assert "doc=rb_cc10_2024_advance_reservation.pdf | page=1 | date=2024-10-16" in doc_block
    assert "doc=pib_2026-03-24_reform_express.html | page=null (HTML/no page)" in doc_block

    db_block = format_database_evidence(
        [{"number": "12002", "departure": "06:15"}],
        sql_queries="SELECT departure FROM schedules WHERE train_number = '12002';",
    )
    assert "SELECT departure FROM schedules" in db_block
    assert '"departure": "06:15"' in db_block

    ctx_block = format_conversation_context(
        [
            {"role": "user", "content": "Tell me about Tatkal."},
            {"role": "assistant", "content": "Please clarify which rule you want."},
        ]
    )
    assert "USER: Tell me about Tatkal." in ctx_block
    assert "ASSISTANT: Please clarify which rule you want." in ctx_block


def test_build_grounded_answer_messages_structure():
    messages = build_grounded_answer_messages(
        question="What is the advance reservation limit?",
        chunks=[{"doc": "rb_cc10_2024_advance_reservation.pdf", "page": 1, "text": "60 days."}],
        db_results=None,
        sql=None,
        context_messages=[{"role": "user", "content": "Previous question"}],
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "RECENT CONVERSATION CONTEXT" in messages[1]["content"]
    assert "DOCUMENT EVIDENCE" in messages[1]["content"]
    assert "PASSENGER QUESTION:" in messages[1]["content"]

