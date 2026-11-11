"""Unit tests for helpdesk.answer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from helpdesk.answer import AnswerGenerator, extract_allowed_sources
from helpdesk.llm import LLMInvalidResponseError


def _mock_llm_returning_json(payload):
    llm = MagicMock()
    llm.config = MagicMock()
    llm.config.known_secrets.return_value = ()
    if isinstance(payload, Exception):
        llm.generate_json.side_effect = payload
    else:
        llm.generate_json.return_value = payload
    return llm


def test_answer_structure_and_source_page_preservation():
    llm = _mock_llm_returning_json(
        {
            "answer": "Advance reservation period is 60 days.",
            "sources": [
                {"doc": "rb_cc10_2024_advance_reservation.pdf", "page": 1},
                {"doc": "pib_2026-03-24_reform_express.html", "page": 99},  # HTML page forced to None if matching null
            ],
            "dont_know": False,
        }
    )
    gen = AnswerGenerator(llm)

    chunks = [
        {"doc": "rb_cc10_2024_advance_reservation.pdf", "page": 1, "text": "60 days ARP."},
        {"doc": "pib_2026-03-24_reform_express.html", "page": None, "text": "Reform Express."},
    ]
    result = gen.generate(question="How early can I book?", chunks=chunks)

    assert set(result.keys()) == {"answer", "sources", "sql", "dont_know"}
    assert result["answer"] == "Advance reservation period is 60 days."
    assert result["sql"] is None
    assert result["dont_know"] is False
    assert {"doc": "rb_cc10_2024_advance_reservation.pdf", "page": 1} in result["sources"]
    assert {"doc": "pib_2026-03-24_reform_express.html", "page": None} in result["sources"]


def test_rejects_invented_sources_pages_and_sql():
    llm = _mock_llm_returning_json(
        {
            "answer": "The refund charge is Rs 60.",
            "sources": [
                {"doc": "irctc_refund_rules_jan2026.pdf", "page": 3},
                {"doc": "irctc_refund_rules_jan2026.pdf", "page": 999},  # hallucinated page
                {"doc": "fake_circular_2029.pdf", "page": 1},  # hallucinated doc
            ],
            "sql": "DROP TABLE trains;",  # hallucinated/malicious SQL from LLM must be ignored
            "dont_know": False,
        }
    )
    gen = AnswerGenerator(llm)

    result = gen.generate(
        question="What is the clerkage charge?",
        chunks=[{"doc": "irctc_refund_rules_jan2026.pdf", "page": 3, "text": "Clerkage Rs 60."}],
        sql="SELECT * FROM trains WHERE number = '12951';",
        db_results=[{"number": "12951"}],
    )

    assert result["sources"] == [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 3}]
    assert result["sql"] == "SELECT * FROM trains WHERE number = '12951';"
    assert result["dont_know"] is False


def test_missing_evidence_returns_dont_know_without_calling_llm():
    llm = _mock_llm_returning_json({"answer": "Should not run", "sources": [], "dont_know": False})
    gen = AnswerGenerator(llm)

    result = gen.generate(question="How much is a 3AC ticket?", chunks=[], db_results=None)

    assert result["dont_know"] is True
    assert result["sources"] == []
    assert result["sql"] is None
    assert "i don't know" in result["answer"].lower()
    llm.generate_json.assert_not_called()


def test_invalid_llm_response_falls_back_safely_or_raises_if_configured():
    llm_bad = _mock_llm_returning_json(LLMInvalidResponseError("Malformed JSON"))
    gen_safe = AnswerGenerator(llm_bad, raise_on_error=False)
    safe_res = gen_safe.generate(
        question="What is the refund rule?",
        chunks=[{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1, "text": "Refund text."}],
    )
    assert safe_res["dont_know"] is True
    assert safe_res["sources"] == []

    gen_strict = AnswerGenerator(llm_bad, raise_on_error=True)
    with pytest.raises(LLMInvalidResponseError):
        gen_strict.generate(
            question="What is the refund rule?",
            chunks=[{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1, "text": "Refund text."}],
        )


def test_refusal_answer_forces_dont_know_and_clears_sources():
    llm = _mock_llm_returning_json(
        {
            "answer": "I don't know whether your waitlisted ticket will confirm.",
            "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 3}],
            "dont_know": False,  # LLM forgot to flip boolean
        }
    )
    gen = AnswerGenerator(llm)
    res = gen.generate(
        question="Will my WL ticket confirm?",
        chunks=[{"doc": "irctc_refund_rules_jan2026.pdf", "page": 3, "text": "Waitlist cancellation."}],
    )
    assert res["dont_know"] is True
    assert res["sources"] == []


def test_extract_allowed_sources_deduplicates_and_handles_paths():
    allowed = extract_allowed_sources(
        chunks=[
            {"doc": "documents/rb_cc08_2025_tatkal.pdf", "page": "2"},
            {"source": "rb_cc08_2025_tatkal.pdf", "page": 2},
            {"filename": "indianrail_refund_rules_2015.html", "page": 1},
        ]
    )
    assert [(c.doc, c.page) for c in allowed] == [
        ("rb_cc08_2025_tatkal.pdf", 2),
        ("indianrail_refund_rules_2015.html", None),
    ]
