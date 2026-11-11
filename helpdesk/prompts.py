"""Modular, reusable prompt components for the Railway Passenger Help Desk.

All prompts enforce strict evidence grounding, document-date and train-specific
rule precedence, explicit distinction between the 2016 timetable snapshot and
live railway operations, and zero hallucination of facts, sources, or SQL.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Reusable Prompt Building Blocks
# ---------------------------------------------------------------------------

ROLE_IDENTITY_PROMPT = (
    "You are the Indian Railways Passenger Help Desk Assistant. "
    "Your job is to answer passenger questions accurately and concisely in plain "
    "language using ONLY the official railway data pack provided to you."
)

EVIDENCE_GROUNDING_PROMPT = (
    "STRICT EVIDENCE GROUNDING RULES:\n"
    "1. Base every factual claim exclusively on the retrieved document excerpts "
    "and database query results provided in this conversation.\n"
    "2. Never rely on outside knowledge, assumptions, or unverified memory for "
    "railway rules, refund amounts, train numbers, station codes, timings, or routes.\n"
    "3. Never invent train numbers, station names, timings, circular numbers, "
    "document filenames, page numbers, or SQL queries."
)

RULE_PRECEDENCE_PROMPT = (
    "DOCUMENT DATES AND RULE PRECEDENCE:\n"
    "1. Check the issue/effective date of every document excerpt.\n"
    "2. When two documents disagree because rules changed over time, the newer "
    "applicable Railway Board circular, Gazette notification, or PIB release wins.\n"
    "3. Train-specific rules (for example, circulars covering a specific train "
    "service) override general passenger refund/cancellation rules for that train."
)

TIMETABLE_SCOPE_PROMPT = (
    "TIMETABLE SNAPSHOT VS. LIVE DATA:\n"
    "1. The timetable database is a static 2016 snapshot of Indian Railways schedules "
    "(covering stations, train_types, trains, schedules, and the stops view).\n"
    "2. Treat schedule answers as reflecting the timetable snapshot in the data pack, "
    "NOT today's live train running status.\n"
    "3. The data pack contains NO ticket fares, NO live train delays/locations, "
    "NO PNR status, NO seat availability, and NO waitlist confirmation predictions."
)

MISSING_AND_CONFLICT_PROMPT = (
    "MISSING, AMBIGUOUS, OR UNSUPPORTED REQUESTS:\n"
    "1. If the supplied documents and database results do not contain enough evidence "
    "to answer reliably, clearly say \"I don't know\" and explain what information is "
    "missing from the pack (set dont_know=true).\n"
    "2. If a user asks for out-of-scope data (fares, PNR, live running status, "
    "waitlist confirmation chances) or asks to modify/insert/delete database records, "
    "politely decline and set dont_know=true.\n"
    "3. If a query is ambiguous (e.g., multiple stations match a city name or multiple "
    "trains match), list the matching options returned by the evidence or ask a brief "
    "clarifying question instead of guessing."
)

CITATION_INTEGRITY_PROMPT = (
    "SOURCE CITATION RULES:\n"
    "1. Only cite document filenames (`doc`) and page numbers (`page`) that appear "
    "explicitly in the retrieved evidence metadata.\n"
    "2. Preserve exact filenames (e.g., `example.pdf` or `example.html`).\n"
    "3. For PDF files, preserve the exact integer page number; for HTML web pages "
    "(where page is null/None), use null for `page`."
)

TOOL_USAGE_PROMPT = (
    "TOOL USAGE GUIDELINES:\n"
    "1. Use document search tools for railway rules, refund slabs, cancellation windows, "
    "Tatkal requirements, advance reservation limits, and reservation chart rules.\n"
    "2. Use timetable database tools for train numbers, train names, station codes, "
    "routes, intermediate stops, departure/arrival times, durations, and counts.\n"
    "3. For multi-part questions involving both timetable details and passenger rules, "
    "call BOTH the database tool and the document search tool.\n"
    "4. Always verify exact codes, train numbers, and rule clauses via tools before "
    "answering. Never attempt write/mutating SQL operations."
)

STRUCTURED_ANSWER_JSON_PROMPT = (
    "OUTPUT FORMAT REQUIREMENT:\n"
    "Respond strictly as a single valid JSON object with these keys:\n"
    "{\n"
    '  "answer": "Clear, plain-language reply grounded in evidence (or I don\'t know ...)",\n'
    '  "sources": [{"doc": "exact_filename", "page": 1}],\n'
    '  "dont_know": false\n'
    "}\n"
    "Rules for JSON fields:\n"
    "- `answer`: plain text only, accurate and concise.\n"
    "- `sources`: list of cited source objects copied strictly from provided evidence "
    "(empty list [] when only database evidence is used or when declining).\n"
    "- `dont_know`: boolean; true whenever evidence is insufficient, the question is "
    "out of scope, or the request is unsafe/read-only-violating."
)


def build_assistant_system_prompt(
    *,
    include_tools: bool = True,
    extra_instructions: str | None = None,
) -> str:
    """Compose the system prompt for the tool-using AI assistant loop."""
    sections = [
        ROLE_IDENTITY_PROMPT,
        EVIDENCE_GROUNDING_PROMPT,
        RULE_PRECEDENCE_PROMPT,
        TIMETABLE_SCOPE_PROMPT,
        MISSING_AND_CONFLICT_PROMPT,
        CITATION_INTEGRITY_PROMPT,
    ]
    if include_tools:
        sections.append(TOOL_USAGE_PROMPT)
    if extra_instructions and extra_instructions.strip():
        sections.append(extra_instructions.strip())
    return "\n\n".join(sections)


def build_answer_system_prompt(*, strict_json: bool = True) -> str:
    """Compose the system prompt for grounded final answer synthesis."""
    sections = [
        ROLE_IDENTITY_PROMPT,
        EVIDENCE_GROUNDING_PROMPT,
        RULE_PRECEDENCE_PROMPT,
        TIMETABLE_SCOPE_PROMPT,
        MISSING_AND_CONFLICT_PROMPT,
        CITATION_INTEGRITY_PROMPT,
    ]
    if strict_json:
        sections.append(STRUCTURED_ANSWER_JSON_PROMPT)
    return "\n\n".join(sections)


def _chunk_field(chunk: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in chunk and chunk[key] is not None:
            return chunk[key]
    return default


def format_document_evidence(chunks: Sequence[Mapping[str, Any]]) -> str:
    """Format retrieved document chunks with explicit doc, page, and date metadata."""
    if not chunks:
        return "DOCUMENT EVIDENCE: None provided."

    lines = ["DOCUMENT EVIDENCE (use exact doc and page values when citing):"]
    for idx, chunk in enumerate(chunks, start=1):
        doc = str(_chunk_field(chunk, "doc", "source", "filename", "document", default="unknown"))
        page = _chunk_field(chunk, "page", "page_number", default=None)
        date = _chunk_field(chunk, "date", "doc_date", "issued_date", default="unknown")
        text = str(_chunk_field(chunk, "text", "content", "chunk", "body", default="")).strip()
        page_label = "null (HTML/no page)" if page is None else str(page)
        lines.append(
            f"[DocChunk {idx}] doc={doc} | page={page_label} | date={date}\n{text}"
        )
    return "\n\n".join(lines)


def format_database_evidence(
    db_results: Any,
    sql_queries: Sequence[str] | str | None = None,
) -> str:
    """Format SQLite query results and executed SQL safely for grounded synthesis."""
    if isinstance(sql_queries, str):
        queries = [sql_queries] if sql_queries.strip() else []
    elif sql_queries:
        queries = [str(q).strip() for q in sql_queries if str(q).strip()]
    else:
        queries = []

    has_results = db_results is not None and db_results != [] and db_results != {}
    if not has_results and not queries:
        return "DATABASE EVIDENCE: None provided."

    lines = ["DATABASE EVIDENCE (2016 SQLite timetable snapshot):"]
    if queries:
        lines.append("Executed SQL:")
        for q in queries:
            lines.append(f"  {q}")
    if has_results:
        try:
            serialized = json.dumps(db_results, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            serialized = str(db_results)
        lines.append(f"Returned rows/data:\n{serialized}")
    else:
        lines.append("Returned rows/data: [] (0 rows)")
    return "\n".join(lines)


def format_conversation_context(
    context_messages: Sequence[Mapping[str, Any]] | None,
) -> str:
    """Format prior conversation turns clearly so follow-up questions resolve cleanly."""
    if not context_messages:
        return ""
    lines = ["RECENT CONVERSATION CONTEXT:"]
    for msg in context_messages:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if len(lines) > 1 else ""


def build_grounded_answer_messages(
    *,
    question: str,
    chunks: Sequence[Mapping[str, Any]] = (),
    db_results: Any = None,
    sql: Sequence[str] | str | None = None,
    context_messages: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build the complete message list for AnswerGenerator LLM synthesis."""
    system_prompt = build_answer_system_prompt(strict_json=True)
    parts: list[str] = []

    context_block = format_conversation_context(context_messages)
    if context_block:
        parts.append(context_block)

    parts.append(format_document_evidence(chunks))
    parts.append(format_database_evidence(db_results, sql))
    parts.append(f"PASSENGER QUESTION:\n{question.strip()}")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n---\n\n".join(parts)},
    ]

