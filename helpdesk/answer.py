"""Grounded answer generation for the Railway Passenger Help Desk.

Synthesizes plain-language answers strictly from retrieved document chunks and
SQLite timetable results while deterministically enforcing source allowlists,
page numbers, SQL preservation, and safe ``dont_know`` behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from helpdesk.llm import LLMClient, LLMError, LLMInvalidResponseError, redact_secrets
from helpdesk.prompts import build_grounded_answer_messages

logger = logging.getLogger(__name__)

_HTML_EXTENSIONS = {".html", ".htm"}
_REFUSAL_PREFIX_RE = re.compile(
    r"^\s*(i\s+don['’]?t\s+know|i\s+do\s+not\s+know|i\s+can['’]?t|i\s+cannot|"
    r"sorry,\s+i\s+(?:don['’]?t|cannot|can['’]?t))\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceCitation:
    """Verified source document reference."""

    doc: str
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the official capstone source schema."""
        return {"doc": self.doc, "page": self.page}


@dataclass(frozen=True)
class GroundedAnswer:
    """Official four-field help desk answer payload."""

    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    sql: str | None = None
    dont_know: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the exact dict required by ask(question)."""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "sql": self.sql,
            "dont_know": self.dont_know,
        }


def _clean_doc_name(raw_doc: Any) -> str:
    text = str(raw_doc or "").strip()
    if not text:
        return ""
    # Normalize slashes across OSes while keeping just the canonical filename
    return Path(text.replace("\\", "/")).name


def _normalize_page(doc_name: str, raw_page: Any) -> int | None:
    if Path(doc_name).suffix.lower() in _HTML_EXTENSIONS:
        return None
    if raw_page is None or isinstance(raw_page, bool):
        return None
    if isinstance(raw_page, int):
        return raw_page if raw_page >= 1 else None
    text = str(raw_page).strip()
    if text.isdigit():
        num = int(text)
        return num if num >= 1 else None
    return None


def normalize_sql_value(sql: str | Sequence[str] | None) -> str | None:
    """Normalize executed SQL query input to a single string or None."""
    if sql is None:
        return None
    if isinstance(sql, str):
        stripped = sql.strip()
        return stripped or None
    seen: list[str] = []
    for item in sql:
        cleaned = str(item or "").strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return "\n".join(seen) if seen else None


def extract_allowed_sources(
    chunks: Sequence[Mapping[str, Any]] = (),
    source_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> list[SourceCitation]:
    """Build an ordered, deduplicated list of verified citations from supplied evidence."""
    citations: list[SourceCitation] = []
    seen: set[tuple[str, int | None]] = set()

    items: list[Mapping[str, Any]] = list(chunks)
    if source_metadata:
        items.extend(source_metadata)

    for item in items:
        if not isinstance(item, Mapping):
            continue
        raw_doc = (
            item.get("doc")
            or item.get("source")
            or item.get("filename")
            or item.get("document")
        )
        doc_name = _clean_doc_name(raw_doc)
        if not doc_name:
            continue
        raw_page = item.get("page") if "page" in item else item.get("page_number")
        page = _normalize_page(doc_name, raw_page)
        key = (doc_name, page)
        if key not in seen:
            seen.add(key)
            citations.append(SourceCitation(doc=doc_name, page=page))
    return citations


def _has_database_evidence(db_results: Any) -> bool:
    """Return True if database results represent actual query output (not None or error)."""
    if db_results is None:
        return False
    if isinstance(db_results, Mapping):
        if db_results.get("ok") is False:
            return False
        if "rows" in db_results:
            return db_results["rows"] is not None
        if "results" in db_results:
            return db_results["results"] is not None
        if "data" in db_results:
            return db_results["data"] is not None
        return bool(db_results)
    return True


def _validate_citations(
    raw_sources: Any,
    allowed_sources: Sequence[SourceCitation],
) -> list[SourceCitation]:
    """Keep only citations whose (doc, page) strictly exist in retrieved evidence."""
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        return []

    allowed_set = {(s.doc.lower(), s.page): s for s in allowed_sources}
    validated: list[SourceCitation] = []
    seen: set[tuple[str, int | None]] = set()

    for entry in raw_sources:
        if not isinstance(entry, Mapping):
            continue
        doc_name = _clean_doc_name(entry.get("doc") or entry.get("source") or entry.get("filename"))
        if not doc_name:
            continue
        page = _normalize_page(doc_name, entry.get("page"))
        match = allowed_set.get((doc_name.lower(), page))
        if match is None:
            continue
        dedup_key = (match.doc, match.page)
        if dedup_key not in seen:
            seen.add(dedup_key)
            validated.append(match)
    return validated


class AnswerGenerator:
    """Generates and validates grounded help desk responses using an LLMClient."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        raise_on_error: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.raise_on_error = raise_on_error

    def generate(
        self,
        *,
        question: str,
        chunks: Sequence[Mapping[str, Any]] = (),
        db_results: Any = None,
        source_metadata: Sequence[Mapping[str, Any]] | None = None,
        sql: str | Sequence[str] | None = None,
        conversation_context: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate a grounded, validated response dict matching ask(question)."""
        cleaned_question = (question or "").strip()
        validated_sql = normalize_sql_value(sql)
        allowed_sources = extract_allowed_sources(chunks, source_metadata)
        has_docs = bool(chunks)
        has_db = _has_database_evidence(db_results)

        if not cleaned_question:
            return GroundedAnswer(
                answer="I don't know. Please provide a question.",
                sources=[],
                sql=None,
                dont_know=True,
            ).to_dict()

        if not has_docs and not has_db:
            return GroundedAnswer(
                answer=(
                    "I don't know. I could not find sufficient evidence in the official "
                    "railway documents or timetable database to answer that."
                ),
                sources=[],
                sql=validated_sql,
                dont_know=True,
            ).to_dict()

        messages = build_grounded_answer_messages(
            question=cleaned_question,
            chunks=chunks,
            db_results=db_results,
            sql=validated_sql,
            context_messages=conversation_context,
        )

        try:
            raw_payload = self.llm_client.generate_json(messages)
            grounded = self.validate_llm_payload(
                raw_payload=raw_payload,
                allowed_sources=allowed_sources,
                validated_sql=validated_sql,
                has_docs=has_docs,
                has_db=has_db,
            )
            return grounded.to_dict()
        except (LLMInvalidResponseError, LLMError) as exc:
            logger.error(
                "Answer generation failed safely (%s): %s",
                type(exc).__name__,
                redact_secrets(str(exc), self.llm_client.config.known_secrets()),
            )
            if self.raise_on_error:
                raise
            return GroundedAnswer(
                answer=(
                    "I don't know. I encountered an issue verifying the response from "
                    "the available railway evidence."
                ),
                sources=[],
                sql=validated_sql,
                dont_know=True,
            ).to_dict()

    def validate_llm_payload(
        self,
        *,
        raw_payload: Mapping[str, Any],
        allowed_sources: Sequence[SourceCitation],
        validated_sql: str | None,
        has_docs: bool,
        has_db: bool,
    ) -> GroundedAnswer:
        """Validate and sanitize raw LLM JSON against deterministic evidence rules."""
        if not isinstance(raw_payload, Mapping):
            raise LLMInvalidResponseError("Answer payload must be a JSON object.")

        raw_answer = raw_payload.get("answer")
        if not isinstance(raw_answer, str) or not raw_answer.strip():
            raise LLMInvalidResponseError("Answer payload is missing a non-empty 'answer' string.")

        answer_text = raw_answer.strip()
        raw_dont_know = raw_payload.get("dont_know")
        if raw_dont_know is None:
            dont_know = bool(_REFUSAL_PREFIX_RE.match(answer_text))
        elif isinstance(raw_dont_know, bool):
            dont_know = raw_dont_know or bool(_REFUSAL_PREFIX_RE.match(answer_text))
        else:
            raise LLMInvalidResponseError("'dont_know' must be a boolean if provided.")

        if dont_know:
            return GroundedAnswer(
                answer=answer_text,
                sources=[],
                sql=validated_sql,
                dont_know=True,
            )

        validated_sources = _validate_citations(raw_payload.get("sources"), allowed_sources)
        # Preserve document citations if doc chunks were the primary evidence and LLM omitted them.
        if not validated_sources and has_docs and not has_db and allowed_sources:
            validated_sources = list(allowed_sources)

        return GroundedAnswer(
            answer=answer_text,
            sources=validated_sources,
            sql=validated_sql,
            dont_know=False,
        )
