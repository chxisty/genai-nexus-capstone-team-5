"""Main AI Assistant coordinator owned by Member 1 (Christy Marrya George).

Orchestrates multi-turn conversation context (via MemoryAdapter / Prasad),
LLM reasoning and multi-step tool calling (via LLMClient), tool execution
delegation (to Neha's tool executor / Krishna's search & SQL tools), and
grounded answer synthesis (via AnswerGenerator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from helpdesk.answer import (
    AnswerGenerator,
    GroundedAnswer,
    _REFUSAL_PREFIX_RE,
    extract_allowed_sources,
    normalize_sql_value,
)
from helpdesk.llm import (
    LLMClient,
    LLMError,
    LLMInvalidResponseError,
    LLMResponse,
    extract_json_object,
    redact_secrets,
)
from helpdesk.memory_adapter import MemoryAdapter
from helpdesk.prompts import build_assistant_system_prompt

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolExecutorProtocol(Protocol):
    """Expected integration contract for Member 3 (Neha)'s tool executor."""

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a named tool with validated arguments and return structured data."""


@dataclass
class _EvidenceAccumulator:
    """Tracks all verified document chunks and SQL results collected during a turn."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    source_metadata: list[dict[str, Any]] = field(default_factory=list)
    db_results: list[dict[str, Any]] = field(default_factory=list)
    sql_queries: list[str] = field(default_factory=list)
    had_db_query: bool = False
    tool_errors: list[str] = field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks) or self.had_db_query or bool(self.db_results)

    def formatted_db_results(self) -> list[dict[str, Any]] | None:
        if not self.had_db_query and not self.db_results:
            return None
        return self.db_results


def _looks_like_doc_chunk(item: Mapping[str, Any]) -> bool:
    doc_keys = {"doc", "source", "filename", "document"}
    text_keys = {"text", "content", "chunk", "body"}
    return bool(doc_keys & set(item.keys())) or bool(text_keys & set(item.keys()))


def _collect_evidence_from_tool_result(
    tool_name: str,
    raw_result: Any,
    acc: _EvidenceAccumulator,
) -> dict[str, Any]:
    """Normalize a tool output payload and accumulate verified evidence."""
    parsed: Any = raw_result
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except json.JSONDecodeError:
            parsed = {"ok": True, "text": raw_result}

    if isinstance(parsed, Mapping):
        if parsed.get("ok") is False or (parsed.get("error") and not parsed.get("chunks") and not parsed.get("rows")):
            acc.tool_errors.append(str(parsed.get("error") or f"Tool {tool_name} failed"))
            return dict(parsed)

        sql_val = parsed.get("sql") or parsed.get("query")
        if isinstance(sql_val, str) and sql_val.strip():
            cleaned_sql = sql_val.strip()
            if cleaned_sql not in acc.sql_queries:
                acc.sql_queries.append(cleaned_sql)
            acc.had_db_query = True

        # Extract document chunks from standard keys
        chunk_candidates = (
            parsed.get("chunks")
            or parsed.get("documents")
            or parsed.get("matches")
        )
        if isinstance(chunk_candidates, Sequence) and not isinstance(chunk_candidates, (str, bytes)):
            for item in chunk_candidates:
                if isinstance(item, Mapping):
                    acc.chunks.append(dict(item))
                    acc.source_metadata.append(dict(item))

        # Handle ambiguous "results" key depending on whether it's SQL rows or doc chunks
        results_val = parsed.get("results")
        if isinstance(results_val, Sequence) and not isinstance(results_val, (str, bytes)):
            if results_val and all(isinstance(r, Mapping) and _looks_like_doc_chunk(r) for r in results_val) and not sql_val:
                for item in results_val:
                    acc.chunks.append(dict(item))
                    acc.source_metadata.append(dict(item))
            else:
                acc.had_db_query = True
                acc.db_results.append(
                    {
                        "tool": tool_name,
                        "sql": sql_val,
                        "columns": parsed.get("columns"),
                        "rows": list(results_val),
                    }
                )

        # Extract explicit SQL row fields
        for row_key in ("rows", "data"):
            if row_key in parsed and parsed[row_key] is not None:
                acc.had_db_query = True
                acc.db_results.append(
                    {
                        "tool": tool_name,
                        "sql": sql_val,
                        "columns": parsed.get("columns"),
                        "rows": parsed[row_key],
                    }
                )
                break

        # Also check explicit standalone text chunk
        if "text" in parsed and _looks_like_doc_chunk(parsed) and not chunk_candidates:
            acc.chunks.append(dict(parsed))
            acc.source_metadata.append(dict(parsed))

        return dict(parsed)

    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
        items = list(parsed)
        if items and all(isinstance(i, Mapping) and _looks_like_doc_chunk(i) for i in items):
            for item in items:
                acc.chunks.append(dict(item))
                acc.source_metadata.append(dict(item))
        elif items:
            acc.had_db_query = True
            acc.db_results.append({"tool": tool_name, "rows": items})
        return {"ok": True, "results": items}

    return {"ok": True, "result": parsed}


class AIAssistant:
    """AI assistant orchestration layer connecting LLM, tools, memory, and grounding."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        tool_executor: Any = None,
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
        memory_adapter: MemoryAdapter | None = None,
        answer_generator: AnswerGenerator | None = None,
        max_tool_steps: int = 5,
        extra_system_instructions: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.tool_schemas = list(tool_schemas) if tool_schemas else []
        self.memory_adapter = memory_adapter or MemoryAdapter()
        self.answer_generator = answer_generator or AnswerGenerator(llm_client)
        self.max_tool_steps = max(1, max_tool_steps)
        self.extra_system_instructions = extra_system_instructions

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Alias for process_question matching the capstone ask(question) signature."""
        return self.process_question(question, session_id=session_id)

    def process_question(
        self,
        question: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a passenger question end-to-end and return the 4-key response dict."""
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            return GroundedAnswer(
                answer="I don't know. Please ask a railway question.",
                sources=[],
                sql=None,
                dont_know=True,
            ).to_dict()

        sid = self.memory_adapter.ensure_session(session_id)
        context_messages = self.memory_adapter.get_context_messages(sid)
        self.memory_adapter.record_user_message(sid, cleaned_question)

        use_tools = bool(self.tool_schemas and self.tool_executor is not None)
        system_prompt = build_assistant_system_prompt(
            include_tools=use_tools,
            extra_instructions=self.extra_system_instructions,
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(context_messages)
        messages.append({"role": "user", "content": cleaned_question})

        evidence = _EvidenceAccumulator()
        final_response: LLMResponse | None = None

        try:
            for step in range(self.max_tool_steps):
                llm_resp = self.llm_client.chat(
                    messages,
                    tools=self.tool_schemas if use_tools else None,
                )
                if llm_resp.has_tool_calls:
                    if not use_tools:
                        logger.warning(
                            "LLM requested tool calls at step %d without tool executor/schemas.",
                            step + 1,
                        )
                        break

                    serialized_calls = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.raw_arguments
                                or json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in llm_resp.tool_calls
                    ]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": llm_resp.content,
                            "tool_calls": serialized_calls,
                        }
                    )
                    self.memory_adapter.record_tool_calls(
                        sid, serialized_calls, content=llm_resp.content
                    )

                    for tc in llm_resp.tool_calls:
                        raw_tool_out = self._invoke_tool(tc.name, tc.arguments)
                        normalized_out = _collect_evidence_from_tool_result(
                            tc.name, raw_tool_out, evidence
                        )
                        serialized_out = json.dumps(
                            normalized_out, ensure_ascii=False, default=str
                        )
                        self.memory_adapter.record_tool_result(
                            sid,
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            result=normalized_out,
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tc.name,
                                "content": serialized_out,
                            }
                        )
                    continue

                final_response = llm_resp
                break

            result = self._finalize_response(
                question=cleaned_question,
                evidence=evidence,
                final_llm_response=final_response,
                context_messages=context_messages,
            )
        except LLMError as exc:
            logger.error(
                "AI assistant LLM failure (%s): %s",
                type(exc).__name__,
                redact_secrets(str(exc), self.llm_client.config.known_secrets()),
            )
            result = GroundedAnswer(
                answer=(
                    "I don't know. I encountered a temporary problem contacting the AI "
                    "service. Please try again."
                ),
                sources=[],
                sql=normalize_sql_value(evidence.sql_queries),
                dont_know=True,
            ).to_dict()

        self.memory_adapter.record_assistant_message(
            sid,
            result["answer"],
            metadata={
                "sources": result["sources"],
                "sql": result["sql"],
                "dont_know": result["dont_know"],
            },
        )
        return result

    def answer_with_evidence(
        self,
        *,
        question: str,
        chunks: Sequence[Mapping[str, Any]] = (),
        db_results: Any = None,
        source_metadata: Sequence[Mapping[str, Any]] | None = None,
        sql: str | Sequence[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate a grounded answer directly from pre-retrieved evidence and record memory."""
        sid = self.memory_adapter.ensure_session(session_id)
        context_messages = self.memory_adapter.get_context_messages(sid)
        self.memory_adapter.record_user_message(sid, question)

        result = self.answer_generator.generate(
            question=question,
            chunks=chunks,
            db_results=db_results,
            source_metadata=source_metadata,
            sql=sql,
            conversation_context=context_messages,
        )
        self.memory_adapter.record_assistant_message(
            sid,
            result["answer"],
            metadata={
                "sources": result["sources"],
                "sql": result["sql"],
                "dont_know": result["dont_know"],
            },
        )
        return result

    def _invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Safely dispatch a tool call to Neha's executor without propagating crashes."""
        if self.tool_executor is None:
            return {"ok": False, "error": "No tool executor configured."}

        try:
            if callable(self.tool_executor):
                return self.tool_executor(tool_name, arguments)
            for method_name in ("execute", "execute_tool", "run_tool", "call"):
                method = getattr(self.tool_executor, method_name, None)
                if callable(method):
                    return method(tool_name, arguments)
            return {
                "ok": False,
                "error": "Tool executor does not expose execute() or callable interface.",
            }
        except Exception as exc:
            safe_msg = redact_secrets(str(exc), self.llm_client.config.known_secrets())
            logger.warning("Tool '%s' raised %s: %s", tool_name, type(exc).__name__, safe_msg)
            return {"ok": False, "error": f"Tool '{tool_name}' failed: {safe_msg}"}

    def _finalize_response(
        self,
        *,
        question: str,
        evidence: _EvidenceAccumulator,
        final_llm_response: LLMResponse | None,
        context_messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        validated_sql = normalize_sql_value(evidence.sql_queries)

        if not evidence.has_evidence:
            # If the LLM explicitly issued a refusal/out-of-scope decline (e.g., write SQL refusal),
            # preserve its polite refusal message with dont_know=True. Otherwise prevent ungrounded claims.
            if final_llm_response is not None and final_llm_response.content:
                text = final_llm_response.content.strip()
                # Check if LLM already emitted a refusal JSON payload
                try:
                    parsed = extract_json_object(text)
                    ans_text = str(parsed.get("answer", "")).strip()
                    if ans_text and (parsed.get("dont_know") is True or _REFUSAL_PREFIX_RE.match(ans_text)):
                        return GroundedAnswer(
                            answer=ans_text,
                            sources=[],
                            sql=None,
                            dont_know=True,
                        ).to_dict()
                except LLMInvalidResponseError:
                    if _REFUSAL_PREFIX_RE.match(text):
                        return GroundedAnswer(
                            answer=text,
                            sources=[],
                            sql=None,
                            dont_know=True,
                        ).to_dict()

            if evidence.tool_errors:
                return GroundedAnswer(
                    answer=(
                        "I don't know. I could not retrieve verified information because "
                        "the lookup returned no usable results."
                    ),
                    sources=[],
                    sql=None,
                    dont_know=True,
                ).to_dict()

            return GroundedAnswer(
                answer=(
                    "I don't know. I could not find supporting evidence in the official "
                    "railway documents or timetable database to answer that."
                ),
                sources=[],
                sql=None,
                dont_know=True,
            ).to_dict()

        # Evidence exists: if final_llm_response already returned valid structured JSON,
        # validate it directly without making a redundant synthesis request.
        if final_llm_response is not None and final_llm_response.content:
            allowed_sources = extract_allowed_sources(
                evidence.chunks, evidence.source_metadata
            )
            try:
                parsed = extract_json_object(final_llm_response.content)
                validated = self.answer_generator.validate_llm_payload(
                    raw_payload=parsed,
                    allowed_sources=allowed_sources,
                    validated_sql=validated_sql,
                    has_docs=bool(evidence.chunks),
                    has_db=evidence.had_db_query or bool(evidence.db_results),
                )
                return validated.to_dict()
            except LLMInvalidResponseError:
                pass

        return self.answer_generator.generate(
            question=question,
            chunks=evidence.chunks,
            db_results=evidence.formatted_db_results(),
            source_metadata=evidence.source_metadata,
            sql=validated_sql,
            conversation_context=context_messages,
        )
