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
    LLMAuthError,
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
    """Expected integration contract for tool executors exposing execute()."""

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a named tool with validated arguments and return structured data."""


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Integration contract matching Member 3 (Neha)'s tools.registry.ToolRegistry."""

    def call(self, tool_name: str, arguments: Any) -> dict[str, Any]:
        """Validate arguments and dispatch tool_name, returning status/result envelope."""


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


def _looks_like_sql(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip().upper()
    return stripped.startswith("SELECT ") or stripped.startswith("WITH ")


def _extract_doc_chunks(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return []
    return [dict(item) for item in items if isinstance(item, Mapping) and _looks_like_doc_chunk(item)]


def _collect_evidence_from_tool_result(
    tool_name: str,
    raw_result: Any,
    acc: _EvidenceAccumulator,
) -> dict[str, Any]:
    """Normalize a tool output payload (including ToolRegistry envelopes) and collect evidence."""
    parsed: Any = raw_result
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except json.JSONDecodeError:
            parsed = {"ok": True, "text": raw_result}

    if isinstance(parsed, Mapping):
        status = str(parsed.get("status", "")).lower()
        is_error = (
            parsed.get("ok") is False
            or status == "error"
            or (
                bool(parsed.get("error"))
                and not parsed.get("chunks")
                and not parsed.get("rows")
                and not parsed.get("result")
            )
        )
        if is_error:
            acc.tool_errors.append(str(parsed.get("error") or f"Tool {tool_name} failed"))
            return dict(parsed)

        # Unwrap Member 3 (Neha)'s ToolRegistry envelope: {"status": "success", "tool": ..., "result": {...}}
        if status == "success" and "result" in parsed:
            inner = parsed["result"]
            if isinstance(inner, Mapping):
                _collect_mapping_payload(tool_name, inner, acc)
            elif isinstance(inner, Sequence) and not isinstance(inner, (str, bytes)):
                _collect_sequence_payload(tool_name, inner, acc)
            elif inner is not None:
                acc.had_db_query = True
                acc.db_results.append({"tool": tool_name, "result": inner})
            return dict(parsed)

        _collect_mapping_payload(tool_name, parsed, acc)
        return dict(parsed)

    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
        items = list(parsed)
        _collect_sequence_payload(tool_name, items, acc)
        return {"ok": True, "results": items}

    return {"ok": True, "result": parsed}


def _collect_sequence_payload(
    tool_name: str,
    items: Sequence[Any],
    acc: _EvidenceAccumulator,
) -> None:
    doc_chunks = _extract_doc_chunks(items)
    if doc_chunks and len(doc_chunks) == len(items):
        acc.chunks.extend(doc_chunks)
        acc.source_metadata.extend(doc_chunks)
    elif items:
        acc.had_db_query = True
        acc.db_results.append({"tool": tool_name, "rows": list(items)})


def _collect_mapping_payload(
    tool_name: str,
    payload: Mapping[str, Any],
    acc: _EvidenceAccumulator,
) -> None:
    if payload.get("error") and not any(
        k in payload for k in ("chunks", "documents", "matches", "results", "rows", "data", "trains", "stops")
    ):
        acc.tool_errors.append(str(payload["error"]))
        return

    sql_val = payload.get("sql")
    if not _looks_like_sql(sql_val) and _looks_like_sql(payload.get("query")):
        sql_val = payload.get("query")
    if _looks_like_sql(sql_val):
        cleaned_sql = str(sql_val).strip()
        if cleaned_sql not in acc.sql_queries:
            acc.sql_queries.append(cleaned_sql)
        acc.had_db_query = True

    found_docs = False
    for doc_key in ("chunks", "documents"):
        val = payload.get(doc_key)
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            for item in val:
                if isinstance(item, Mapping):
                    acc.chunks.append(dict(item))
                    acc.source_metadata.append(dict(item))
                    found_docs = True

    # Check "matches" / "results" carefully: could be doc chunks OR structured station/DB rows
    for list_key in ("matches", "results"):
        val = payload.get(list_key)
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            doc_items = _extract_doc_chunks(val)
            if doc_items and len(doc_items) == len(val):
                acc.chunks.extend(doc_items)
                acc.source_metadata.extend(doc_items)
                found_docs = True
            else:
                acc.had_db_query = True
                acc.db_results.append(
                    {
                        "tool": tool_name,
                        "sql": cleaned_sql if _looks_like_sql(sql_val) else None,
                        "columns": payload.get("columns"),
                        "rows": list(val),
                    }
                )

    for row_key in ("rows", "data", "trains", "stops"):
        if row_key in payload and payload[row_key] is not None:
            acc.had_db_query = True
            acc.db_results.append(
                {
                    "tool": tool_name,
                    "sql": cleaned_sql if _looks_like_sql(sql_val) else None,
                    "columns": payload.get("columns"),
                    "rows": payload[row_key],
                }
            )
            return

    if "text" in payload and _looks_like_doc_chunk(payload) and not found_docs:
        acc.chunks.append(dict(payload))
        acc.source_metadata.append(dict(payload))
        return

    # Catch structured dict responses like train_details({"train_number": ..., "name": ...})
    if not found_docs and "matches" not in payload and "results" not in payload:
        informative_keys = set(payload.keys()) - {"ok", "status", "tool", "_note"}
        if informative_keys:
            acc.had_db_query = True
            acc.db_results.append({"tool": tool_name, "data": dict(payload)})


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
            config = getattr(self.llm_client, "config", None)
            known_secrets = config.known_secrets() if (config and hasattr(config, "known_secrets")) else ()
            if isinstance(exc, LLMAuthError):
                logger.debug(
                    "AI assistant LLM failure (%s): %s",
                    type(exc).__name__,
                    redact_secrets(str(exc), known_secrets),
                )
            else:
                logger.warning(
                    "AI assistant LLM failure (%s): %s",
                    type(exc).__name__,
                    redact_secrets(str(exc), known_secrets),
                )
            fallback = self._try_direct_fallback(cleaned_question)
            if fallback:
                result = fallback
            else:
                auth_hint = ""
                if isinstance(exc, LLMAuthError):
                    auth_hint = " (GEMINI_API_KEY is missing from .env)"
                result = GroundedAnswer(
                    answer=(
                        f"I don't know. I encountered a temporary problem contacting the AI "
                        f"service{auth_hint}. Please verify your .env configuration."
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

    def _try_direct_fallback(self, question: str) -> dict[str, Any] | None:
        """Heuristic fallback executing real database tools when LLM is unavailable."""
        if not self.tool_executor:
            return None

        import re
        q = question.strip()
        norm = re.sub(r"\s+", " ", q).lower()

        # 1. Train search between two stations
        route_match = re.search(r"\bfrom\s+([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+?)(?:\?|$|\.|\s+on|\s+date)", q, re.IGNORECASE)
        if not route_match:
            route_match = re.search(r"\b([A-Za-z]{2,5})\s+to\s+([A-Za-z]{2,5})\b", q, re.IGNORECASE)
        if route_match:
            src = route_match.group(1).strip()
            dst = route_match.group(2).strip()
            src = re.sub(r"(?i)\b(trains?|available|any)\b", "", src).strip()
            if src and dst:
                res = self._invoke_tool("train_search", {"source": src, "destination": dst, "date": "2026-11-15"})
                parsed = res.get("result") if isinstance(res, dict) and "result" in res else res
                if isinstance(parsed, dict) and parsed.get("trains"):
                    train_list = [f"- {t.get('train_number', t.get('number'))} ({t.get('name')})" for t in parsed["trains"][:10]]
                    ans = (
                        f"Based on the timetable database, found {parsed.get('count', len(train_list))} train(s) "
                        f"between {src.upper()} and {dst.upper()}:\n" + "\n".join(train_list)
                    )
                    return GroundedAnswer(answer=ans, sources=[], sql=None, dont_know=False).to_dict()

        # 2. Train route or details for 5-digit train number
        train_num_match = re.search(r"\b(\d{5})\b", q)
        if train_num_match:
            tnum = train_num_match.group(1)
            if any(w in norm for w in ("route", "stops", "halts", "stations")):
                res = self._invoke_tool("route_lookup", {"train_number": tnum})
                parsed = res.get("result") if isinstance(res, dict) and "result" in res else res
                if isinstance(parsed, dict) and parsed.get("stops"):
                    stops = [s.get("station_name", s.get("station_code")) for s in parsed["stops"] if s.get("is_halt", 1)]
                    major_stops = ", ".join(stops[:15]) + ("..." if len(stops) > 15 else "")
                    ans = f"The route for train {tnum} ({parsed.get('train_name', '')}) includes stops at: {major_stops}."
                    return GroundedAnswer(answer=ans, sources=[], sql=None, dont_know=False).to_dict()
            else:
                res = self._invoke_tool("train_details", {"train_number": tnum})
                parsed = res.get("result") if isinstance(res, dict) and "result" in res else res
                if isinstance(parsed, dict) and "name" in parsed:
                    cls_str = ", ".join(parsed.get("classes", [])) or "None specified"
                    ans = f"Train {tnum} is {parsed.get('name')} (Type: {parsed.get('train_type', 'Express')}). Available classes: {cls_str}."
                    return GroundedAnswer(answer=ans, sources=[], sql=None, dont_know=False).to_dict()

        # 3. Station search
        stn_match = re.search(r"(?:station\s+(?:code\s+)?|search\s+for\s+station\s+|search\s+stations?\s+)(?:in\s+|matching\s+)?['\"]?([A-Za-z\s]+?)['\"]?(?:\?|$|\.|\s+in)", q, re.IGNORECASE)
        if stn_match:
            stn_query = stn_match.group(1).strip()
            if stn_query:
                res = self._invoke_tool("station_search", {"query": stn_query})
                parsed = res.get("result") if isinstance(res, dict) and "result" in res else res
                if isinstance(parsed, dict) and parsed.get("matches"):
                    matches = [f"- {m.get('name')} ({m.get('code')})" for m in parsed["matches"][:5]]
                    ans = f"Found station(s) matching '{stn_query}':\n" + "\n".join(matches)
                    return GroundedAnswer(answer=ans, sources=[], sql=None, dont_know=False).to_dict()

        return None
