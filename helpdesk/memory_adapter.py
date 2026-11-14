"""Conversation memory adapter coordinating with Member 4 (Prasad)'s session store.

Provides a clean adapter layer over an injected session/memory backend for
session resolution, message and tool-call history recording, missing-session
safety, message-count limits, and token-aware context budgeting.

Note: ``InMemorySessionMock`` is an ephemeral in-memory stub used ONLY when
Prasad's production session store is not yet supplied. It is NOT a competing
persistent memory database.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable
import uuid


def default_token_estimator(text: str) -> int:
    """Estimate token count conservatively (~4 characters per token + small framing)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_message_tokens(
    message: Mapping[str, Any],
    token_counter: Callable[[str], int] = default_token_estimator,
) -> int:
    """Estimate token usage for a single normalized conversation message."""
    total = 4  # Role and structural framing overhead
    content = message.get("content")
    if content:
        total += token_counter(str(content))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        try:
            serialized = json.dumps(tool_calls, ensure_ascii=False, default=str)
        except TypeError:
            serialized = str(tool_calls)
        total += token_counter(serialized)
    if message.get("name"):
        total += token_counter(str(message["name"]))
    return total


@runtime_checkable
class MemoryBackendProtocol(Protocol):
    """Expected integration contract for Member 4 (Prasad)'s memory store."""

    def create_session(self, session_id: str | None = None) -> str:
        """Create (or return existing) session and return its session_id."""

    def has_session(self, session_id: str) -> bool:
        """Return True if the session exists."""

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return stored messages for session_id in chronological order."""

    def add_message(self, session_id: str, message: dict[str, Any]) -> None:
        """Append a single message record to session_id."""

    def clear_session(self, session_id: str) -> None:
        """Remove stored history for session_id."""


class InMemorySessionMock:
    """Ephemeral in-memory mock backend used when Prasad's module is unavailable."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    def create_session(self, session_id: str | None = None) -> str:
        sid = (session_id or "").strip() or f"sess_{uuid.uuid4().hex[:12]}"
        self._sessions.setdefault(sid, [])
        return sid

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        return deepcopy(self._sessions.get(session_id, []))

    def add_message(self, session_id: str, message: dict[str, Any]) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(deepcopy(message))

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class MemoryAdapter:
    """Adapter connecting the AI assistant to Prasad's session/context module."""

    def __init__(
        self,
        backend: Any | None = None,
        *,
        default_max_messages: int = 12,
        default_max_tokens: int = 2500,
        token_counter: Callable[[str], int] = default_token_estimator,
    ) -> None:
        self._backend = backend if backend is not None else InMemorySessionMock()
        self.uses_mock_backend = backend is None
        self.default_max_messages = max(1, default_max_messages)
        self.default_max_tokens = max(64, default_max_tokens)
        self._token_counter = token_counter

    def ensure_session(self, session_id: str | None = None) -> str:
        """Resolve an existing session ID or create one cleanly if missing."""
        cleaned = (session_id or "").strip()
        if cleaned and self.has_session(cleaned):
            return cleaned

        if hasattr(self._backend, "create_session"):
            created = self._backend.create_session(cleaned or None)
            return str(created)
        if hasattr(self._backend, "get_or_create_session"):
            created = self._backend.get_or_create_session(cleaned or None)
            return str(created)
        return cleaned or f"sess_{uuid.uuid4().hex[:12]}"

    def has_session(self, session_id: str) -> bool:
        """Check whether a session exists in the underlying backend."""
        cleaned = (session_id or "").strip()
        if not cleaned:
            return False
        if hasattr(self._backend, "has_session"):
            return bool(self._backend.has_session(cleaned))
        if hasattr(self._backend, "session_exists"):
            return bool(self._backend.session_exists(cleaned))
        if hasattr(self._backend, "get_session"):
            return self._backend.get_session(cleaned) is not None
        # Fallback: inspect history if backend only exposes get_history
        return bool(self._fetch_raw_from_backend(cleaned))

    def _fetch_raw_from_backend(self, session_id: str) -> list[dict[str, Any]]:
        for method_name in ("get_history", "get_messages", "load_history"):
            method = getattr(self._backend, method_name, None)
            if callable(method):
                raw = method(session_id)
                return [dict(m) for m in (raw or []) if isinstance(m, Mapping)]
        raise AttributeError(
            "Memory backend must implement get_history(session_id), "
            "get_messages(session_id), or load_history(session_id)."
        )

    def _append_to_backend(self, session_id: str, record: dict[str, Any]) -> None:
        sid = self.ensure_session(session_id)
        for method_name in ("add_message", "append_message", "save_message"):
            method = getattr(self._backend, method_name, None)
            if callable(method):
                method(sid, record)
                return
        raise AttributeError(
            "Memory backend must implement add_message(session_id, message), "
            "append_message(session_id, message), or save_message(session_id, message)."
        )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def get_raw_history(self, session_id: str | None) -> list[dict[str, Any]]:
        """Retrieve complete chronological message history; return [] if missing."""
        cleaned = (session_id or "").strip()
        if not cleaned or not self.has_session(cleaned):
            return []
        return self._fetch_raw_from_backend(cleaned)

    def record_user_message(
        self,
        session_id: str | None,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Persist a user message and return the resolved session ID."""
        sid = self.ensure_session(session_id)
        record: dict[str, Any] = {
            "role": "user",
            "content": str(content),
            "timestamp": self._now_iso(),
        }
        if metadata:
            record["metadata"] = dict(metadata)
        self._append_to_backend(sid, record)
        return sid

    def record_assistant_message(
        self,
        session_id: str | None,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Persist a final assistant response and optional grounding metadata."""
        sid = self.ensure_session(session_id)
        record: dict[str, Any] = {
            "role": "assistant",
            "content": str(content),
            "timestamp": self._now_iso(),
        }
        if metadata:
            record["metadata"] = dict(metadata)
        self._append_to_backend(sid, record)
        return sid

    def record_tool_calls(
        self,
        session_id: str | None,
        tool_calls: Sequence[Mapping[str, Any]],
        content: str = "",
    ) -> str:
        """Persist an assistant message containing requested tool calls."""
        sid = self.ensure_session(session_id)
        record: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "tool_calls": [dict(tc) for tc in tool_calls],
            "timestamp": self._now_iso(),
        }
        self._append_to_backend(sid, record)
        return sid

    def record_tool_result(
        self,
        session_id: str | None,
        *,
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> str:
        """Persist a tool execution result associated with its tool_call_id."""
        sid = self.ensure_session(session_id)
        if isinstance(result, str):
            serialized = result
        else:
            serialized = json.dumps(result, ensure_ascii=False, default=str)
        record: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": str(tool_call_id),
            "name": str(tool_name),
            "content": serialized,
            "timestamp": self._now_iso(),
        }
        self._append_to_backend(sid, record)
        return sid

    def get_tool_history(self, session_id: str | None) -> list[dict[str, Any]]:
        """Return only tool-call and tool-result entries for diagnostics/auditing."""
        history = self.get_raw_history(session_id)
        return [
            msg
            for msg in history
            if msg.get("role") == "tool" or (msg.get("role") == "assistant" and msg.get("tool_calls"))
        ]

    def clear_session(self, session_id: str | None) -> None:
        """Clear a session safely even if it does not exist."""
        cleaned = (session_id or "").strip()
        if not cleaned:
            return
        for method_name in ("clear_session", "reset_session", "delete_session"):
            method = getattr(self._backend, method_name, None)
            if callable(method):
                method(cleaned)
                return

    def get_context_messages(
        self,
        session_id: str | None,
        *,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        include_tool_history: bool = False,
    ) -> list[dict[str, Any]]:
        """Prepare recent LLM-ready context messages respecting message and token limits."""
        raw = self.get_raw_history(session_id)
        if not raw:
            return []

        msg_limit = self.default_max_messages if max_messages is None else max(0, max_messages)
        tok_limit = self.default_max_tokens if max_tokens is None else max(0, max_tokens)
        if msg_limit == 0 or tok_limit == 0:
            return []

        filtered: list[dict[str, Any]] = []
        for entry in raw:
            role = str(entry.get("role", "")).lower()
            if not include_tool_history:
                if role == "tool" or (role == "assistant" and entry.get("tool_calls")):
                    continue
                if role not in {"user", "assistant"}:
                    continue
                filtered.append({"role": role, "content": str(entry.get("content", ""))})
            else:
                if role not in {"user", "assistant", "tool"}:
                    continue
                normalized: dict[str, Any] = {
                    "role": role,
                    "content": str(entry.get("content", "")),
                }
                if entry.get("tool_calls"):
                    normalized["tool_calls"] = deepcopy(entry["tool_calls"])
                if entry.get("tool_call_id"):
                    normalized["tool_call_id"] = str(entry["tool_call_id"])
                if entry.get("name"):
                    normalized["name"] = str(entry["name"])
                filtered.append(normalized)

        if not filtered:
            return []

        # Group tool-calling assistant turns with their tool responses so truncation
        # never splits a tool result away from its parent assistant tool_calls turn.
        blocks = self._group_atomic_blocks(filtered)

        selected_blocks: list[list[dict[str, Any]]] = []
        used_messages = 0
        used_tokens = 0

        for block in reversed(blocks):
            block_msg_count = len(block)
            block_tokens = sum(
                estimate_message_tokens(m, token_counter=self._token_counter) for m in block
            )
            if used_messages + block_msg_count > msg_limit:
                break
            if used_tokens + block_tokens > tok_limit:
                # If even the single latest turn exceeds budget, clip its content safely
                # only if nothing has been selected yet and it's a single plain message.
                if not selected_blocks and block_msg_count == 1 and not block[0].get("tool_calls"):
                    clipped = self._clip_single_message(block[0], tok_limit)
                    if clipped is not None:
                        selected_blocks.append([clipped])
                break
            selected_blocks.append(block)
            used_messages += block_msg_count
            used_tokens += block_tokens

        selected_blocks.reverse()
        return [msg for block in selected_blocks for msg in block]

    @staticmethod
    def _group_atomic_blocks(messages: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        blocks: list[list[dict[str, Any]]] = []
        idx = 0
        n = len(messages)
        while idx < n:
            current = messages[idx]
            if current.get("role") == "assistant" and current.get("tool_calls"):
                group = [current]
                idx += 1
                while idx < n and messages[idx].get("role") == "tool":
                    group.append(messages[idx])
                    idx += 1
                blocks.append(group)
            else:
                blocks.append([current])
                idx += 1
        return blocks

    def _clip_single_message(
        self,
        message: dict[str, Any],
        token_budget: int,
    ) -> dict[str, Any] | None:
        overhead = 4
        available_tokens = token_budget - overhead
        if available_tokens <= 0:
            return None
        char_limit = max(1, available_tokens * 4)
        content = str(message.get("content", ""))
        if len(content) <= char_limit:
            return dict(message)
        truncated = content[: max(1, char_limit - 3)].rstrip() + "..."
        return {"role": message["role"], "content": truncated}

