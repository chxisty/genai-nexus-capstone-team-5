"""Unit tests for helpdesk.memory_adapter."""

from __future__ import annotations

from helpdesk.memory_adapter import MemoryAdapter, MemoryBackendProtocol


class _FakePrasadBackend:
    """Mock simulating Member 4 (Prasad)'s external memory module API."""

    def __init__(self) -> None:
        self.sessions: dict[str, list[dict]] = {}

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    def create_session(self, session_id: str | None = None) -> str:
        sid = session_id or "prasad_sess_1"
        self.sessions.setdefault(sid, [])
        return sid

    def append_message(self, session_id: str, message: dict) -> None:
        self.sessions.setdefault(session_id, []).append(message)

    def get_messages(self, session_id: str) -> list[dict]:
        return list(self.sessions.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


def test_new_and_existing_session_with_default_mock():
    adapter = MemoryAdapter()
    assert adapter.uses_mock_backend is True
    assert isinstance(adapter._backend, MemoryBackendProtocol)

    sid = adapter.ensure_session()
    assert sid.startswith("sess_")
    assert adapter.has_session(sid) is True
    assert adapter.ensure_session(sid) == sid

    adapter.record_user_message(sid, "What is the Tatkal rule?")
    adapter.record_assistant_message(sid, "Aadhaar authentication is required.")

    history = adapter.get_raw_history(sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_missing_session_returns_empty_history():
    adapter = MemoryAdapter()
    assert adapter.has_session("nonexistent") is False
    assert adapter.get_raw_history("nonexistent") == []
    assert adapter.get_context_messages("nonexistent") == []


def test_tool_call_history_and_atomic_truncation():
    adapter = MemoryAdapter(default_max_messages=4, default_max_tokens=500)
    sid = adapter.ensure_session("s1")

    adapter.record_user_message(sid, "Old user question")
    adapter.record_assistant_message(sid, "Old answer")
    adapter.record_user_message(sid, "Which train leaves NDLS at 06:15?")
    adapter.record_tool_calls(
        sid,
        [{"id": "tc1", "type": "function", "function": {"name": "sql_tool", "arguments": "{}"}}],
    )
    adapter.record_tool_result(sid, tool_call_id="tc1", tool_name="sql_tool", result={"rows": [{"number": "12002"}]})
    adapter.record_assistant_message(sid, "Bhopal Shatabdi (12002) leaves NDLS at 06:15.")

    # Without tool history, clean user/assistant turns returned
    clean_ctx = adapter.get_context_messages(sid, include_tool_history=False)
    assert [m["role"] for m in clean_ctx] == ["user", "assistant", "user", "assistant"]

    # Tool history audit
    tools_only = adapter.get_tool_history(sid)
    assert len(tools_only) == 2
    assert tools_only[0]["role"] == "assistant"
    assert tools_only[1]["role"] == "tool"

    # With tool history and max_messages=3: assistant(tool_calls)+tool(result) stay grouped atomically
    ctx_with_tools = adapter.get_context_messages(sid, max_messages=3, include_tool_history=True)
    roles = [m["role"] for m in ctx_with_tools]
    assert roles == ["assistant", "tool", "assistant"]
    assert ctx_with_tools[0].get("tool_calls")


def test_token_budget_enforced():
    adapter = MemoryAdapter()
    sid = adapter.ensure_session("tok")

    adapter.record_user_message(sid, "First message " + ("x" * 200))
    adapter.record_assistant_message(sid, "First reply " + ("y" * 200))
    adapter.record_user_message(sid, "Short latest question")

    ctx = adapter.get_context_messages(sid, max_tokens=20)
    assert len(ctx) == 1
    assert ctx[0]["content"] == "Short latest question"


def test_integrates_with_prasad_custom_backend():
    prasad_store = _FakePrasadBackend()
    adapter = MemoryAdapter(backend=prasad_store)
    assert adapter.uses_mock_backend is False

    sid = adapter.ensure_session("team_session")
    adapter.record_user_message(sid, "Hi")
    adapter.record_assistant_message(sid, "Hello!")

    assert len(prasad_store.get_messages("team_session")) == 2
    ctx = adapter.get_context_messages("team_session")
    assert len(ctx) == 2
    adapter.clear_session("team_session")
    assert adapter.get_raw_history("team_session") == []

