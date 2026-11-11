"""Unit tests for helpdesk.llm (multi-provider google-genai Gemini + Groq client)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from helpdesk.llm import (
    LLMAuthError,
    LLMClient,
    LLMConfig,
    LLMConfigurationError,
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    extract_json_object,
    redact_secrets,
)


class _FakeRateLimitError(Exception):
    status_code = 429


class _FakeAuthError(Exception):
    status_code = 401


class _FakeTimeoutError(TimeoutError):
    pass


class _FakeConnectionError(ConnectionError):
    pass


class _FakeGeminiAuthError(Exception):
    code = 401


class _FakeGeminiRateLimitError(Exception):
    code = 429


class _FakeGeminiTimeoutError(Exception):
    code = 504


class _FakeGeminiServerError(Exception):
    code = 503


def _make_fake_gemini_client(response_factory):
    """Build a minimal stub mimicking google.genai.Client & models.generate_content."""
    fake_client = MagicMock()
    fake_client.models = MagicMock()
    fake_client.models.generate_content = MagicMock(side_effect=response_factory)
    return fake_client


def test_config_defaults_to_gemini_and_hides_secrets(monkeypatch):
    monkeypatch.setattr("helpdesk.llm.load_dotenv", lambda *a, **kw: False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    cfg = LLMConfig.from_env(env_file=None)
    assert cfg.resolve_provider() == "gemini"
    assert cfg.active_model() == "gemini-3.5-flash"

    secret_cfg = LLMConfig(
        provider="groq",
        groq_api_key="gsk_supersecret1234567890",
        gemini_api_key="AIzaSySuperSecretKey123456",
    )
    assert secret_cfg.resolve_provider() == "groq"
    assert secret_cfg.active_model() == "openai/gpt-oss-120b"
    rendered = repr(secret_cfg)
    assert "gsk_supersecret" not in rendered
    assert "AIzaSySuperSecret" not in rendered


def test_auto_provider_prefers_gemini_then_groq():
    cfg_both = LLMConfig(
        provider="auto",
        gemini_api_key="AIza_fake_gemini",
        groq_api_key="gsk_fake_groq",
    )
    assert cfg_both.resolve_provider() == "gemini"

    cfg_groq_only = LLMConfig(provider="auto", gemini_api_key="", groq_api_key="gsk_fake_groq")
    assert cfg_groq_only.resolve_provider() == "groq"


def test_explicit_provider_does_not_silently_fallback():
    cfg = LLMConfig(provider="groq", groq_api_key="", gemini_api_key="AIza_present")
    client = LLMClient(config=cfg)
    with pytest.raises(LLMAuthError, match="GROQ_API_KEY is missing"):
        client.chat([{"role": "user", "content": "Hello"}])


def test_invalid_provider_raises_configuration_error():
    cfg = LLMConfig(provider="unknown_provider")
    with pytest.raises(LLMConfigurationError):
        cfg.resolve_provider()


def test_groq_successful_response_and_tool_call_parsing():
    raw_tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="search_documents", arguments='{"query": "Tatkal Aadhaar"}'),
    )
    raw_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[raw_tool_call]),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        model="openai/gpt-oss-120b",
    )
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = raw_resp

    cfg = LLMConfig(provider="groq", groq_api_key="gsk_test_key")
    client = LLMClient(config=cfg, groq_client=mock_groq)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search official documents",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    res = client.chat([{"role": "user", "content": "Need Aadhaar for Tatkal?"}], tools=tools)
    assert res.provider == "groq"
    assert res.model == "openai/gpt-oss-120b"
    assert res.has_tool_calls is True
    assert res.tool_calls[0].name == "search_documents"
    assert res.tool_calls[0].arguments == {"query": "Tatkal Aadhaar"}
    assert res.usage["total_tokens"] == 30


def test_gemini_successful_response_and_tool_translation_with_thought_signature():
    text_part = SimpleNamespace(text="Train 12002 departs New Delhi at 06:15.", function_call=None, thought=False)
    fc_part = SimpleNamespace(
        text=None,
        thought=False,
        thought_signature=b"sig_bytes_123",
        function_call=SimpleNamespace(
            id="call_gem_1",
            name="query_timetable",
            args={"train_number": "12002"},
        ),
    )

    responses = iter(
        [
            SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[fc_part]), finish_reason="STOP")],
                usage_metadata=SimpleNamespace(prompt_token_count=15, candidates_token_count=8, total_token_count=23),
            ),
            SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[text_part]), finish_reason="STOP")],
                usage_metadata=SimpleNamespace(prompt_token_count=25, candidates_token_count=12, total_token_count=37),
            ),
        ]
    )
    fake_client = _make_fake_gemini_client(lambda *a, **kw: next(responses))
    cfg = LLMConfig(provider="gemini", gemini_api_key="AIza_test_key")
    client = LLMClient(config=cfg, gemini_client=fake_client)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_timetable",
                "description": "Query SQLite timetable",
                "parameters": {
                    "type": "object",
                    "properties": {"train_number": {"type": "string"}},
                    "required": ["train_number"],
                },
            },
        }
    ]

    first = client.chat(
        [
            {"role": "system", "content": "Be grounded."},
            {"role": "user", "content": "When does 12002 depart?"},
        ],
        tools=tools,
    )
    assert first.provider == "gemini"
    assert first.has_tool_calls is True
    assert first.tool_calls[0].id == "call_gem_1"
    assert first.tool_calls[0].name == "query_timetable"
    assert first.tool_calls[0].arguments == {"train_number": "12002"}
    assert first.tool_calls[0].thought_signature == b"sig_bytes_123"

    # Simulate AIAssistant serializing tool_calls into OpenAI-style dicts (no signature field on dict)
    serialized_calls = [
        {
            "id": first.tool_calls[0].id,
            "type": "function",
            "function": {"name": "query_timetable", "arguments": '{"train_number": "12002"}'},
        }
    ]
    second = client.chat(
        [
            {"role": "system", "content": "Be grounded."},
            {"role": "user", "content": "When does 12002 depart?"},
            {"role": "assistant", "content": "", "tool_calls": serialized_calls},
            {
                "role": "tool",
                "tool_call_id": first.tool_calls[0].id,
                "name": "query_timetable",
                "content": '{"rows": [{"departure": "06:15"}]}',
            },
        ],
        tools=tools,
    )
    assert second.has_tool_calls is False
    assert "06:15" in second.content

    # Verify cached thought_signature and FunctionResponse.id were preserved in reconstructed contents
    sent_contents = fake_client.models.generate_content.call_args_list[1].kwargs["contents"]
    model_part = sent_contents[1].parts[0]
    tool_part = sent_contents[2].parts[0]
    assert model_part.thought_signature == b"sig_bytes_123"
    assert model_part.function_call.id == "call_gem_1"
    assert tool_part.function_response.id == "call_gem_1"


def test_gemini_filters_internal_thought_parts():
    thought_part = SimpleNamespace(text="Internal chain of thought...", function_call=None, thought=True)
    answer_part = SimpleNamespace(text='{"answer": "60 days", "sources": [], "dont_know": false}', function_call=None, thought=False)

    fake_client = _make_fake_gemini_client(
        lambda *a, **kw: SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[thought_part, answer_part]), finish_reason="STOP")],
            usage_metadata=None,
        )
    )
    client = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_ok"),
        gemini_client=fake_client,
    )
    parsed = client.generate_json([{"role": "user", "content": "ARP?"}])
    assert parsed["answer"] == "60 days"
    passed_cfg = fake_client.models.generate_content.call_args.kwargs["config"]
    assert passed_cfg.response_mime_type == "application/json"


def test_groq_auth_error_does_not_retry():
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = _FakeAuthError("401 Invalid API key gsk_secret123456")

    sleep_calls = []
    cfg = LLMConfig(provider="groq", groq_api_key="gsk_secret123456", max_retries=3)
    client = LLMClient(config=cfg, groq_client=mock_groq, sleep_fn=sleep_calls.append)

    with pytest.raises(LLMAuthError) as exc_info:
        client.chat([{"role": "user", "content": "Hi"}])

    assert mock_groq.chat.completions.create.call_count == 1
    assert sleep_calls == []
    assert "gsk_secret123456" not in str(exc_info.value)


def test_rate_limit_retries_and_succeeds():
    raw_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Recovered", tool_calls=[]), finish_reason="stop")],
        usage=None,
        model="openai/gpt-oss-120b",
    )
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = [
        _FakeRateLimitError("429 rate limit"),
        _FakeRateLimitError("429 rate limit"),
        raw_resp,
    ]
    sleep_calls = []
    cfg = LLMConfig(
        provider="groq",
        groq_api_key="gsk_test",
        max_retries=3,
        initial_backoff_seconds=0.5,
    )
    client = LLMClient(
        config=cfg,
        groq_client=mock_groq,
        sleep_fn=sleep_calls.append,
        jitter_fn=lambda a, b: 0.0,
    )

    res = client.chat([{"role": "user", "content": "Hello"}])
    assert res.content == "Recovered"
    assert mock_groq.chat.completions.create.call_count == 3
    assert sleep_calls == [0.5, 1.0]


def test_rate_limit_exhausted_raises():
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = _FakeRateLimitError("429 rate limit")
    cfg = LLMConfig(provider="groq", groq_api_key="gsk_test", max_retries=2)
    client = LLMClient(config=cfg, groq_client=mock_groq, sleep_fn=lambda _: None)

    with pytest.raises(LLMRateLimitError):
        client.chat([{"role": "user", "content": "Hello"}])
    assert mock_groq.chat.completions.create.call_count == 3


def test_timeout_and_connection_errors():
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = _FakeTimeoutError("Request timed out")
    cfg = LLMConfig(provider="groq", groq_api_key="gsk_test", max_retries=1)
    client = LLMClient(config=cfg, groq_client=mock_groq, sleep_fn=lambda _: None)

    with pytest.raises(LLMTimeoutError):
        client.chat([{"role": "user", "content": "Hello"}])
    assert mock_groq.chat.completions.create.call_count == 2

    mock_groq.chat.completions.create.reset_mock()
    mock_groq.chat.completions.create.side_effect = _FakeConnectionError("DNS resolution failed")
    with pytest.raises(LLMConnectionError):
        client.chat([{"role": "user", "content": "Hello"}])
    assert mock_groq.chat.completions.create.call_count == 2


def test_gemini_error_mapping():
    fake_auth = _make_fake_gemini_client(lambda *a, **kw: (_ for _ in ()).throw(_FakeGeminiAuthError("API_KEY_INVALID")))
    client = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_bad", max_retries=2),
        gemini_client=fake_auth,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(LLMAuthError):
        client.chat([{"role": "user", "content": "Hi"}])

    fake_rl = _make_fake_gemini_client(lambda *a, **kw: (_ for _ in ()).throw(_FakeGeminiRateLimitError("429 quota exceeded")))
    client_rl = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_ok", max_retries=1),
        gemini_client=fake_rl,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(LLMRateLimitError):
        client_rl.chat([{"role": "user", "content": "Hi"}])

    fake_to = _make_fake_gemini_client(lambda *a, **kw: (_ for _ in ()).throw(_FakeGeminiTimeoutError("DEADLINE_EXCEEDED")))
    client_to = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_ok", max_retries=1),
        gemini_client=fake_to,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(LLMTimeoutError):
        client_to.chat([{"role": "user", "content": "Hi"}])

    fake_srv = _make_fake_gemini_client(lambda *a, **kw: (_ for _ in ()).throw(_FakeGeminiServerError("ServiceUnavailable")))
    client_srv = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_ok", max_retries=1),
        gemini_client=fake_srv,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(LLMConnectionError):
        client_srv.chat([{"role": "user", "content": "Hi"}])


def test_invalid_responses_and_json_extraction():
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = SimpleNamespace(choices=[])
    client = LLMClient(
        config=LLMConfig(provider="groq", groq_api_key="gsk_ok", max_retries=0),
        groq_client=mock_groq,
    )
    with pytest.raises(LLMInvalidResponseError):
        client.chat([{"role": "user", "content": "Hi"}])

    bad_tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="sql_tool", arguments="{bad json"))
    mock_groq.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[bad_tc]))]
    )
    with pytest.raises(LLMInvalidResponseError):
        client.chat([{"role": "user", "content": "Hi"}])

    mock_groq.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='```json\n{"answer": "60 days", "sources": [], "dont_know": false}\n```',
                    tool_calls=[],
                )
            )
        ]
    )
    parsed = client.generate_json([{"role": "user", "content": "ARP?"}])
    assert parsed["answer"] == "60 days"

    with pytest.raises(LLMInvalidResponseError):
        extract_json_object("Not valid json at all")


def test_redact_secrets_masks_tokens():
    sample = "Failed using AIzaSyTestKey1234567890 and gsk_AbCdEfGhIjKlMnOp and Bearer tok_xyz"
    masked = redact_secrets(sample)
    assert "AIzaSyTestKey1234567890" not in masked
    assert "gsk_AbCdEfGhIjKlMnOp" not in masked
    assert "tok_xyz" not in masked


def test_neha_four_tool_schemas_translate_cleanly_for_gemini_and_groq():
    neha_schemas = [
        {
            "type": "function",
            "function": {
                "name": "station_search",
                "description": "Search for railway stations by name or partial name.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Station name."}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "train_search",
                "description": "Search for trains between two stations on a given date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "destination": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["source", "destination", "date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "train_details",
                "description": "Get detailed information about a specific train by its number.",
                "parameters": {
                    "type": "object",
                    "properties": {"train_number": {"type": "string"}},
                    "required": ["train_number"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "route_lookup",
                "description": "Look up the route for a given train number.",
                "parameters": {
                    "type": "object",
                    "properties": {"train_number": {"type": "string"}},
                    "required": ["train_number"],
                },
            },
        },
    ]

    text_part = SimpleNamespace(text="Found 4 tools.", function_call=None, thought=False)
    fake_gemini = _make_fake_gemini_client(
        lambda *a, **kw: SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[text_part]), finish_reason="STOP")],
            usage_metadata=None,
        )
    )
    gem_client = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_test"),
        gemini_client=fake_gemini,
    )
    gem_res = gem_client.chat([{"role": "user", "content": "Find trains"}], tools=neha_schemas)
    assert gem_res.content == "Found 4 tools."
    passed_cfg = fake_gemini.models.generate_content.call_args.kwargs["config"]
    decl_names = [fd.name for fd in passed_cfg.tools[0].function_declarations]
    assert decl_names == ["station_search", "train_search", "train_details", "route_lookup"]

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK", tool_calls=[]), finish_reason="stop")],
        usage=None,
        model="openai/gpt-oss-120b",
    )
    groq_client = LLMClient(
        config=LLMConfig(provider="groq", groq_api_key="gsk_test"),
        groq_client=mock_groq,
    )
    groq_res = groq_client.chat([{"role": "user", "content": "Find trains"}], tools=neha_schemas)
    assert groq_res.content == "OK"
    assert mock_groq.chat.completions.create.call_args.kwargs["tools"] == neha_schemas


def test_groq_multi_tool_call_and_tool_result_roundtrip():
    tc1 = SimpleNamespace(
        id="call_a",
        function=SimpleNamespace(name="station_search", arguments='{"query": "Jaipur"}'),
    )
    tc2 = SimpleNamespace(
        id="call_b",
        function=SimpleNamespace(name="station_search", arguments='{"query": "Udaipur"}'),
    )
    first_raw = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tc1, tc2]), finish_reason="tool_calls")],
        usage=None,
        model="openai/gpt-oss-120b",
    )
    second_raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Jaipur is JP and Udaipur City is UDZ.", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="openai/gpt-oss-120b",
    )
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = [first_raw, second_raw]

    client = LLMClient(
        config=LLMConfig(provider="groq", groq_api_key="gsk_test"),
        groq_client=mock_groq,
    )
    res1 = client.chat([{"role": "user", "content": "Codes for Jaipur and Udaipur?"}])
    assert len(res1.tool_calls) == 2
    assert [tc.name for tc in res1.tool_calls] == ["station_search", "station_search"]

    res2 = client.chat(
        [
            {"role": "user", "content": "Codes for Jaipur and Udaipur?"},
            {"role": "assistant", "content": "", "tool_calls": res1.tool_calls},
            {"role": "tool", "tool_call_id": "call_a", "name": "station_search", "content": '{"matches": [{"code": "JP"}]}'},
            {"role": "tool", "tool_call_id": "call_b", "name": "station_search", "content": '{"matches": [{"code": "UDZ"}]}'},
        ]
    )
    assert res2.has_tool_calls is False
    assert "JP" in res2.content and "UDZ" in res2.content
    sent_msgs = mock_groq.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert sent_msgs[1]["tool_calls"][0]["id"] == "call_a"
    assert sent_msgs[2]["role"] == "tool" and sent_msgs[3]["role"] == "tool"


def test_gemini_multi_tool_calls_invalid_response_and_connection_error():
    fc1 = SimpleNamespace(text=None, thought=False, function_call=SimpleNamespace(id="c1", name="station_search", args={"query": "Jaipur"}))
    fc2 = SimpleNamespace(text=None, thought=False, function_call=SimpleNamespace(id="c2", name="train_details", args={"train_number": "12315"}))

    fake_multi = _make_fake_gemini_client(
        lambda *a, **kw: SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[fc1, fc2]), finish_reason="STOP")],
            usage_metadata=None,
        )
    )
    client = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_ok"),
        gemini_client=fake_multi,
    )
    res = client.chat([{"role": "user", "content": "Lookup Jaipur and 12315"}])
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0].arguments == {"query": "Jaipur"}
    assert res.tool_calls[1].arguments == {"train_number": "12315"}

    fake_empty = _make_fake_gemini_client(
        lambda *a, **kw: SimpleNamespace(candidates=[], usage_metadata=None)
    )
    client_empty = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_ok", max_retries=0),
        gemini_client=fake_empty,
    )
    with pytest.raises(LLMInvalidResponseError):
        client_empty.chat([{"role": "user", "content": "Hi"}])

    fake_conn = _make_fake_gemini_client(
        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("Socket closed"))
    )
    client_conn = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_ok", max_retries=1),
        gemini_client=fake_conn,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(LLMConnectionError):
        client_conn.chat([{"role": "user", "content": "Hi"}])
