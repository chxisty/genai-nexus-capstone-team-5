"""Unit tests for helpdesk.llm (multi-provider Gemini + Groq client)."""

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


def _make_fake_gemini_module(response_factory):
    """Build a minimal stub mimicking google.generativeai & protos for unit tests."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as real_genai

    fake_mod = MagicMock()
    fake_mod.protos = real_genai.protos
    fake_mod.types = real_genai.types
    fake_mod.configure = MagicMock()

    def _model_ctor(**kwargs):
        model = MagicMock()
        model.init_kwargs = kwargs
        model.generate_content = MagicMock(side_effect=response_factory)
        return model

    fake_mod.GenerativeModel = MagicMock(side_effect=_model_ctor)
    return fake_mod


def test_config_defaults_to_gemini_and_hides_secrets(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    cfg = LLMConfig.from_env(env_file=None)
    assert cfg.resolve_provider() == "gemini"
    assert cfg.active_model() == "gemini-2.5-flash"

    secret_cfg = LLMConfig(
        provider="groq",
        groq_api_key="gsk_supersecret1234567890",
        gemini_api_key="AIzaSySuperSecretKey123456",
    )
    assert secret_cfg.resolve_provider() == "groq"
    assert secret_cfg.active_model() == "llama-3.3-70b-versatile"
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
        model="llama-3.3-70b-versatile",
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
    assert res.has_tool_calls is True
    assert res.tool_calls[0].name == "search_documents"
    assert res.tool_calls[0].arguments == {"query": "Tatkal Aadhaar"}
    assert res.usage["total_tokens"] == 30


def test_gemini_successful_response_and_tool_translation():
    text_part = SimpleNamespace(text="Train 12002 departs New Delhi at 06:15.", function_call=None)
    fc_part = SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name="query_timetable", args={"train_number": "12002"}),
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
    fake_gemini = _make_fake_gemini_module(lambda *a, **kw: next(responses))
    cfg = LLMConfig(provider="gemini", gemini_api_key="AIza_test_key")
    client = LLMClient(config=cfg, gemini_module=fake_gemini)

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
    assert first.tool_calls[0].name == "query_timetable"
    assert first.tool_calls[0].arguments == {"train_number": "12002"}

    second = client.chat(
        [
            {"role": "system", "content": "Be grounded."},
            {"role": "user", "content": "When does 12002 depart?"},
            {"role": "assistant", "content": "", "tool_calls": first.tool_calls},
            {
                "role": "tool",
                "tool_call_id": first.tool_calls[0].id,
                "name": "query_timetable",
                "content": '{"rows": [{"departure": "06:15"}]}',
            },
        ]
    )
    assert second.has_tool_calls is False
    assert "06:15" in second.content


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
        model="llama-3.3-70b-versatile",
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
    fake_auth = _make_fake_gemini_module(lambda *a, **kw: (_ for _ in ()).throw(_FakeGeminiAuthError("API_KEY_INVALID")))
    client = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_bad", max_retries=2),
        gemini_module=fake_auth,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(LLMAuthError):
        client.chat([{"role": "user", "content": "Hi"}])

    fake_rl = _make_fake_gemini_module(lambda *a, **kw: (_ for _ in ()).throw(_FakeGeminiRateLimitError("429 quota exceeded")))
    client_rl = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_ok", max_retries=1),
        gemini_module=fake_rl,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(LLMRateLimitError):
        client_rl.chat([{"role": "user", "content": "Hi"}])

    fake_to = _make_fake_gemini_module(lambda *a, **kw: (_ for _ in ()).throw(_FakeGeminiTimeoutError("DeadlineExceeded")))
    client_to = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_ok", max_retries=1),
        gemini_module=fake_to,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(LLMTimeoutError):
        client_to.chat([{"role": "user", "content": "Hi"}])


def test_invalid_responses_and_json_extraction():
    # Empty choices
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = SimpleNamespace(choices=[])
    client = LLMClient(
        config=LLMConfig(provider="groq", groq_api_key="gsk_ok", max_retries=0),
        groq_client=mock_groq,
    )
    with pytest.raises(LLMInvalidResponseError):
        client.chat([{"role": "user", "content": "Hi"}])

    # Malformed tool arguments JSON
    bad_tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="sql_tool", arguments="{bad json"))
    mock_groq.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[bad_tc]))]
    )
    with pytest.raises(LLMInvalidResponseError):
        client.chat([{"role": "user", "content": "Hi"}])

    # Markdown-fenced JSON parsing works cleanly
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

    text_part = SimpleNamespace(text="Found 4 tools.", function_call=None)
    fake_gemini = _make_fake_gemini_module(
        lambda *a, **kw: SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[text_part]), finish_reason="STOP")],
            usage_metadata=None,
        )
    )
    gem_client = LLMClient(
        config=LLMConfig(provider="gemini", gemini_api_key="AIza_test"),
        gemini_module=fake_gemini,
    )
    gem_res = gem_client.chat([{"role": "user", "content": "Find trains"}], tools=neha_schemas)
    assert gem_res.content == "Found 4 tools."
    passed_tools = fake_gemini.GenerativeModel.call_args.kwargs["tools"]
    decl_names = [fd.name for fd in passed_tools[0].function_declarations]
    assert decl_names == ["station_search", "train_search", "train_details", "route_lookup"]

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK", tool_calls=[]), finish_reason="stop")],
        usage=None,
        model="llama-3.3-70b-versatile",
    )
    groq_client = LLMClient(
        config=LLMConfig(provider="groq", groq_api_key="gsk_test"),
        groq_client=mock_groq,
    )
    groq_res = groq_client.chat([{"role": "user", "content": "Find trains"}], tools=neha_schemas)
    assert groq_res.content == "OK"
    assert mock_groq.chat.completions.create.call_args.kwargs["tools"] == neha_schemas
