"""Configurable Multi-Provider LLM Client for Railway Passenger Help Desk.

Supports both Google Gemini (via the official ``google-genai`` SDK, default for
official capstone evaluation) and Groq (configurable alternative for development),
normalized behind a common interface with tool-calling translation, Gemini 3.x
``thought_signature`` preservation, structured JSON extraction, transient retry
with exponential backoff, and secret-safe logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import logging
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
SUPPORTED_PROVIDERS = ("gemini", "groq", "auto")

_KEY_REDACTION_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{10,}"),
    re.compile(r"gsk_[0-9A-Za-z]{10,}"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)([^\s,;\"']+)"),
    re.compile(r"(?i)(bearer\s+)([^\s,;\"']+)"),
)


def redact_secrets(text: str, extra_secrets: Sequence[str] = ()) -> str:
    """Mask API keys and token-like substrings before logging or surfacing errors."""
    if not text:
        return text
    masked = text
    for secret in extra_secrets:
        if secret and len(secret) >= 6:
            masked = masked.replace(secret, "[REDACTED]")
    for pattern in _KEY_REDACTION_PATTERNS:
        if pattern.groups >= 2:
            masked = pattern.sub(r"\1[REDACTED]", masked)
        else:
            masked = pattern.sub("[REDACTED]", masked)
    return masked


class LLMError(Exception):
    """Base exception for all LLM client errors."""


class LLMConfigurationError(LLMError):
    """Raised when provider or client configuration is invalid."""


class LLMAuthError(LLMError):
    """Raised when an API key is missing, invalid, or lacks permission."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider throttles requests (HTTP 429 / quota)."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM API request exceeds the configured timeout."""


class LLMConnectionError(LLMError):
    """Raised on network, DNS, or transport failures reaching the provider."""


class LLMInvalidResponseError(LLMError):
    """Raised when the provider returns an empty, blocked, or unparseable response."""


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool/function call emitted by an LLM."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    thought_signature: bytes | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned across all configured LLM providers."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Return True when the model requested one or more tool calls."""
        return bool(self.tool_calls)


@dataclass
class LLMConfig:
    """Configuration for multi-provider LLM client initialization and retries."""

    provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    groq_api_key: str = ""
    groq_model: str = DEFAULT_GROQ_MODEL
    timeout_seconds: float = 120.0
    max_retries: int = 3
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 16.0
    temperature: float = 0.1
    max_tokens: int = 2048

    def __repr__(self) -> str:
        return (
            "LLMConfig("
            f"provider={self.provider!r}, "
            f"gemini_model={self.gemini_model!r}, "
            f"gemini_api_key={'<set>' if self.gemini_api_key else '<unset>'}, "
            f"groq_model={self.groq_model!r}, "
            f"groq_api_key={'<set>' if self.groq_api_key else '<unset>'}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"max_retries={self.max_retries})"
        )

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = None,
        override: bool = False,
    ) -> "LLMConfig":
        """Build configuration from environment variables and an optional .env file."""
        if env_file is not None:
            load_dotenv(dotenv_path=Path(env_file), override=override)
        else:
            load_dotenv(override=override)

        raw_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        provider = raw_provider or "gemini"

        return cls(
            provider=provider,
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL,
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model=os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL,
            timeout_seconds=_parse_timeout_env(120.0),
            max_retries=_parse_int_env("LLM_MAX_RETRIES", 3),
            initial_backoff_seconds=_parse_float_env("LLM_INITIAL_BACKOFF_SECONDS", 1.0),
            backoff_multiplier=_parse_float_env("LLM_BACKOFF_MULTIPLIER", 2.0),
            max_backoff_seconds=_parse_float_env("LLM_MAX_BACKOFF_SECONDS", 16.0),
            temperature=_parse_float_env("LLM_TEMPERATURE", 0.1),
            max_tokens=_parse_int_env("LLM_MAX_TOKENS", 2048),
        )

    def resolve_provider(self) -> str:
        """Resolve the active provider cleanly without silent cross-provider overrides.

        When ``provider`` is ``"gemini"`` or ``"groq"``, that exact provider is
        enforced. When ``"auto"``, Gemini is preferred first (matching official
        evaluation expectations), falling back to Groq only when Gemini has no key
        and Groq does.
        """
        normalized = (self.provider or "gemini").strip().lower()
        if normalized not in SUPPORTED_PROVIDERS:
            raise LLMConfigurationError(
                f"Unsupported LLM_PROVIDER {self.provider!r}. "
                f"Expected one of: {', '.join(SUPPORTED_PROVIDERS)}."
            )
        if normalized == "auto":
            if self.gemini_api_key:
                return "gemini"
            if self.groq_api_key:
                return "groq"
            return "gemini"
        return normalized

    def active_model(self, resolved_provider: str | None = None) -> str:
        """Return the configured model name for the active provider."""
        active = resolved_provider or self.resolve_provider()
        if active == "groq":
            return self.groq_model or DEFAULT_GROQ_MODEL
        return self.gemini_model or DEFAULT_GEMINI_MODEL

    def active_api_key(self, resolved_provider: str | None = None) -> str:
        """Return the API key for the active provider, or raise LLMAuthError if missing."""
        active = resolved_provider or self.resolve_provider()
        if active == "groq":
            if not self.groq_api_key:
                raise LLMAuthError(
                    "GROQ_API_KEY is missing. Set GROQ_API_KEY in your local .env file "
                    "or switch LLM_PROVIDER=gemini."
                )
            return self.groq_api_key
        if not self.gemini_api_key:
            raise LLMAuthError(
                "GEMINI_API_KEY is missing. Set GEMINI_API_KEY in your local .env file "
                "or switch LLM_PROVIDER=groq."
            )
        return self.gemini_api_key

    def known_secrets(self) -> tuple[str, ...]:
        """Return configured secret strings so log/error sanitizers can redact them."""
        return tuple(k for k in (self.gemini_api_key, self.groq_api_key) if k)


def _parse_timeout_env(default: float = 120.0) -> float:
    """Parse timeout in seconds from LLM_TIMEOUT_SECONDS, LLM_TIMEOUT_MS, or TIMEOUT.

    Values >= 1000 or keys ending in _MS are automatically converted from milliseconds
    to seconds (e.g., 120000 -> 120.0).
    """
    for key in ("LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT_MS", "TIMEOUT"):
        raw = os.getenv(key, "").strip()
        if raw:
            try:
                val = float(raw)
                if key.endswith("_MS") or val >= 1000.0:
                    return val / 1000.0
                return val
            except ValueError as exc:
                raise LLMConfigurationError(
                    f"Environment variable {key} must be a valid number, got {raw!r}."
                ) from exc
    return default


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMConfigurationError(f"Environment variable {name} must be a float, got {raw!r}.") from exc


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LLMConfigurationError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model output, tolerating markdown code fences."""
    if not text or not text.strip():
        raise LLMInvalidResponseError("LLM returned empty text when JSON was expected.")

    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        raise LLMInvalidResponseError(
            f"Expected JSON object from LLM, got {type(parsed).__name__}."
        )
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError as exc:
                raise LLMInvalidResponseError(
                    f"Could not parse valid JSON object from LLM response: {exc}"
                ) from exc
        raise LLMInvalidResponseError("LLM response did not contain a valid JSON object.")


def _normalize_json_type(json_type: str | None, types_mod: Any) -> Any:
    mapping = {
        "string": types_mod.Type.STRING,
        "integer": types_mod.Type.INTEGER,
        "number": types_mod.Type.NUMBER,
        "boolean": types_mod.Type.BOOLEAN,
        "array": types_mod.Type.ARRAY,
        "object": types_mod.Type.OBJECT,
    }
    return mapping.get((json_type or "string").lower(), types_mod.Type.STRING)


def _json_schema_to_gemini_schema(schema: Mapping[str, Any], types_mod: Any) -> Any:
    """Recursively convert an OpenAI/JSON Schema dict to a google.genai.types.Schema."""
    raw_type = schema.get("type", "object")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        raw_type = non_null[0] if non_null else "string"

    schema_type = _normalize_json_type(raw_type, types_mod)
    kwargs: dict[str, Any] = {"type": schema_type}
    if schema.get("description"):
        kwargs["description"] = str(schema["description"])
    if schema.get("enum"):
        kwargs["enum"] = [str(item) for item in schema["enum"]]

    if schema_type == types_mod.Type.OBJECT:
        raw_props = schema.get("properties") or {}
        props = {
            str(prop_name): _json_schema_to_gemini_schema(prop_schema, types_mod)
            for prop_name, prop_schema in raw_props.items()
            if isinstance(prop_schema, Mapping)
        }
        if props:
            kwargs["properties"] = props
        if schema.get("required"):
            kwargs["required"] = [str(r) for r in schema["required"]]
    elif schema_type == types_mod.Type.ARRAY:
        items_schema = schema.get("items")
        if isinstance(items_schema, Mapping):
            kwargs["items"] = _json_schema_to_gemini_schema(items_schema, types_mod)
        else:
            kwargs["items"] = types_mod.Schema(type=types_mod.Type.STRING)

    return types_mod.Schema(**kwargs)


def _convert_openai_tools_to_gemini(
    tools: Sequence[Mapping[str, Any]] | None,
    types_mod: Any,
) -> list[Any] | None:
    """Convert OpenAI-compatible tool declarations into google.genai.types.Tool objects."""
    if not tools:
        return None

    declarations: list[Any] = []
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, Mapping) else {}
        name = str(fn.get("name", "")).strip()
        if not name:
            raise LLMConfigurationError("Tool definition is missing 'function.name'.")
        description = str(fn.get("description", ""))
        params = fn.get("parameters")

        fd_kwargs: dict[str, Any] = {"name": name, "description": description}
        if isinstance(params, Mapping) and (params.get("properties") or params.get("required")):
            fd_kwargs["parameters"] = _json_schema_to_gemini_schema(params, types_mod)
        declarations.append(types_mod.FunctionDeclaration(**fd_kwargs))

    return [types_mod.Tool(function_declarations=declarations)] if declarations else None


def _to_plain_python(value: Any) -> Any:
    """Recursively convert SDK mapping/sequence containers into plain Python types."""
    if isinstance(value, Mapping):
        return {str(k): _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_python(v) for v in value]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
        if hasattr(value, "items"):
            return {str(k): _to_plain_python(v) for k, v in value.items()}
        return [_to_plain_python(v) for v in value]
    return value


def _sanitize_groq_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize internal messages into valid OpenAI/Groq chat completion payloads."""
    formatted: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        entry: dict[str, Any] = {"role": role}
        if msg.get("content") is not None:
            entry["content"] = str(msg["content"])
        elif role != "assistant":
            entry["content"] = ""

        if role == "assistant" and msg.get("tool_calls"):
            serialized_calls = []
            for tc in msg["tool_calls"]:
                if isinstance(tc, ToolCall):
                    args_str = tc.raw_arguments or json.dumps(tc.arguments, ensure_ascii=False)
                    serialized_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": args_str},
                        }
                    )
                elif isinstance(tc, Mapping):
                    if "function" in tc:
                        serialized_calls.append(dict(tc))
                    else:
                        args_val = tc.get("arguments", {})
                        args_str = args_val if isinstance(args_val, str) else json.dumps(args_val, ensure_ascii=False)
                        serialized_calls.append(
                            {
                                "id": str(tc.get("id", f"call_{uuid.uuid4().hex[:8]}")),
                                "type": "function",
                                "function": {
                                    "name": str(tc.get("name", "")),
                                    "arguments": args_str,
                                },
                            }
                        )
            entry["tool_calls"] = serialized_calls
            if "content" not in entry:
                entry["content"] = ""

        if role == "tool":
            if msg.get("tool_call_id"):
                entry["tool_call_id"] = str(msg["tool_call_id"])
            if msg.get("name"):
                entry["name"] = str(msg["name"])

        formatted.append(entry)
    return formatted


def _convert_messages_to_gemini(
    messages: Sequence[Mapping[str, Any]],
    types_mod: Any,
    thought_signatures: Mapping[str, bytes] | None = None,
) -> tuple[str | None, list[Any]]:
    """Translate OpenAI-style chat messages into Gemini (system_instruction, contents).

    Preserves Gemini 3.x ``thought_signature`` bytes and ``FunctionCall.id`` /
    ``FunctionResponse.id`` across serialized tool-calling turns.
    """
    sig_lookup = thought_signatures or {}
    system_parts: list[str] = []
    contents: list[Any] = []
    pending_tool_parts: list[Any] = []

    def flush_tool_parts() -> None:
        if pending_tool_parts:
            contents.append(types_mod.Content(role="user", parts=list(pending_tool_parts)))
            pending_tool_parts.clear()

    for msg in messages:
        role = str(msg.get("role", "user")).lower()
        content_text = msg.get("content")

        if role == "system":
            if content_text:
                system_parts.append(str(content_text))
            continue

        if role == "tool":
            tool_name = str(msg.get("name") or "tool")
            raw_content = content_text if content_text is not None else "{}"
            try:
                parsed = json.loads(str(raw_content))
                response_payload = parsed if isinstance(parsed, dict) else {"result": parsed}
            except (json.JSONDecodeError, TypeError):
                response_payload = {"result": str(raw_content)}

            fr_kwargs: dict[str, Any] = {
                "name": tool_name,
                "response": response_payload,
            }
            if msg.get("tool_call_id"):
                fr_kwargs["id"] = str(msg["tool_call_id"])

            pending_tool_parts.append(
                types_mod.Part(function_response=types_mod.FunctionResponse(**fr_kwargs))
            )
            continue

        flush_tool_parts()

        if role == "assistant":
            parts: list[Any] = []
            if content_text:
                parts.append(types_mod.Part.from_text(text=str(content_text)))
            for tc in msg.get("tool_calls") or []:
                tc_id = ""
                tc_sig: bytes | None = None
                if isinstance(tc, ToolCall):
                    tc_id = tc.id
                    tc_name = tc.name
                    tc_args = tc.arguments
                    tc_sig = tc.thought_signature
                elif isinstance(tc, Mapping):
                    tc_id = str(tc.get("id", ""))
                    raw_sig = tc.get("thought_signature")
                    if isinstance(raw_sig, (bytes, bytearray)):
                        tc_sig = bytes(raw_sig)
                    fn = tc.get("function", tc)
                    tc_name = str(fn.get("name", ""))
                    raw_args = fn.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            tc_args = json.loads(raw_args) if raw_args.strip() else {}
                        except json.JSONDecodeError:
                            tc_args = {}
                    elif isinstance(raw_args, Mapping):
                        tc_args = dict(raw_args)
                    else:
                        tc_args = {}
                else:
                    continue

                if tc_sig is None and tc_id and tc_id in sig_lookup:
                    tc_sig = sig_lookup[tc_id]

                if tc_name:
                    fc_kwargs: dict[str, Any] = {"name": tc_name, "args": tc_args}
                    if tc_id:
                        fc_kwargs["id"] = tc_id
                    part_kwargs: dict[str, Any] = {
                        "function_call": types_mod.FunctionCall(**fc_kwargs)
                    }
                    if tc_sig is not None:
                        part_kwargs["thought_signature"] = tc_sig
                    parts.append(types_mod.Part(**part_kwargs))
            if not parts:
                parts.append(types_mod.Part.from_text(text=""))
            contents.append(types_mod.Content(role="model", parts=parts))
        else:
            contents.append(
                types_mod.Content(
                    role="user",
                    parts=[types_mod.Part.from_text(text=str(content_text or ""))],
                )
            )

    flush_tool_parts()
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


class LLMClient:
    """Unified multi-provider LLM client supporting Gemini (google-genai) and Groq.

    Handles provider resolution, lazy SDK client setup, tool-calling normalization,
    Gemini 3.x thought_signature caching, retry with exponential backoff on transient
    failures, and secret-safe error mapping.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        groq_client: Any = None,
        gemini_client: Any = None,
        gemini_module: Any = None,
        gemini_types: Any = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        jitter_fn: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.config = replace(config) if config is not None else LLMConfig.from_env()
        if provider is not None:
            self.config.provider = provider.strip().lower()

        self.provider = self.config.resolve_provider()

        if model is not None:
            if self.provider == "groq":
                self.config.groq_model = model
            else:
                self.config.gemini_model = model
        if api_key is not None:
            if self.provider == "groq":
                self.config.groq_api_key = api_key
            else:
                self.config.gemini_api_key = api_key

        self.model = self.config.active_model(self.provider)
        self._groq_client = groq_client
        self._gemini_client = gemini_client
        self._gemini_module = gemini_module
        self._gemini_types = gemini_types
        self._sleep = sleep_fn
        self._jitter = jitter_fn
        # Cache thought_signature bytes by tool_call_id so serialized OpenAI-style
        # assistant turns preserve Gemini 3.x signatures across AIAssistant loops.
        self._thought_signatures: dict[str, bytes] = {}

        logger.info("Initialized LLMClient with provider=%s, model=%s", self.provider, self.model)

    def _safe_error_message(self, exc: Exception) -> str:
        return redact_secrets(str(exc), self.config.known_secrets())

    def _get_groq_client(self) -> Any:
        api_key = self.config.active_api_key("groq")
        if self._groq_client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise LLMConfigurationError(
                    "The 'groq' package is not installed. Run: pip install groq"
                ) from exc
            self._groq_client = Groq(
                api_key=api_key,
                timeout=self.config.timeout_seconds,
                max_retries=0,
            )
        return self._groq_client

    def _get_gemini_types(self) -> Any:
        if self._gemini_types is None:
            try:
                from google.genai import types
            except ImportError as exc:
                raise LLMConfigurationError(
                    "The 'google-genai' package is not installed. Run: pip install google-genai"
                ) from exc
            self._gemini_types = types
        return self._gemini_types

    def _get_gemini_client(self) -> Any:
        api_key = self.config.active_api_key("gemini")
        types_mod = self._get_gemini_types()
        if self._gemini_client is None:
            if self._gemini_module is not None:
                genai_mod = self._gemini_module
            else:
                try:
                    from google import genai
                except ImportError as exc:
                    raise LLMConfigurationError(
                        "The 'google-genai' package is not installed. Run: pip install google-genai"
                    ) from exc
                genai_mod = genai

            timeout_ms = int(max(1000, round(self.config.timeout_seconds * 1000)))
            http_options = types_mod.HttpOptions(
                timeout=timeout_ms,
                retry_options=types_mod.HttpRetryOptions(attempts=1),
            )
            self._gemini_client = genai_mod.Client(
                api_key=api_key,
                http_options=http_options,
            )
        return self._gemini_client

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to the configured provider with retries."""
        if not messages:
            raise ValueError("messages must contain at least one message.")

        temp = self.config.temperature if temperature is None else temperature
        tokens = self.config.max_tokens if max_tokens is None else max_tokens

        def _dispatch() -> LLMResponse:
            if self.provider == "groq":
                return self._call_groq(
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                    temperature=temp,
                    max_tokens=tokens,
                )
            return self._call_gemini(
                messages=messages,
                tools=tools,
                response_format=response_format,
                temperature=temp,
                max_tokens=tokens,
            )

        return self._execute_with_retry(_dispatch)

    def generate_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Request a structured JSON object response and parse it reliably."""
        response = self.chat(
            messages,
            response_format="json",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return extract_json_object(response.content)

    def _execute_with_retry(self, fn: Callable[[], LLMResponse]) -> LLMResponse:
        attempts = max(1, self.config.max_retries + 1)
        last_error: LLMError | None = None

        for attempt in range(attempts):
            try:
                return fn()
            except (LLMRateLimitError, LLMTimeoutError, LLMConnectionError) as exc:
                last_error = exc
                # Do not retry daily/exhausted quota limits which cannot succeed
                is_quota_exhausted = isinstance(exc, LLMRateLimitError) and (
                    "quota" in str(exc).lower() or "resource_exhausted" in str(exc).lower()
                )
                if attempt >= attempts - 1 or is_quota_exhausted:
                    logger.error(
                        "LLM call failed after %d attempts (%s): %s",
                        attempt + 1,
                        type(exc).__name__,
                        self._safe_error_message(exc),
                    )
                    raise
                base_delay = min(
                    self.config.max_backoff_seconds,
                    self.config.initial_backoff_seconds * (self.config.backoff_multiplier ** attempt),
                )
                jitter = self._jitter(0.0, max(0.05, base_delay * 0.25))
                delay = min(self.config.max_backoff_seconds, base_delay + jitter)
                logger.warning(
                    "Transient %s on provider=%s (attempt %d/%d); retrying in %.2fs.",
                    type(exc).__name__,
                    self.provider,
                    attempt + 1,
                    attempts,
                    delay,
                )
                self._sleep(delay)

        assert last_error is not None  # pragma: no cover
        raise last_error

    def _call_groq(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        response_format: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        client = self._get_groq_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _sanitize_groq_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        elif response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        try:
            raw = client.chat.completions.create(**payload)
        except Exception as exc:
            raise self._map_groq_exception(exc) from exc

        return self._parse_groq_response(raw)

    def _parse_groq_response(self, raw: Any) -> LLMResponse:
        choices = getattr(raw, "choices", None)
        if not choices:
            raise LLMInvalidResponseError("Groq response contained no choices.")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise LLMInvalidResponseError("Groq response choice contained no message.")

        content = getattr(message, "content", None) or ""
        raw_tool_calls = getattr(message, "tool_calls", None) or []

        parsed_tool_calls: list[ToolCall] = []
        for idx, tc in enumerate(raw_tool_calls):
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            tc_name = getattr(fn, "name", "") or ""
            raw_args = getattr(fn, "arguments", "") or "{}"
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                if not isinstance(parsed_args, dict):
                    raise LLMInvalidResponseError(
                        f"Tool arguments for {tc_name!r} must decode to a JSON object."
                    )
            except json.JSONDecodeError as exc:
                raise LLMInvalidResponseError(
                    f"Malformed JSON arguments in tool call {tc_name!r}: {exc}"
                ) from exc

            parsed_tool_calls.append(
                ToolCall(
                    id=str(getattr(tc, "id", None) or f"call_{idx}_{uuid.uuid4().hex[:6]}"),
                    name=str(tc_name),
                    arguments=parsed_args,
                    raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(parsed_args),
                )
            )

        usage_obj = getattr(raw, "usage", None)
        usage: dict[str, int] = {}
        if usage_obj is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = getattr(usage_obj, key, None)
                if isinstance(val, int):
                    usage[key] = val

        return LLMResponse(
            content=str(content),
            tool_calls=parsed_tool_calls,
            provider="groq",
            model=str(getattr(raw, "model", None) or self.model),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
            raw_response=raw,
        )

    def _call_gemini(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        response_format: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        client = self._get_gemini_client()
        types_mod = self._get_gemini_types()

        system_instruction, contents = _convert_messages_to_gemini(
            messages,
            types_mod,
            thought_signatures=self._thought_signatures,
        )
        gemini_tools = _convert_openai_tools_to_gemini(tools, types_mod)

        cfg_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        if gemini_tools:
            cfg_kwargs["tools"] = gemini_tools
        elif response_format == "json":
            # Gemini API rejects combining response_mime_type='application/json' with tools.
            cfg_kwargs["response_mime_type"] = "application/json"

        config = types_mod.GenerateContentConfig(**cfg_kwargs)

        try:
            raw = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise self._map_gemini_exception(exc) from exc

        return self._parse_gemini_response(raw)

    def _parse_gemini_response(self, raw: Any) -> LLMResponse:
        candidates = getattr(raw, "candidates", None)
        if not candidates:
            raise LLMInvalidResponseError("Gemini response contained no candidates.")

        candidate = candidates[0]
        content_obj = getattr(candidate, "content", None)
        parts = getattr(content_obj, "parts", None) or []

        text_segments: list[str] = []
        parsed_tool_calls: list[ToolCall] = []

        for idx, part in enumerate(parts):
            # Ignore internal Gemini 3.x thinking summary parts when extracting final text.
            if getattr(part, "thought", False):
                continue

            part_text = getattr(part, "text", None)
            if part_text:
                text_segments.append(str(part_text))

            fc = getattr(part, "function_call", None)
            fc_name = getattr(fc, "name", None) if fc is not None else None
            if fc is not None and fc_name:
                raw_args = getattr(fc, "args", None)
                plain_args = _to_plain_python(raw_args) if raw_args is not None else {}
                if not isinstance(plain_args, dict):
                    raise LLMInvalidResponseError(
                        f"Gemini function_call args for {fc_name!r} could not be converted to dict."
                    )
                call_id = str(
                    getattr(fc, "id", None) or f"gemini_call_{idx}_{uuid.uuid4().hex[:6]}"
                )
                raw_sig = getattr(part, "thought_signature", None)
                sig = bytes(raw_sig) if isinstance(raw_sig, (bytes, bytearray)) else None
                if sig is not None:
                    self._thought_signatures[call_id] = sig

                parsed_tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=str(fc_name),
                        arguments=plain_args,
                        raw_arguments=json.dumps(plain_args, ensure_ascii=False),
                        thought_signature=sig,
                    )
                )

        if not text_segments and not parsed_tool_calls:
            finish_reason = getattr(candidate, "finish_reason", None)
            raise LLMInvalidResponseError(
                f"Gemini returned an empty response (finish_reason={finish_reason!r})."
            )

        usage_meta = getattr(raw, "usage_metadata", None)
        usage: dict[str, int] = {}
        if usage_meta is not None:
            for src_key, dst_key in (
                ("prompt_token_count", "prompt_tokens"),
                ("candidates_token_count", "completion_tokens"),
                ("total_token_count", "total_tokens"),
            ):
                val = getattr(usage_meta, src_key, None)
                if isinstance(val, int):
                    usage[dst_key] = val

        finish_reason = getattr(candidate, "finish_reason", None)
        return LLMResponse(
            content="\n".join(text_segments).strip(),
            tool_calls=parsed_tool_calls,
            provider="gemini",
            model=self.model,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            usage=usage,
            raw_response=raw,
        )

    def _map_groq_exception(self, exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc

        msg = self._safe_error_message(exc)
        cls_name = type(exc).__name__
        status_code = getattr(exc, "status_code", None)

        if cls_name in {"AuthenticationError", "PermissionDeniedError"} or status_code in {401, 403}:
            return LLMAuthError(f"Groq authentication failed: {msg}")
        if cls_name == "RateLimitError" or status_code == 429:
            return LLMRateLimitError(f"Groq rate limit exceeded: {msg}")
        if cls_name == "APITimeoutError" or isinstance(exc, TimeoutError) or "timeout" in msg.lower():
            return LLMTimeoutError(f"Groq request timed out: {msg}")
        if cls_name == "APIConnectionError" or isinstance(exc, (ConnectionError, OSError)):
            return LLMConnectionError(f"Groq network connection error: {msg}")
        if isinstance(status_code, int) and status_code >= 500:
            return LLMConnectionError(f"Groq service unavailable (HTTP {status_code}): {msg}")
        return LLMError(f"Groq API error: {msg}")

    def _map_gemini_exception(self, exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc

        msg = self._safe_error_message(exc)
        lower = msg.lower()
        cls_name = type(exc).__name__
        code = getattr(exc, "code", None)

        if (
            cls_name in {"Unauthenticated", "PermissionDenied", "Unauthorized"}
            or code in {401, 403}
            or "api_key_invalid" in lower
            or "api key not valid" in lower
            or "invalid api key" in lower
            or "permission denied" in lower
        ):
            return LLMAuthError(f"Gemini authentication failed: {msg}")
        if (
            cls_name == "ResourceExhausted"
            or code == 429
            or "429" in lower
            or "resource_exhausted" in lower
            or "quota" in lower
        ):
            return LLMRateLimitError(f"Gemini rate limit exceeded: {msg}")
        if (
            cls_name in {
                "DeadlineExceeded",
                "ReadTimeout",
                "ConnectTimeout",
                "WriteTimeout",
                "PoolTimeout",
                "APITimeoutError",
            }
            or code == 504
            or isinstance(exc, TimeoutError)
            or "deadline exceeded" in lower
            or "deadlineexceeded" in lower
            or "deadline_exceeded" in lower
            or "timed out" in lower
            or "timeout" in lower
        ):
            return LLMTimeoutError(f"Gemini request timed out: {msg}")
        if (
            cls_name in {
                "ServiceUnavailable",
                "InternalServerError",
                "ConnectError",
                "NetworkError",
                "RemoteProtocolError",
            }
            or code in {500, 502, 503}
            or isinstance(exc, (ConnectionError, OSError))
            or "connection" in lower
            or "network" in lower
        ):
            return LLMConnectionError(f"Gemini connection error: {msg}")
        return LLMError(f"Gemini API error: {msg}")
