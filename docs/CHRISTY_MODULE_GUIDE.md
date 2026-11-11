# Christy's Module Guide — AI Assistant Lead (Member 1)

**Owner:** Christy Marrya George (`feature/christy-ai-assistant`)  
**Scope:** LLM Integration, System Prompts, Grounded Answer Generation, AI Tool Calling, and Memory Adapter.

---

## 1. Module Overview & File Inventory

| File | Responsibility | Status |
| :--- | :--- | :--- |
| `helpdesk/__init__.py` | Clean package exports for Member 1's classes and exceptions | Implemented |
| `helpdesk/llm.py` | Multi-provider LLM client (Gemini default + Groq), retries, error mapping, secret redaction | Implemented |
| `helpdesk/prompts.py` | Reusable prompt components, date/train precedence rules, evidence formatting | Implemented |
| `helpdesk/answer.py` | Grounded 4-key response generation, `(doc, page)` allowlist validation, SQL locking | Implemented |
| `helpdesk/memory_adapter.py` | Adapter for Member 4 (Prasad)'s session store + token-aware truncation + mock stub | Implemented |
| `helpdesk/ai_assistant.py` | Multi-step tool-calling orchestration connecting LLM, Neha's tools, and memory | Implemented |
| `.env.example` / `.gitignore` | Environment variable template and secret/dataset Git safety rules | Implemented |
| `requirements.txt` | Core Python dependencies for Member 1's modules and `pytest` | Implemented |

### Dependencies (`requirements.txt`)
- `google-generativeai>=0.8.6` — Official Google Gemini SDK (default capstone provider).
- `groq>=1.7.0` — Official Groq SDK (preferred fast-inference development provider).
- `python-dotenv>=1.0.0` — Loads local `.env` configuration without hardcoding keys.
- `pydantic>=2.0.0` — Data validation dependency aligned with the project stack.
- `pytest>=8.0.0` — Unit testing framework with mocked external API clients.

---

## 2. Detailed Module Reference

### 2.1 `helpdesk/llm.py` — Multi-Provider LLM Integration

#### Exception Hierarchy
- `LLMError(Exception)`: Base error for all client failures.
- `LLMConfigurationError(LLMError)`: Unsupported `LLM_PROVIDER`, invalid env types, missing SDK package, or malformed tool definitions.
- `LLMAuthError(LLMError)`: Missing or invalid `GEMINI_API_KEY` / `GROQ_API_KEY` (HTTP 401/403). **Never retried.**
- `LLMRateLimitError(LLMError)`: HTTP 429 / quota / `ResourceExhausted`. **Retried with backoff.**
- `LLMTimeoutError(LLMError)`: Request timeout / `DeadlineExceeded` / HTTP 504. **Retried with backoff.**
- `LLMConnectionError(LLMError)`: Network/DNS/transport failure or provider 5xx (`ServiceUnavailable`). **Retried with backoff.**
- `LLMInvalidResponseError(LLMError)`: Empty candidates/choices, malformed tool JSON arguments, or unparseable JSON output.

#### Data Classes
- `ToolCall(id: str, name: str, arguments: dict[str, Any], raw_arguments: str = "")` — Normalized tool invocation regardless of whether Gemini or Groq emitted it.
- `LLMResponse(content: str, tool_calls: list[ToolCall], provider: str, model: str, finish_reason: str | None, usage: dict[str, int], raw_response: Any)` — Normalized response object with `@property has_tool_calls -> bool`.
- `LLMConfig`:
  - `provider: str = "gemini"` (`"gemini"`, `"groq"`, or `"auto"`).
  - `gemini_api_key: str`, `gemini_model: str = "gemini-2.5-flash"`.
  - `groq_api_key: str`, `groq_model: str = "llama-3.3-70b-versatile"`.
  - `timeout_seconds: float = 30.0`, `max_retries: int = 3`, `initial_backoff_seconds: float = 1.0`, `backoff_multiplier: float = 2.0`, `max_backoff_seconds: float = 16.0`, `temperature: float = 0.1`, `max_tokens: int = 2048`.
  - `__repr__()` masks API keys as `<set>` or `<unset>`.
  - `from_env(env_file: str | Path | None = None, override: bool = False) -> LLMConfig`: Loads `.env` and builds config.
  - `resolve_provider() -> str`: Enforces explicit `"gemini"` or `"groq"` without silently switching; in `"auto"` mode, prefers `"gemini"` if `GEMINI_API_KEY` is present, else `"groq"` if `GROQ_API_KEY` is present, else `"gemini"`.
  - `active_model(resolved_provider: str | None = None) -> str`
  - `active_api_key(resolved_provider: str | None = None) -> str`
  - `known_secrets() -> tuple[str, ...]`

#### Utility Functions
- `redact_secrets(text: str, extra_secrets: Sequence[str] = ()) -> str`: Replaces configured keys, `AIza...`, `gsk_...`, `api_key=...`, and `Bearer ...` tokens with `[REDACTED]`.
- `extract_json_object(text: str) -> dict[str, Any]`: Extracts JSON cleanly even when wrapped in markdown ` ```json ` fences, raising `LLMInvalidResponseError` if no valid JSON object exists.

#### `LLMClient`
- `__init__(config: LLMConfig | None = None, *, provider=None, model=None, api_key=None, groq_client=None, gemini_module=None, sleep_fn=time.sleep, jitter_fn=random.uniform)`
  - Copies injected `config` before applying overrides so caller instances are not mutated.
  - Supports dependency injection of mock SDK clients and deterministic `sleep_fn` for fast unit tests.
- `chat(messages, *, tools=None, response_format=None, temperature=None, max_tokens=None) -> LLMResponse`:
  - Dispatches to `_call_groq` or `_call_gemini` wrapped in `_execute_with_retry`.
  - Translates OpenAI-style `tools` and `tool` / `assistant(tool_calls)` messages into Gemini `protos.Tool`, `protos.FunctionDeclaration`, `protos.FunctionCall`, and `protos.FunctionResponse` when using Gemini.
  - Only applies `response_mime_type="application/json"` on Gemini when `tools` are absent (avoiding the Gemini API restriction on combining JSON MIME mode with function calling).
- `generate_json(messages, *, temperature=None, max_tokens=None) -> dict[str, Any]`:
  - Calls `chat(..., response_format="json")` and parses via `extract_json_object`.

---

### 2.2 `helpdesk/prompts.py` — Modular System Prompts

Instead of one monolithic prompt, reusable constants represent distinct policy rules:
- `ROLE_IDENTITY_PROMPT`: Establishes Indian Railways Passenger Help Desk identity.
- `EVIDENCE_GROUNDING_PROMPT`: Restricts claims strictly to supplied excerpts and DB results.
- `RULE_PRECEDENCE_PROMPT`: Instructs the model that newer Railway Board circulars/Gazette/PIB releases supersede older rules, and train-specific rules override general rules.
- `TIMETABLE_SCOPE_PROMPT`: Distinguishes the static **2016 SQLite snapshot** from live train running status, fares, PNR, seat availability, and waitlist predictions.
- `MISSING_AND_CONFLICT_PROMPT`: Enforces `"I don't know"` (`dont_know=true`) for missing/unsupported/mutating requests, and clarification/disambiguation on ambiguous queries.
- `CITATION_INTEGRITY_PROMPT`: Enforces exact `doc` filenames and integer PDF pages (`null` for HTML).
- `TOOL_USAGE_PROMPT`: Guides selection between document search, timetable SQL, or both.
- `STRUCTURED_ANSWER_JSON_PROMPT`: Specifies the JSON shape (`answer`, `sources`, `dont_know`).

#### Builder & Formatter Functions
- `build_assistant_system_prompt(*, include_tools: bool = True, extra_instructions: str | None = None) -> str`
- `build_answer_system_prompt(*, strict_json: bool = True) -> str`
- `format_document_evidence(chunks: Sequence[Mapping[str, Any]]) -> str`
- `format_database_evidence(db_results: Any, sql_queries: Sequence[str] | str | None = None) -> str`
- `format_conversation_context(context_messages: Sequence[Mapping[str, Any]] | None) -> str`
- `build_grounded_answer_messages(*, question, chunks=(), db_results=None, sql=None, context_messages=None) -> list[dict[str, str]]`

*(Important: No sample question answers or hidden test hints appear anywhere in `prompts.py`.)*

---

### 2.3 `helpdesk/answer.py` — Grounded Answer Generation

#### Classes & Helpers
- `SourceCitation(doc: str, page: int | None = None)` -> `.to_dict()` returns `{"doc": doc, "page": page}`.
- `GroundedAnswer(answer: str, sources: list[SourceCitation], sql: str | None = None, dont_know: bool = False)` -> `.to_dict()` returns the exact 4-key capstone schema.
- `normalize_sql_value(sql: str | Sequence[str] | None) -> str | None`: Deduplicates and normalizes executed SQL statements.
- `extract_allowed_sources(chunks=(), source_metadata=None) -> list[SourceCitation]`: Extracts an allowlist of `(doc, page)` tuples directly from retrieved evidence, forcing `.html`/`.htm` pages to `None`.
- `AnswerGenerator(llm_client: LLMClient, *, raise_on_error: bool = False)`:
  - `generate(*, question, chunks=(), db_results=None, source_metadata=None, sql=None, conversation_context=None) -> dict[str, Any]`:
    1. Immediately returns `dont_know=True` without calling the LLM if `question` is empty or if both `chunks` and `db_results` provide no usable evidence.
    2. Calls `llm_client.generate_json(...)` with `build_grounded_answer_messages(...)`.
    3. Passes output to `validate_llm_payload(...)`:
       - Drops any citation whose `(doc, page)` does not strictly match `allowed_sources`.
       - Forces `sql` to come exclusively from `validated_sql` (never from LLM output).
       - Detects refusal prefixes (`"I don't know..."`, `"I cannot..."`, `"I can't..."`) to enforce `dont_know=True` and `sources=[]`.
       - Catches `LLMError`/`LLMInvalidResponseError` safely unless `raise_on_error=True`.

---

### 2.4 `helpdesk/memory_adapter.py` — Conversation Memory Integration

#### Contracts & Mock Stub
- `MemoryBackendProtocol`: Runtime-checkable `Protocol` defining `create_session`, `has_session`, `get_history`, `add_message`, and `clear_session`.
- `InMemorySessionMock`: Ephemeral in-memory fallback used **only** when Member 4 (Prasad)'s backend is not passed (`backend=None`).

#### `MemoryAdapter`
- `__init__(backend=None, *, default_max_messages=12, default_max_tokens=2500, token_counter=default_token_estimator)`
- Duck-types common backend method aliases (`get_history`/`get_messages`/`load_history`, `add_message`/`append_message`/`save_message`, `has_session`/`session_exists`/`get_session`, `clear_session`/`reset_session`/`delete_session`).
- `ensure_session(session_id=None) -> str`
- `has_session(session_id: str) -> bool`
- `get_raw_history(session_id: str | None) -> list[dict[str, Any]]` (returns `[]` on missing session)
- `record_user_message(session_id, content, metadata=None) -> str`
- `record_assistant_message(session_id, content, metadata=None) -> str`
- `record_tool_calls(session_id, tool_calls, content="") -> str`
- `record_tool_result(session_id, *, tool_call_id, tool_name, result) -> str`
- `get_tool_history(session_id) -> list[dict[str, Any]]`
- `get_context_messages(session_id, *, max_messages=None, max_tokens=None, include_tool_history=False) -> list[dict[str, Any]]`:
  - Applies message count and token budgets from most recent backward.
  - Groups `assistant(tool_calls)` + consecutive `tool` messages atomically so truncation never strands an orphan tool message.
- `clear_session(session_id) -> None`

---

### 2.5 `helpdesk/ai_assistant.py` — AI Assistant Orchestrator

#### `ToolExecutorProtocol` & `AIAssistant`
- `__init__(llm_client, *, tool_executor=None, tool_schemas=None, memory_adapter=None, answer_generator=None, max_tool_steps=5, extra_system_instructions=None)`
- `ask(question: str, *, session_id: str | None = None) -> dict[str, Any]` (alias for `process_question`)
- `process_question(question: str, *, session_id: str | None = None) -> dict[str, Any]`:
  1. Validates non-empty question.
  2. Ensures session and fetches bounded prior context via `memory_adapter` before storing the new user turn.
  3. Runs up to `max_tool_steps` LLM iterations with Neha's `tool_schemas` and `tool_executor`.
  4. Accumulates verified `chunks`, `source_metadata`, `db_results`, and `sql_queries` in `_EvidenceAccumulator`.
  5. Catches tool execution exceptions and passes structured `{"ok": False, "error": ...}` back to the LLM without crashing.
  6. Finalizes response:
     - Blocks ungrounded factual replies if zero evidence was collected (`dont_know=True`), while preserving polite refusal explanations (e.g., declining write SQL).
     - Validates direct JSON replies or synthesizes via `AnswerGenerator.generate(...)`.
  7. Records assistant reply + grounding metadata in `memory_adapter`.
- `answer_with_evidence(*, question, chunks=(), db_results=None, source_metadata=None, sql=None, session_id=None) -> dict[str, Any]`: Convenience path when retrieval happens upstream.

---

## 3. Team Integration Contracts

1. **Member 2 (Krishna — `helpdesk/sql.py`, `helpdesk/search.py`):**
   - Document chunks should provide `doc` (or `source`/`filename`), `page` (`int` for PDF, `None` for HTML), `text` (or `content`), and optional `date`.
   - SQL results should provide `sql` (the read-only `SELECT` statement actually executed) and `rows` (or `results`/`data`).
2. **Member 3 (Neha — Tool Schemas & Executor):**
   - Provide OpenAI-compatible `tool_schemas` (`[{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]`).
   - Provide `tool_executor` implementing `.execute(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]` (or a callable `(tool_name, arguments) -> dict`).
3. **Member 4 (Prasad — Session/Memory Store):**
   - Provide a store implementing `MemoryBackendProtocol` (`create_session`, `has_session`, `get_history`, `add_message`, `clear_session`), passed to `MemoryAdapter(backend=prasad_store)`.
4. **Member 5 (Ram — `helpdesk/pipeline.py`, `ask.py`, `run_questions.py`):**
   - Instantiate `LLMClient`, wire Neha's `tool_executor` + `tool_schemas` and Prasad's `MemoryAdapter(backend=...)` into `AIAssistant`, and delegate `ask(question)` directly to `assistant.ask(question)`.
