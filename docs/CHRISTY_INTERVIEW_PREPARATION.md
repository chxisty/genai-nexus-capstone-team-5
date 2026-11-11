# Christy's Technical Interview & Viva Preparation Guide

This Q&A guide prepares you to explain every design decision, line of code, and GenAI engineering concept in your Member 1 contribution (`llm.py`, `prompts.py`, `answer.py`, `memory_adapter.py`, `ai_assistant.py`).

---

## Beginner Level Questions

### 1. What was your role as Member 1 in the Railway Passenger Help Desk project?
I owned the **AI Assistant layer** (`helpdesk/llm.py`, `helpdesk/prompts.py`, `helpdesk/answer.py`, `helpdesk/memory_adapter.py`, and `helpdesk/ai_assistant.py`). My code connects the user's question and conversation history to the LLM, lets the LLM decide which retrieval or database tools to call (delegating execution to Neha's tool executor and Krishna's data layer), and enforces strict source-and-SQL grounding before returning the final 4-key response (`answer`, `sources`, `sql`, `dont_know`).

### 2. Why did we build a terminal Python application instead of a web app?
The official GenAI Nexus capstone specification requires a pure terminal Python package evaluated automatically via `ask(question)` and `python run_questions.py <questions> <results>`. Keeping the architecture free of web UI frameworks guarantees clean separation of concerns and direct evaluation compatibility.

### 3. Why are `.env` and `data/` listed in `.gitignore`?
- `.env` contains private API keys (`GEMINI_API_KEY`, `GROQ_API_KEY`). Committing any key—even in Git history—violates security best practices and results in zero repo points under the capstone rules. Instead, we commit `.env.example` with empty placeholders.
- `data/` holds the 13 MB railway data pack (`rail_capstone_pack.zip` contents), which the brief specifies must be unzipped locally and never committed.

### 4. What is the difference between `LLMClient` using Gemini vs. Groq in `helpdesk/llm.py`?
- **Google Gemini** (`google-generativeai`, default model `gemini-2.5-flash`) is the default provider when `LLM_PROVIDER` is unset or `gemini`, guaranteeing that automated evaluation runs using `GEMINI_API_KEY` work out of the box.
- **Groq** (`groq`, default model `llama-3.3-70b-versatile`) is supported via `LLM_PROVIDER=groq` for fast inference during development and testing.
- Both return a single normalized `LLMResponse` dataclass containing `content`, `tool_calls`, `provider`, `model`, `finish_reason`, and `usage`.

### 5. What does `dont_know: true` mean, and when does our code trigger it?
`dont_know: true` signals that the assistant cannot answer reliably from the official data pack. Our code sets it when:
1. No document chunks or database rows support the question (e.g., ticket fares, PNR status, live running status, or waitlist confirmation chances).
2. A user requests an unsafe database mutation (e.g., `UPDATE` / `CHANGE` train times).
3. An LLM or tool error prevents retrieving verified evidence.

---

## Intermediate Level Questions

### 6. How does `helpdesk/llm.py` normalize tool calling across both Groq and Gemini?
Groq uses OpenAI-style schemas (`{"type": "function", "function": {...}}`) and message roles (`system`, `user`, `assistant` with `tool_calls`, `tool` with `tool_call_id`). Gemini's SDK (`google-generativeai`) instead requires:
- `system_instruction` passed separately to `GenerativeModel(...)`.
- Tool schemas converted recursively via `_convert_openai_tools_to_gemini()` into `genai.protos.Tool` and `genai.protos.FunctionDeclaration` with `genai.protos.Schema` / `genai.protos.Type`.
- `assistant` tool requests converted to `Content(role="model", parts=[Part(function_call=...)])` and tool outputs converted to `Content(role="user", parts=[Part(function_response=...)])` via `_convert_messages_to_gemini()`.
- Gemini `MapComposite` protobuf tool arguments converted to plain Python dicts via `_to_plain_python()`.
- Both SDK outputs are parsed into identical `ToolCall(id, name, arguments, raw_arguments)` objects inside `LLMResponse`.

### 7. Why didn't we silently fall back from Groq to Gemini if `GROQ_API_KEY` is missing?
Explicit configuration must never be silently overridden behind the developer's back—if someone configures `LLM_PROVIDER=groq` without `GROQ_API_KEY`, `active_api_key()` immediately raises `LLMAuthError` with instructions to set the key. Only when `LLM_PROVIDER=auto` is set does `resolve_provider()` auto-select Gemini (if `GEMINI_API_KEY` is present) or Groq (if only `GROQ_API_KEY` is present). And when `LLM_PROVIDER` is unset, it defaults cleanly to `"gemini"`.

### 8. How does retry logic work in `LLMClient`, and which errors are NOT retried?
`_execute_with_retry()` uses exponential backoff (`initial_backoff * 2^attempt`, capped at `max_backoff_seconds`) plus bounded random jitter (`jitter_fn`) to avoid thundering-herd spikes:
- **Retried (transient):** `LLMRateLimitError` (429/quota), `LLMTimeoutError` (timeouts/504), `LLMConnectionError` (network errors/5xx).
- **Not retried (deterministic):** `LLMAuthError` (invalid/missing API key — retrying immediately wastes time), `LLMConfigurationError`, and `LLMInvalidResponseError`.
We also inject `sleep_fn` in `__init__` so unit tests verify backoff math instantaneously without waiting in real time.

### 9. Why are prompts in `helpdesk/prompts.py` split into modular constants instead of one giant string?
Modular prompt constants (`ROLE_IDENTITY_PROMPT`, `EVIDENCE_GROUNDING_PROMPT`, `RULE_PRECEDENCE_PROMPT`, `TIMETABLE_SCOPE_PROMPT`, `MISSING_AND_CONFLICT_PROMPT`, `CITATION_INTEGRITY_PROMPT`, `TOOL_USAGE_PROMPT`, `STRUCTURED_ANSWER_JSON_PROMPT`) allow:
1. Composing different system prompts for the tool-calling loop (`build_assistant_system_prompt`) vs. final JSON synthesis (`build_answer_system_prompt`).
2. Toggling tool instructions off when running direct pre-retrieved QA (`include_tools=False`).
3. Easier maintenance, testing, and auditing—ensuring zero sample-question answers leak into system prompts.

### 10. How do prompts handle conflicting Railway Board circulars and the 2016 timetable?
- `RULE_PRECEDENCE_PROMPT` instructs the LLM to inspect document issue dates (`date=...` in `format_document_evidence`) so newer circulars/Gazette/PIB notices supersede older 2015/earlier rules when they conflict, and train-specific rules (such as Vande Bharat Sleeper rules) override general refund rules for that train.
- `TIMETABLE_SCOPE_PROMPT` clarifies that `rail_timetable.db` is a static **2016 snapshot** with no fares, live delays, seat availability, or PNR status.

---

## Advanced Level Questions

### 11. How does `AnswerGenerator` guarantee the LLM cannot hallucinate document names, page numbers, or SQL queries?
Prompting alone cannot eliminate hallucinations, so `helpdesk/answer.py` enforces **deterministic post-generation verification**:
1. **Pre-LLM Gate:** If both `chunks` and `db_results` lack evidence, `AnswerGenerator.generate()` short-circuits and returns `dont_know=True` without even calling the LLM.
2. **Citation Allowlist:** `extract_allowed_sources()` builds the exact set of `(doc, page)` pairs from the retrieved `chunks` and `source_metadata`. In `_validate_citations()`, any source returned by the LLM whose filename or page number is not in that allowlist is **dropped**. HTML documents (`.html`/`.htm`) always normalize `page` to `None`.
3. **SQL Locking:** `AnswerGenerator` ignores any `"sql"` field emitted by the LLM and populates `"sql"` exclusively via `normalize_sql_value(sql)` using the actual SQL queries executed by Krishna/Neha's tool layer.
4. **Refusal Consistency:** `_REFUSAL_PREFIX_RE` catches answers starting with `"I don't know"` or `"I cannot"` even if the LLM forgot to flip `"dont_know": true`, forcing `dont_know=True` and clearing `sources`.

### 12. What happens in `AIAssistant` if an LLM skips calling tools and tries to answer a railway question from pretraining memory?
`_finalize_response()` inspects `_EvidenceAccumulator.has_evidence`. If zero tool evidence (neither document chunks nor DB queries) was collected:
- If the model issued an explicit refusal (e.g., declining an unsafe write query like changing train times), the refusal message is preserved with `dont_know=True`.
- If the model attempted to give a factual answer from pretraining memory without calling tools, `_finalize_response()` **blocks** that unverified reply and returns a grounded `"I don't know. I could not find supporting evidence..."` response (`dont_know=True`).

### 13. Why does `MemoryAdapter` group `assistant(tool_calls)` and `tool` messages into atomic blocks during token truncation?
If naive sliding-window truncation cuts in the middle of a tool turn, it can leave a `role="tool"` message in history without the preceding `role="assistant"` `tool_calls` message that generated its `tool_call_id`. Both Groq (OpenAI chat format) and Gemini reject orphaned tool responses with HTTP 400 errors. `_group_atomic_blocks()` keeps each `assistant(tool_calls)` together with all its corresponding `tool` replies as one indivisible block when enforcing `max_messages` and `max_tokens`.

### 14. Why did you use `MemoryBackendProtocol` and `ToolExecutorProtocol` instead of writing Krishna's SQL code or Prasad's database inside your files?
Respecting team ownership avoids duplicate implementations and merge conflicts. Using Python `Protocol` interfaces and dependency injection (`MemoryAdapter(backend=...)`, `AIAssistant(..., tool_executor=..., tool_schemas=...)`) decouples Member 1's AI orchestration from Member 2, 3, and 4's internal implementations. When Prasad's branch merges, passing his store to `MemoryAdapter` switches seamlessly from `InMemorySessionMock` to production without altering a single line of `ai_assistant.py`.

### 15. How did you test your modules without spending API quota or needing live network access?
Every unit test in `tests/` mocks the external SDK clients (`groq_client`, `gemini_module`) and injects `sleep_fn=lambda _: None`:
- `tests/test_llm.py` verifies Gemini/Groq normalization, tool schema/message translation, auth non-retry, exponential backoff math on 429/timeout/network failures, and secret redaction.
- `tests/test_prompts.py` verifies guardrail rules, evidence formatting (`page=null` for HTML), and absence of hardcoded test answers.
- `tests/test_answer.py` verifies citation allowlist filtering, SQL locking, empty-evidence short-circuiting, and invalid JSON recovery.
- `tests/test_memory_adapter.py` verifies missing-session safety, token budgeting, atomic block truncation, and duck-typed custom backends.
- `tests/test_ai_assistant.py` verifies multi-step hybrid tool calling (doc + SQL), tool error recovery, ungrounded-reply blocking, and multi-turn follow-up context.

### 16. What subtle Gemini SDK constraint did you handle in `LLMClient._call_gemini`?
Two important real-world Gemini SDK details:
1. `google-generativeai` raises a `400 InvalidArgument` error if `response_mime_type="application/json"` is combined with `tools` (function declarations). So `LLMClient` only sets `response_mime_type="application/json"` when `response_format == "json"` and `tools` is empty—while `AIAssistant` uses tools in the reasoning loop and then runs JSON synthesis without tools in `AnswerGenerator`.
2. Gemini requires `FunctionResponse.response` to be a dictionary (`Struct`), not a bare list or string, so `_convert_messages_to_gemini()` wraps non-dict tool results as `{"result": ...}`.
