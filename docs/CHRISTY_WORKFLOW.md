# Christy's Workflow & Architecture Guide — Railway Passenger Help Desk

This document traces the complete execution lifecycle, data flow, tool-calling loop, multi-turn memory integration, error recovery, and cross-team interfaces for Member 1's AI Assistant module.

---

## 1. High-Level Architecture Diagram

```mermaid
flowchart TD
    User["Passenger / CLI ask(question)"] --> Pipeline["Member 5: Ram (helpdesk/pipeline.py)"]
    Pipeline --> Assistant["Member 1: AIAssistant (helpdesk/ai_assistant.py)"]
    Assistant --> Memory["Member 1: MemoryAdapter (helpdesk/memory_adapter.py)"]
    Memory --> Prasad["Member 4: Prasad Session Store"]
    Assistant --> Prompts["Member 1: Prompt Builders (helpdesk/prompts.py)"]
    Assistant --> LLM["Member 1: LLMClient (helpdesk/llm.py)"]
    LLM -->|Default| Gemini["Google Gemini API"]
    LLM -->|Configurable| Groq["Groq API"]
    LLM -->|Tool Call| Assistant
    Assistant --> Neha["Member 3: Neha Tool Executor"]
    Neha --> Search["Member 2: Krishna Document Search"]
    Neha --> SQL["Member 2: Krishna Read-Only SQLite"]
    Search --> Evidence["Evidence Accumulator (chunks, sources)"]
    SQL --> Evidence2["Evidence Accumulator (rows, sql)"]
    Evidence --> AnswerGen["Member 1: AnswerGenerator (helpdesk/answer.py)"]
    Evidence2 --> AnswerGen
    AnswerGen --> Output["Verified JSON: answer, sources, sql, dont_know"]
```

---

## 2. End-to-End Execution Data Flow

1. **Entry (`AIAssistant.ask` / `process_question`)**:
   - Validates that `question` is non-empty.
   - Calls `MemoryAdapter.ensure_session(session_id)` to resolve or initialize a session ID.
   - Fetches prior context via `MemoryAdapter.get_context_messages(sid)` within message-count and token budgets.
   - Records the current passenger message with `MemoryAdapter.record_user_message(...)`.
2. **Prompt Construction (`helpdesk/prompts.py`)**:
   - Composes the system prompt from modular components (`ROLE_IDENTITY_PROMPT`, `EVIDENCE_GROUNDING_PROMPT`, `RULE_PRECEDENCE_PROMPT`, `TIMETABLE_SCOPE_PROMPT`, `MISSING_AND_CONFLICT_PROMPT`, `CITATION_INTEGRITY_PROMPT`, `TOOL_USAGE_PROMPT`).
   - Appends bounded prior conversation turns and the current passenger question.
3. **Multi-Step Tool-Calling Loop (`helpdesk/ai_assistant.py` + `helpdesk/llm.py`)**:
   - Calls `LLMClient.chat(messages, tools=tool_schemas)` up to `max_tool_steps` iterations.
   - `LLMClient` translates OpenAI-style schemas and `tool` history into `google.genai.types` (`Tool`, `FunctionDeclaration`, `Content`, `Part` with Gemini 3.x `thought_signature` preservation) when using Gemini, or passes them directly when using Groq.
   - If the LLM emits `ToolCall` requests, `AIAssistant._invoke_tool` executes each tool safely via Neha's `tool_executor` (which routes to Krishna's document search or read-only SQLite DB).
   - `_collect_evidence_from_tool_result` captures normalized document chunks, source `(doc, page)` pairs, DB rows, and executed `SELECT` SQL strings.
   - Tool calls and tool results are recorded in `MemoryAdapter` and fed back into `messages` so the LLM can either perform additional lookups (e.g., hybrid schedule + refund question) or complete reasoning.
4. **Grounded Answer Synthesis & Validation (`helpdesk/answer.py`)**:
   - If zero usable evidence was retrieved:
     - Explicit refusal text (e.g., refusing a database modification request) is preserved with `dont_know=True`.
     - Any ungrounded factual claim made without evidence is blocked and replaced with `"I don't know..."` (`dont_know=True`, `sources=[]`, `sql=None`).
   - If evidence exists:
     - `AnswerGenerator` builds grounded messages (or validates direct JSON) and filters citations strictly against the retrieved `(doc, page)` allowlist.
     - HTML documents always normalize `page` to `None`.
     - `sql` is locked to the actual executed SQL string(s), never LLM-generated SQL.
5. **Memory Persistence & Return**:
   - Records the final assistant answer and grounding metadata in `MemoryAdapter`.
   - Returns `{"answer": ..., "sources": ..., "sql": ..., "dont_know": ...}`.

---

## 3. Function-Call & Hybrid Tool-Calling Sequence

```mermaid
sequenceDiagram
    participant U as Ram (pipeline.py)
    participant A as AIAssistant
    participant M as MemoryAdapter
    participant L as LLMClient
    participant T as Neha ToolExecutor
    participant G as AnswerGenerator

    U->>A: ask("When does 12002 leave NDLS and what is the 5-day AC CC fee?", session_id)
    A->>M: ensure_session(session_id) & get_context_messages(sid)
    M-->>A: prior context turns
    A->>M: record_user_message(sid, question)
    A->>L: chat(messages, tools=tool_schemas)
    L-->>A: ToolCall("query_timetable", {"sql": "SELECT departure ..."})
    A->>T: execute("query_timetable", ...)
    T-->>A: {"ok": true, "sql": "SELECT ...", "rows": [{"departure": "06:15"}]}
    A->>M: record_tool_calls & record_tool_result
    A->>L: chat(messages + tool result, tools=tool_schemas)
    L-->>A: ToolCall("search_documents", {"query": "AC Chair Car cancellation"})
    A->>T: execute("search_documents", ...)
    T-->>A: {"ok": true, "chunks": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1, "text": "..."}]}
    A->>L: chat(messages + second tool result)
    L-->>A: Draft plain response
    A->>G: generate(question, chunks, db_results, sql, context)
    G->>L: generate_json(grounded_messages)
    L-->>G: JSON {"answer": "...", "sources": [...], "dont_know": false}
    G-->>A: Verified 4-key dict
    A->>M: record_assistant_message(sid, answer, metadata)
    A-->>U: {"answer": "...", "sources": [...], "sql": "SELECT ...", "dont_know": false}
```

---

## 4. Multi-Turn Conversation Sequence

```mermaid
sequenceDiagram
    participant P as Passenger
    participant A as AIAssistant
    participant M as MemoryAdapter (Prasad Backend)
    participant L as LLMClient

    P->>A: Turn 1: "Do I need Aadhaar for Tatkal?" (session="s1")
    A->>M: get_context_messages("s1") -> []
    A->>M: record_user_message("s1", ...)
    A->>L: Tool loop -> search_documents("Tatkal Aadhaar")
    L-->>A: Grounded answer citing rb_cc08_2025_tatkal.pdf
    A->>M: record_assistant_message("s1", answer)
    A-->>P: Reply 1

    P->>A: Turn 2: "Which circular mentions that?" (session="s1")
    A->>M: get_context_messages("s1", max_messages=12, max_tokens=2500)
    M-->>A: [Turn 1 user + Turn 1 assistant]
    A->>M: record_user_message("s1", "Which circular mentions that?")
    A->>L: Messages include Turn 1 context + Turn 2 question
    L-->>A: Resolves "that" -> Tatkal Aadhaar rule -> verified answer
    A->>M: record_assistant_message("s1", reply_2)
    A-->>P: Reply 2
```

---

## 5. Error-Handling & Retry Sequence

```mermaid
flowchart TD
    Call["LLMClient.chat()"] --> TryReq["Send request to Gemini or Groq"]
    TryReq -->|Success| Parse["Normalize LLMResponse & ToolCall list"]
    TryReq -->|401 / 403 / Missing Key| AuthErr["Raise LLMAuthError immediately (no retry)"]
    TryReq -->|429 / Timeout / Network 5xx| Transient{"Retries remaining?"}
    Transient -->|Yes| Backoff["Sleep exponential backoff + jitter (no key logged)"]
    Backoff --> TryReq
    Transient -->|No| RaiseErr["Raise LLMRateLimitError / LLMTimeoutError / LLMConnectionError"]
    AuthErr --> SafeWrap["AIAssistant / AnswerGenerator catches LLMError"]
    RaiseErr --> SafeWrap
    SafeWrap --> Fallback["Return safe 4-key dict with dont_know=true and redacted log"]

    ToolExec["ToolExecutor.execute()"] -->|Raises Exception| CatchTool["Catch exception in _invoke_tool()"]
    CatchTool --> ToolPayload["Return ok=false, error=... to LLM"]
    ToolPayload --> NextDecision{"Did other tools supply evidence?"}
    NextDecision -->|No| DontKnow["Return dont_know=true safely"]
    NextDecision -->|Yes| Grounded["Synthesize from valid evidence only"]
```

---

## 6. Integration Summary Across All Five Members

| Member | Responsibility | How Christy's Module Connects |
| :--- | :--- | :--- |
| **1. Christy (Me)** | LLM client, prompts, grounded answers, AI tool loop, memory adapter | Owns `llm.py`, `prompts.py`, `answer.py`, `memory_adapter.py`, `ai_assistant.py`. |
| **2. Krishna** | SQLite DB (`rail_timetable.db`), station/train queries, document search | Provides chunks (`doc`, `page`, `date`, `text`) and read-only SQL outputs (`sql`, `rows`) surfaced via tools. |
| **3. Neha** | Tool schemas, argument validation, query routing, tool execution | Supplies OpenAI-style `tool_schemas` and `ToolRegistry.call(name, args)` (or `.execute`) injected into `AIAssistant`. |
| **4. Prasad** | Session store, conversation storage, response reliability | Supplies session store matching `MemoryBackendProtocol` injected into `MemoryAdapter(backend=...)`. |
| **5. Ram** | `pipeline.py`, `ask(question)`, CLI runners, evaluation, main README | Instantiates `AIAssistant` with all team components and delegates `ask(question)` calls to it. |

---

## 7. Step-by-Step Technical Evaluation Walkthrough

When demonstrating Member 1 during the live presentation or viva, walk through these 5 steps:

1. **Multi-Provider Initialization (`helpdesk/llm.py`):** Show `.env.example` and explain how `LLMConfig.from_env()` defaults to Gemini (`gemini-3.5-flash` via `google-genai`) for capstone evaluation while allowing instant switching to Groq (`openai/gpt-oss-120b`) via `LLM_PROVIDER=groq`.
2. **Modular System Prompts (`helpdesk/prompts.py`):** Point out how `build_assistant_system_prompt()` composes evidence-only grounding, circular date precedence, and the 2016 snapshot scope without embedding any test answers.
3. **Tool-Calling Loop (`helpdesk/ai_assistant.py`):** Explain how `AIAssistant.process_question()` runs up to `max_tool_steps`, dispatches calls to Neha's `ToolRegistry.call()`, unwraps `{"status": "success", "result": ...}`, and collects evidence across multiple tools.
4. **Memory Budgeting (`helpdesk/memory_adapter.py`):** Explain how `MemoryAdapter` connects to Prasad's session store via `MemoryBackendProtocol`, enforces `max_messages` and `max_tokens`, and groups `assistant(tool_calls)` + `tool` pairs atomically.
5. **Grounded Validation (`helpdesk/answer.py`) & Verification (`pytest`):** Show how `AnswerGenerator` validates `(doc, page)` citations against an allowlist, normalizes HTML pages to `null`, locks `sql` to real executed SQL only, and run `.\.venv\Scripts\python.exe -m pytest -v` to demonstrate all 50 unit tests passing.

