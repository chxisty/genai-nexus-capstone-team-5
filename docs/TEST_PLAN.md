# Test Plan: Railway Passenger Help Desk

**GenAI Nexus Fellowship · Module 2 Capstone**  
**Team 5 (Neuroverse)**  
**Integration Lead:** Ram Narisetti (Member 5)

---

## 1. Overview & Verification Objectives

The primary objective of this test plan is to establish a rigorous, repeatable automated testing strategy for the integrated Railway Passenger Help Desk application. 

Key testing guarantees:
1. **Zero External Dependency in CI:** All automated tests must execute without requiring live API keys, active internet access, or external cloud services.
2. **Defensive Integration Coverage:** Verify that components authored by different team members interoperate seamlessly across protocol boundaries.
3. **Strict Contract Conformance:** Ensure every response adheres to `{ "answer": str, "sources": list, "sql": str | None, "dont_know": bool }`.
4. **Data Safety:** Guarantee that the SQLite database cannot be mutated or corrupted through the help desk tools.

---

## 2. Test Architecture & Directory Structure

```text
tests/
├── conftest.py                     # Shared pytest fixtures
├── test_ai_assistant.py            # AI assistant orchestration & tool loops (Christy)
├── test_llm.py                     # LLM client, retries, secrets, and SDK translation (Christy)
├── test_answer.py                  # Answer grounding & citation allowlist validation (Christy)
├── test_prompts.py                 # Grounded prompt construction (Christy)
├── test_memory_adapter.py          # MemoryAdapter protocol & token budgeting (Christy)
├── test_tools.py                   # ToolRegistry, validation, and mock tools (Neha)
├── test_database_tools.py          # Real SQLite database tools (Krishna)
├── test_session_manager.py         # SQLite persistent session manager (Prasad)
├── test_context_handler.py         # Follow-up analysis & pronoun resolution (Prasad)
├── test_reliability.py             # Response contract validator (Prasad)
├── test_pipeline_integration.py    # End-to-end integration test suite (Ram)
└── test_cli.py                     # Command-line interface and REPL tests (Ram)
```

---

## 3. Test Suites & Verification Scope

### Suite 1: AI Assistant & LLM Client (`tests/test_ai_assistant.py`, `tests/test_llm.py`)
- **Focus:** Multi-step function calling loop, Gemini 3.x `thought_signature` preservation, Groq tool-call translation, secret redaction, and retry backoff.
- **Mocking:** Uses deterministic mock LLM clients and mock tool executors.

### Suite 2: Tool Registry & Schema Validation (`tests/test_tools.py`)
- **Focus:** Argument type checking, required parameter enforcement, non-empty string checks, unknown argument rejection, and error envelope formatting.

### Suite 3: Database Tools & SQLite Integrity (`tests/test_database_tools.py`)
- **Focus:** Real queries against `data/rail_timetable.db` using `search_stations`, `search_trains`, `get_train_details`, and `get_train_route`.
- **Safety:** Verifies read-only connection behavior and handling of invalid station/train codes.

### Suite 4: Context & Session Persistence (`tests/test_session_manager.py`, `tests/test_context_handler.py`, `tests/test_reliability.py`)
- **Focus:** SQLite session creation, message serialization, multi-turn history ordering, follow-up detection regex patterns, and 4-field schema validation.

### Suite 5: End-to-End Pipeline Integration (`tests/test_pipeline_integration.py`)
- **Focus:** Full pipeline wiring from `ask()` through `ReliableHelpDesk` -> `ContextHandler` -> `AIAssistant` -> `MemoryAdapter` -> `SessionManager` -> `ToolRegistry` -> `database_tools`.
- **Scenarios Tested:**
  - Application & database initialization.
  - Station search, train search, details, route lookups on real database.
  - Multi-turn session sharing and persistence across pipeline re-instantiations.
  - Contextual follow-up resolution without repeating train numbers.
  - Graceful refusal for unanswerable/out-of-scope questions.
  - Handling of missing API keys and upstream HTTP errors without crashes.

### Suite 6: CLI & User Experience (`tests/test_cli.py`)
- **Focus:** Single-question CLI arguments, interactive REPL loop, `/help`, `/new`, `/history`, and `/exit` commands, and terminal output formatting.

---

## 4. Execution Commands

```bash
# Run entire test suite
python -m pytest -q

# Run with verbose output
python -m pytest -v

# Run only integration tests
python -m pytest tests/test_pipeline_integration.py tests/test_cli.py -v
```
