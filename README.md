# Railway Passenger Help Desk — Tool Calling Module

**Author:** Member 3 (Tool Calling & Query Processing)
**Branch:** `feature/tool-calling`

## Overview

This module provides a **tool-calling and query-processing layer** for the
Railway Passenger Help Desk GenAI capstone.  It lets an LLM (via Groq/OpenAI
function-calling) invoke structured railway-data tools with full argument
validation, error handling, and structured responses.

## File Structure

```
tools/
├── __init__.py          # Package exports
├── schemas.py           # JSON tool schemas (OpenAI/Groq format)
├── registry.py          # ToolRegistry — register, validate, dispatch
└── railway_tools.py     # Mock railway functions + register helper

tests/
└── test_tools.py        # Unit tests (20 test cases)

demo_tools.py            # Runnable demonstration script
```

## Quick Start

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run the demo
python demo_tools.py
```

## Integration Guide

### For Christy (Member 1) — LLM / Groq Integration

**Step 1:** Pass the schemas to your LLM so it knows which tools exist:

```python
from tools import TOOL_SCHEMAS

# Send schemas to Groq / OpenAI as the `tools` parameter
response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=TOOL_SCHEMAS,        # ← plug in directly
)
```

**Step 2:** When the LLM returns a tool call, dispatch it through the registry:

```python
from tools import ToolRegistry, register_railway_tools

registry = ToolRegistry()
register_railway_tools(registry)   # loads mock tools (or Krishna's real ones)

# Extract tool call from LLM response
tool_call = response.choices[0].message.tool_calls[0]
tool_name = tool_call.function.name          # e.g. "station_search"
arguments = tool_call.function.arguments     # JSON string from LLM

# Dispatch — handles validation, execution, and errors
result = registry.call(tool_name, arguments)
# result = {"status": "success", "tool": "station_search", "result": {...}}
```

**Step 3:** Feed the result back to the LLM as a tool response message.

### For Krishna (Member 2) — Database Integration

Replace mock functions with real database queries — **no registry changes needed:**

```python
from tools import ToolRegistry

registry = ToolRegistry()

# Register your real functions (same signature as the mocks)
def real_station_search(query: str) -> dict:
    # ... query SQLite database ...
    return {"query": query, "matches": rows}

registry.register("station_search", real_station_search)
registry.register("train_search",   real_train_search)
registry.register("train_details",  real_train_details)
registry.register("route_lookup",   real_route_lookup)
```

Each function must:
- Accept the arguments defined in `schemas.py` as keyword arguments
- Return a JSON-serialisable `dict`

## Features

| Feature                        | Details                                    |
|--------------------------------|--------------------------------------------|
| Schema format                  | OpenAI / Groq function-calling compatible  |
| Argument validation            | Required fields, types, empty strings, unknown fields |
| JSON-string arguments          | Parses LLM-returned JSON strings automatically |
| Structured responses           | `{"status": "success"/"error", "tool": ..., "result"/"error": ...}` |
| Exception safety               | Catches and wraps function exceptions      |
| Swap-in real implementations   | `registry.register(name, func)` — no code changes |
| No external dependencies       | Python standard library only               |
