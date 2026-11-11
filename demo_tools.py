#!/usr/bin/env python3
"""
demo_tools.py — Interactive demonstration of the tool-calling module.

Run:  python demo_tools.py

Shows how an LLM's tool-call output (tool name + JSON arguments) flows
through validation and dispatch, and prints structured responses.
"""

import json
from tools import TOOL_SCHEMAS, ToolRegistry, register_railway_tools


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def demo_call(registry, tool_name, arguments, label="") -> None:
    """Call a tool and pretty-print the response."""
    if label:
        print(f"\n--- {label} ---")
    print(f"  Tool : {tool_name}")
    print(f"  Args : {arguments}")
    result = registry.call(tool_name, arguments)
    print(f"  Response:")
    print(json.dumps(result, indent=4))


def main():
    # ---- Setup ----------------------------------------------------------
    registry = ToolRegistry()
    register_railway_tools(registry)

    # ---- 1. Show available schemas (what the LLM sees) ------------------
    print_section("1. Tool Schemas (sent to LLM)")
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        print(f"  • {fn['name']:20s} — {fn['description']}")

    print(f"\n  Registered tools: {registry.list_tools()}")

    # ---- 2. Successful calls (dict arguments) ---------------------------
    print_section("2. Successful Tool Calls")

    demo_call(registry, "station_search",
              {"query": "Delhi"},
              "Search stations matching 'Delhi'")

    demo_call(registry, "train_search",
              {"source": "NDLS", "destination": "BCT", "date": "2026-12-01"},
              "Search trains NDLS -> BCT")

    demo_call(registry, "train_details",
              {"train_number": "12345"},
              "Get details for train 12345")

    demo_call(registry, "route_lookup",
              {"train_number": "12345"},
              "Route for train 12345")

    # ---- 3. JSON-string arguments (as LLM returns them) -----------------
    print_section("3. JSON-String Arguments (LLM-style)")

    json_args = json.dumps({"query": "Mumbai"})
    demo_call(registry, "station_search", json_args,
              "station_search with JSON string")

    # ---- 4. Validation errors -------------------------------------------
    print_section("4. Validation Errors")

    demo_call(registry, "station_search", {},
              "Missing required argument")

    demo_call(registry, "station_search", {"query": ""},
              "Empty-string argument")

    demo_call(registry, "station_search", {"query": 42},
              "Wrong type (int instead of string)")

    demo_call(registry, "station_search", {"query": "X", "foo": "bar"},
              "Unknown field 'foo'")

    # ---- 5. Unknown tool ------------------------------------------------
    print_section("5. Unknown Tool")

    demo_call(registry, "book_ticket", {"passenger": "Alice"},
              "Tool not registered")

    # ---- 6. Malformed JSON ----------------------------------------------
    print_section("6. Malformed JSON")

    demo_call(registry, "station_search", "{bad json!!!}",
              "Invalid JSON string")

    # ---- 7. Function exception ------------------------------------------
    print_section("7. Function Exception Handling")

    def crashing_search(query: str):
        raise ConnectionError("Database unreachable!")

    registry.register("station_search", crashing_search)
    demo_call(registry, "station_search", {"query": "test"},
              "Function raises an exception")

    # ---- Done -----------------------------------------------------------
    print_section("Demo Complete")
    print("  All scenarios demonstrated successfully.\n")


if __name__ == "__main__":
    main()
