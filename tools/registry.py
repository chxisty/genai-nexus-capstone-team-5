"""
registry.py — ToolRegistry: register, validate, and dispatch tool calls.

Design goals
------------
* Accept tool calls exactly as an LLM returns them (tool name + JSON string).
* Validate arguments against the schema (required fields, types, empty
  strings, unknown fields) *before* calling the function.
* Return a uniform { "status": "success"|"error", ... } envelope so the
  caller never has to handle raw exceptions.
* Allow Krishna (Member 2) to register real database-backed functions later
  by calling  registry.register("tool_name", real_function)  — no changes
  to this file needed.
* Never use eval() or exec().
"""

import json
from tools.schemas import get_schema_by_name

# Python type strings  →  allowed Python types
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


class ToolRegistry:
    """Central registry that maps tool names to callable functions."""

    def __init__(self):
        self._tools = {}          # name  →  callable

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, name: str, func):
        """Register *func* under *name*.  Overwrites any previous binding."""
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}")
        self._tools[name] = func

    def list_tools(self):
        """Return a sorted list of registered tool names."""
        return sorted(self._tools.keys())

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_arguments(tool_name: str, arguments: dict) -> list:
        """
        Validate *arguments* against the schema for *tool_name*.
        Returns a list of error strings (empty = valid).
        """
        schema = get_schema_by_name(tool_name)
        if schema is None:
            return [f"No schema found for tool '{tool_name}'."]

        params = schema["function"]["parameters"]
        properties = params.get("properties", {})
        required = params.get("required", [])
        errors = []

        # 1. Check for unknown fields
        for key in arguments:
            if key not in properties:
                errors.append(f"Unknown argument '{key}'.")

        # 2. Check required fields are present and non-empty
        for field in required:
            if field not in arguments:
                errors.append(f"Missing required argument '{field}'.")
            elif isinstance(arguments[field], str) and arguments[field].strip() == "":
                errors.append(f"Argument '{field}' must not be empty.")

        # 3. Type checks for provided fields
        for field, value in arguments.items():
            if field not in properties:
                continue  # already flagged above
            expected_type_str = properties[field].get("type")
            if expected_type_str and expected_type_str in _TYPE_MAP:
                expected = _TYPE_MAP[expected_type_str]
                if not isinstance(value, expected):
                    errors.append(
                        f"Argument '{field}' must be {expected_type_str}, "
                        f"got {type(value).__name__}."
                    )

        return errors

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def call(self, tool_name: str, arguments):
        """
        Validate and execute a tool call.

        Parameters
        ----------
        tool_name : str
            Registered tool name (e.g. "station_search").
        arguments : dict | str
            Either a Python dict **or** a JSON string (as LLMs typically
            return).  If a string, it is parsed with json.loads().

        Returns
        -------
        dict   with keys:
            "status"  – "success" or "error"
            "tool"    – the tool name that was called
            "result"  – function return value  (on success)
            "error"   – human-readable error message (on error)
        """
        # --- 1.  Parse arguments if they arrive as a JSON string ----------
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                return {
                    "status": "error",
                    "tool": tool_name,
                    "error": f"Invalid JSON arguments: {exc}",
                }

        if not isinstance(arguments, dict):
            return {
                "status": "error",
                "tool": tool_name,
                "error": (
                    f"Arguments must be a dict or JSON string, "
                    f"got {type(arguments).__name__}."
                ),
            }

        # --- 2.  Unknown tool? -------------------------------------------
        if tool_name not in self._tools:
            return {
                "status": "error",
                "tool": tool_name,
                "error": f"Unknown tool '{tool_name}'.",
            }

        # --- 3.  Validate against schema ---------------------------------
        validation_errors = self._validate_arguments(tool_name, arguments)
        if validation_errors:
            return {
                "status": "error",
                "tool": tool_name,
                "error": "; ".join(validation_errors),
            }

        # --- 4.  Execute --------------------------------------------------
        try:
            result = self._tools[tool_name](**arguments)
            return {
                "status": "success",
                "tool": tool_name,
                "result": result,
            }
        except Exception as exc:
            return {
                "status": "error",
                "tool": tool_name,
                "error": f"Execution failed: {exc}",
            }
