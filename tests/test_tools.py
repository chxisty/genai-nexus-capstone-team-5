"""
test_tools.py — Unit tests for the tool-calling module.

Covers:
  ✓ Successful execution of each tool
  ✓ Missing required arguments
  ✓ Wrong argument types
  ✓ Empty-string validation
  ✓ Unknown argument fields
  ✓ Unknown tool names
  ✓ Malformed JSON string arguments
  ✓ Function exceptions are caught safely
  ✓ JSON-string arguments (LLM-style)
  ✓ Registry listing & registration
"""

import json
import unittest

from tools.registry import ToolRegistry
from tools.railway_tools import register_railway_tools


class TestToolRegistry(unittest.TestCase):
    """Tests for ToolRegistry validation, dispatch, and error handling."""

    def setUp(self):
        """Create a fresh registry with mock tools before each test."""
        self.registry = ToolRegistry()
        register_railway_tools(self.registry)

    # ------------------------------------------------------------------
    # Successful execution
    # ------------------------------------------------------------------
    def test_station_search_success(self):
        result = self.registry.call("station_search", {"query": "Delhi"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool"], "station_search")
        self.assertIn("matches", result["result"])
        self.assertTrue(len(result["result"]["matches"]) > 0)

    def test_train_search_success(self):
        args = {"source": "NDLS", "destination": "BCT", "date": "2026-12-01"}
        result = self.registry.call("train_search", args)
        self.assertEqual(result["status"], "success")
        self.assertIn("trains", result["result"])

    def test_train_details_success(self):
        result = self.registry.call("train_details", {"train_number": "12345"})
        self.assertEqual(result["status"], "success")
        self.assertIn("name", result["result"])

    def test_route_lookup_success(self):
        result = self.registry.call("route_lookup", {"train_number": "12345"})
        self.assertEqual(result["status"], "success")
        self.assertIn("stops", result["result"])

    # ------------------------------------------------------------------
    # JSON-string arguments (LLM-style)
    # ------------------------------------------------------------------
    def test_json_string_arguments(self):
        json_args = json.dumps({"query": "Mumbai"})
        result = self.registry.call("station_search", json_args)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["query"], "Mumbai")

    # ------------------------------------------------------------------
    # Missing required arguments
    # ------------------------------------------------------------------
    def test_missing_required_arg(self):
        result = self.registry.call("station_search", {})
        self.assertEqual(result["status"], "error")
        self.assertIn("Missing required argument 'query'", result["error"])

    def test_missing_multiple_required_args(self):
        result = self.registry.call("train_search", {"source": "NDLS"})
        self.assertEqual(result["status"], "error")
        self.assertIn("destination", result["error"])
        self.assertIn("date", result["error"])

    # ------------------------------------------------------------------
    # Wrong argument types
    # ------------------------------------------------------------------
    def test_wrong_type_int_instead_of_string(self):
        result = self.registry.call("station_search", {"query": 123})
        self.assertEqual(result["status"], "error")
        self.assertIn("must be string", result["error"])

    def test_wrong_type_list_instead_of_string(self):
        result = self.registry.call("train_details", {"train_number": ["12345"]})
        self.assertEqual(result["status"], "error")
        self.assertIn("must be string", result["error"])

    # ------------------------------------------------------------------
    # Empty-string validation
    # ------------------------------------------------------------------
    def test_empty_string_rejected(self):
        result = self.registry.call("station_search", {"query": ""})
        self.assertEqual(result["status"], "error")
        self.assertIn("must not be empty", result["error"])

    def test_whitespace_only_rejected(self):
        result = self.registry.call("station_search", {"query": "   "})
        self.assertEqual(result["status"], "error")
        self.assertIn("must not be empty", result["error"])

    # ------------------------------------------------------------------
    # Unknown argument fields
    # ------------------------------------------------------------------
    def test_unknown_field_rejected(self):
        result = self.registry.call(
            "station_search", {"query": "Delhi", "foo": "bar"}
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown argument 'foo'", result["error"])

    # ------------------------------------------------------------------
    # Unknown tool
    # ------------------------------------------------------------------
    def test_unknown_tool(self):
        result = self.registry.call("nonexistent_tool", {"a": 1})
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown tool", result["error"])

    # ------------------------------------------------------------------
    # Malformed JSON
    # ------------------------------------------------------------------
    def test_malformed_json(self):
        result = self.registry.call("station_search", "{bad json!!}")
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid JSON", result["error"])

    def test_non_dict_json(self):
        result = self.registry.call("station_search", '"just a string"')
        self.assertEqual(result["status"], "error")
        self.assertIn("must be a dict", result["error"])

    # ------------------------------------------------------------------
    # Function exception handling
    # ------------------------------------------------------------------
    def test_function_exception_caught(self):
        def exploding_tool(query: str):
            raise RuntimeError("Database connection lost!")

        self.registry.register("station_search", exploding_tool)
        result = self.registry.call("station_search", {"query": "Delhi"})
        self.assertEqual(result["status"], "error")
        self.assertIn("Execution failed", result["error"])
        self.assertIn("Database connection lost", result["error"])

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------
    def test_list_tools(self):
        tools = self.registry.list_tools()
        self.assertEqual(
            tools,
            ["route_lookup", "station_search", "train_details", "train_search"],
        )

    def test_register_overwrite(self):
        """Registering the same name again overwrites the old function."""
        self.registry.register("station_search", lambda query: {"custom": True})
        result = self.registry.call("station_search", {"query": "test"})
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["result"]["custom"])

    def test_register_non_callable_raises(self):
        with self.assertRaises(TypeError):
            self.registry.register("bad", "not a function")

    def test_arguments_not_dict_or_string(self):
        result = self.registry.call("station_search", 42)
        self.assertEqual(result["status"], "error")
        self.assertIn("must be a dict", result["error"])


if __name__ == "__main__":
    unittest.main()
