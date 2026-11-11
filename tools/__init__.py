"""
tools — Tool Calling & Query Processing package
for the Railway Passenger Help Desk GenAI Capstone.

Provides:
  - TOOL_SCHEMAS   : JSON-compatible schema definitions for LLM function calling
  - ToolRegistry   : Register, validate, and dispatch tool calls
  - register_railway_tools : Pre-register mock railway tools for demo/testing

Author : Member 3 (Tool Calling & Query Processing)
"""

from tools.schemas import TOOL_SCHEMAS
from tools.registry import ToolRegistry
from tools.railway_tools import register_railway_tools

__all__ = ["TOOL_SCHEMAS", "ToolRegistry", "register_railway_tools"]
