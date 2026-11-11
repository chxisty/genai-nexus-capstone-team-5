"""
schemas.py — JSON-compatible tool schemas for Groq / OpenAI-style function calling.

Each schema follows the OpenAI function-calling format so it can be passed
directly to an LLM's `tools` parameter.  The registry uses the same schemas
for argument validation at runtime.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "station_search",
            "description": "Search for railway stations by name or partial name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Full or partial station name to search for."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "train_search",
            "description": (
                "Search for trains between two stations on a given date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Departure station name or code."
                    },
                    "destination": {
                        "type": "string",
                        "description": "Arrival station name or code."
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date in YYYY-MM-DD format."
                    }
                },
                "required": ["source", "destination", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "train_details",
            "description": "Get detailed information about a specific train by its number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "train_number": {
                        "type": "string",
                        "description": "The unique train number (e.g. '12345')."
                    }
                },
                "required": ["train_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_lookup",
            "description": "Look up the route (list of stops) for a given train number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "train_number": {
                        "type": "string",
                        "description": "The unique train number whose route to look up."
                    }
                },
                "required": ["train_number"]
            }
        }
    },
]


def get_schema_by_name(tool_name: str):
    """Return the schema dict for a tool, or None if not found."""
    for schema in TOOL_SCHEMAS:
        if schema["function"]["name"] == tool_name:
            return schema
    return None
