"""
tests/test_pipeline_integration.py
----------------------------------
End-to-end integration tests for the Railway Passenger Help Desk pipeline.

Tests real ToolRegistry, real SQLite timetable database, real SessionManager,
real MemoryAdapter, ContextHandler, and ReliableHelpDesk with mock/fake LLM clients.
Does NOT require a real API key.
"""

from __future__ import annotations

from typing import Any
import pytest

from helpdesk.llm import LLMAuthError, LLMResponse, ToolCall
from helpdesk.pipeline import RailwayHelpDeskPipeline
from helpdesk.session_manager import SessionManager
from tools.database_tools import (
    get_connection,
    get_train_details,
    get_train_route,
    search_stations,
    search_trains,
)
from tools.registry import ToolRegistry


class FakeLLMClient:
    """Mock LLMClient for deterministic integration testing."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.responses = list(responses) if responses else []
        self.call_count = 0
        self.history: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], *, tools=None) -> LLMResponse:
        self.call_count += 1
        self.history.append(messages)
        if self.responses:
            return self.responses.pop(0)
        # Default synthesized grounded answer
        return LLMResponse(
            content='{"answer": "Here is the train information based on the timetable.", "sources": [], "sql": null, "dont_know": false}',
            tool_calls=[],
        )


class FailingLLMClient:
    """Mock LLMClient that simulates API failures / network errors."""

    def chat(self, messages, *, tools=None):
        raise LLMAuthError("Invalid or missing API key.")


@pytest.fixture
def temp_dbs(tmp_path):
    session_db = tmp_path / "test_sessions.db"
    return str(session_db)


# 1. Application initializes successfully
def test_pipeline_initialization(temp_dbs):
    pipeline = RailwayHelpDeskPipeline(
        llm_client=FakeLLMClient(),
        session_db_path=temp_dbs,
    )
    assert pipeline is not None
    assert isinstance(pipeline.registry, ToolRegistry)
    assert isinstance(pipeline.session_manager, SessionManager)


# 2. Database initializes and can be opened
def test_database_connection():
    with get_connection() as conn:
        cursor = conn.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1


# 3. Station search works
def test_station_search_real_db():
    result = search_stations("Delhi")
    assert result["count"] > 0
    assert any("NDLS" in m["code"] for m in result["matches"])


# 4. Train search works
def test_train_search_real_db():
    result = search_trains("LTT", "GWL", "2026-11-15")
    assert result["count"] == 6
    assert result["source"]["code"] == "LTT"
    assert result["destination"]["code"] == "GWL"


# 5. Train details work
def test_train_details_real_db():
    result = get_train_details("01101")
    assert result["number"] == "01101"
    assert "3A" in result["classes"]


# 6. Train route lookup works
def test_train_route_real_db():
    result = get_train_route("01101")
    assert result["count"] > 0
    assert result["stops"][0]["station_code"] == "LTT"


# 7. Tool schema validation works
def test_tool_schema_validation():
    registry = ToolRegistry()
    registry.register("station_search", search_stations)

    # Valid arguments
    res = registry.call("station_search", {"query": "Mumbai"})
    assert res["status"] == "success"

    # Missing required argument
    err_res = registry.call("station_search", {})
    assert err_res["status"] == "error"
    assert "Missing required argument" in err_res["error"]


# 8. Invalid tools return safe errors
def test_invalid_tool_call():
    registry = ToolRegistry()
    res = registry.call("non_existent_tool", {"foo": "bar"})
    assert res["status"] == "error"
    assert "Unknown tool" in res["error"]


# 9. Multiple turns share the same session
def test_multi_turn_session_sharing(temp_dbs):
    fake_llm = FakeLLMClient([
        # Turn 1: tool call then answer
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="train_search",
                    arguments={"source": "LTT", "destination": "GWL", "date": "2026-11-15"},
                )
            ],
        ),
        LLMResponse(
            content='{"answer": "Trains are available between LTT and GWL.", "sources": [], "sql": null, "dont_know": false}',
            tool_calls=[],
        ),
        # Turn 2: tool call then answer
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c2",
                    name="train_details",
                    arguments={"train_number": "01101"},
                )
            ],
        ),
        LLMResponse(
            content='{"answer": "Train 01101 departs LTT at 15:50.", "sources": [], "sql": null, "dont_know": false}',
            tool_calls=[],
        ),
    ])
    pipeline = RailwayHelpDeskPipeline(
        llm_client=fake_llm,
        session_db_path=temp_dbs,
    )

    sid = "test_shared_session"
    res1 = pipeline.ask("Find trains from LTT to GWL", session_id=sid)
    res2 = pipeline.ask("What time does train 01101 depart?", session_id=sid)

    assert res1["dont_know"] is False
    assert res2["dont_know"] is False

    history = pipeline.get_history(sid)
    # Check that both turns were stored under the same session
    assert len(history) >= 4


# 10. Follow-up questions work when supported
def test_follow_up_context_handling(temp_dbs):
    fake_llm = FakeLLMClient([
        # Turn 1: tool call and answer
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="train_details",
                    arguments={"train_number": "01101"},
                )
            ],
        ),
        LLMResponse(
            content='{"answer": "Found train 01101 from LTT to GWL.", "sources": [], "sql": null, "dont_know": false}',
            tool_calls=[],
        ),
        # Turn 2: route tool call and answer
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c2",
                    name="route_lookup",
                    arguments={"train_number": "01101"},
                )
            ],
        ),
        LLMResponse(
            content='{"answer": "Its stops include Kalyan and Nashik.", "sources": [], "sql": null, "dont_know": false}',
            tool_calls=[],
        ),
    ])
    pipeline = RailwayHelpDeskPipeline(
        llm_client=fake_llm,
        session_db_path=temp_dbs,
    )

    sid = "follow_up_sess"
    res1 = pipeline.ask("Tell me about train 01101", session_id=sid)
    res2 = pipeline.ask("What are its stops?", session_id=sid)

    assert res1["dont_know"] is False
    assert res2["dont_know"] is False
    assert "stops" in res2["answer"].lower()


# 11. Session history is retained in SQLite
def test_session_history_retention(temp_dbs):
    fake_llm = FakeLLMClient([
        LLMResponse(
            content='{"answer": "Delhi stations include NDLS and DLI.", "sources": [], "sql": null, "dont_know": false}',
            tool_calls=[],
        )
    ])
    pipeline = RailwayHelpDeskPipeline(
        llm_client=fake_llm,
        session_db_path=temp_dbs,
    )

    sid = "persisted_session"
    pipeline.ask("Search stations in Delhi", session_id=sid)

    # Reconnect to same SQLite db with fresh pipeline
    new_pipeline = RailwayHelpDeskPipeline(
        llm_client=fake_llm,
        session_db_path=temp_dbs,
    )
    history = new_pipeline.get_history(sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


# 12. Unknown railway questions handled safely without crashing
def test_unknown_railway_question(temp_dbs):
    # LLM does not call tools and returns no evidence
    fake_llm = FakeLLMClient([
        LLMResponse(content="", tool_calls=[])
    ])
    pipeline = RailwayHelpDeskPipeline(
        llm_client=fake_llm,
        session_db_path=temp_dbs,
    )

    res = pipeline.ask("What is the flight schedule from Mumbai to London?", session_id="test_unknown")
    assert res["dont_know"] is True
    assert "I don't know" in res["answer"]
    assert res["sources"] == []
    assert res["sql"] is None


# 13. Missing API keys handled gracefully
def test_missing_api_key_graceful_handling(temp_dbs):
    pipeline = RailwayHelpDeskPipeline(
        llm_client=FailingLLMClient(),
        session_db_path=temp_dbs,
    )
    res = pipeline.ask("Which trains go to Jaipur?", session_id="key_fail")
    assert res["dont_know"] is True
    assert "I don't know" in res["answer"]


# 14. LLM API failure does not crash
def test_api_failure_does_not_crash(temp_dbs):
    class CrashingLLM:
        def chat(self, *args, **kwargs):
            raise RuntimeError("Unexpected upstream HTTP timeout")

    pipeline = RailwayHelpDeskPipeline(
        llm_client=CrashingLLM(),
        session_db_path=temp_dbs,
    )
    try:
        pipeline.ask("Tell me train details", session_id="crash_test")
    except Exception as exc:
        # Should raise an informative exception without corrupting database
        assert "timeout" in str(exc).lower()


# 15. The answer follows the required response contract
def test_response_contract_compliance(temp_dbs):
    fake_llm = FakeLLMClient([
        LLMResponse(
            content='{"answer": "New Delhi station code is NDLS.", "sources": [], "sql": null, "dont_know": false}',
            tool_calls=[],
        )
    ])
    pipeline = RailwayHelpDeskPipeline(
        llm_client=fake_llm,
        session_db_path=temp_dbs,
    )

    res = pipeline.ask("What is the code for New Delhi?", session_id="contract_test")
    assert isinstance(res, dict)
    assert set(res.keys()) == {"answer", "sources", "sql", "dont_know"}
    assert isinstance(res["answer"], str)
    assert isinstance(res["sources"], list)
    assert res["sql"] is None or isinstance(res["sql"], str)
    assert isinstance(res["dont_know"], bool)


# 16. Multi-step Tool Calling with Real DB Results
def test_tool_calling_with_real_database_results(temp_dbs):
    # Step 1: LLM decides to call train_search
    # Step 2: LLM receives real DB results and emits final answer
    fake_llm = FakeLLMClient([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="train_search",
                    arguments={"source": "LTT", "destination": "GWL", "date": "2026-11-15"},
                )
            ],
        ),
        LLMResponse(
            content='{"answer": "There are 6 trains from LTT to GWL including 01101 and 01127.", "sources": [], "sql": null, "dont_know": false}',
            tool_calls=[],
        ),
    ])

    pipeline = RailwayHelpDeskPipeline(
        llm_client=fake_llm,
        session_db_path=temp_dbs,
    )

    res = pipeline.ask("Find trains from LTT to GWL", session_id="tool_step_test")
    assert res["dont_know"] is False
    assert "6 trains" in res["answer"]
