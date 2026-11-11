"""Unit tests for helpdesk.ai_assistant."""

from __future__ import annotations

from unittest.mock import MagicMock

from helpdesk.ai_assistant import AIAssistant, ToolExecutorProtocol
from helpdesk.llm import LLMConnectionError, LLMResponse, ToolCall
from helpdesk.memory_adapter import MemoryAdapter

MOCK_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search railway circulars and refund rules.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_timetable",
            "description": "Run a read-only SQL query against the timetable DB.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
]


def _make_mock_llm(chat_side_effect, json_return=None):
    llm = MagicMock()
    llm.config = MagicMock()
    llm.config.known_secrets.return_value = ()
    llm.chat.side_effect = chat_side_effect
    if json_return is not None:
        llm.generate_json.return_value = json_return
    return llm


class _MockToolExecutor:
    """Mock simulating Member 3 (Neha)'s tool executor."""

    def __init__(self, handler):
        self._handler = handler
        self.calls: list[tuple[str, dict]] = []

    def execute(self, tool_name: str, arguments: dict) -> dict:
        self.calls.append((tool_name, arguments))
        return self._handler(tool_name, arguments)


def test_single_tool_call_document_flow():
    executor = _MockToolExecutor(
        lambda name, args: {
            "ok": True,
            "chunks": [
                {
                    "doc": "rb_cc10_2024_advance_reservation.pdf",
                    "page": 1,
                    "date": "2024-10-16",
                    "text": "Advance Reservation Period reduced from 120 to 60 days.",
                }
            ],
        }
    )
    assert isinstance(executor, ToolExecutorProtocol)

    llm = _make_mock_llm(
        chat_side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="t1", name="search_documents", arguments={"query": "advance reservation"})
                ],
            ),
            LLMResponse(
                content=(
                    '{"answer": "The advance reservation period is 60 days.", '
                    '"sources": [{"doc": "rb_cc10_2024_advance_reservation.pdf", "page": 1}], '
                    '"dont_know": false}'
                )
            ),
        ]
    )

    assistant = AIAssistant(llm, tool_executor=executor, tool_schemas=MOCK_TOOL_SCHEMAS)
    res = assistant.ask("How early can I book a ticket?")

    assert res == {
        "answer": "The advance reservation period is 60 days.",
        "sources": [{"doc": "rb_cc10_2024_advance_reservation.pdf", "page": 1}],
        "sql": None,
        "dont_know": False,
    }
    assert executor.calls == [("search_documents", {"query": "advance reservation"})]


def test_multiple_tool_calls_hybrid_sql_and_documents():
    def _handler(name, args):
        if name == "query_timetable":
            return {
                "ok": True,
                "sql": "SELECT departure FROM schedules WHERE train_number = '12002' AND station_code = 'NDLS';",
                "rows": [{"departure": "06:15"}],
            }
        return {
            "ok": True,
            "chunks": [
                {
                    "doc": "irctc_refund_rules_jan2026.pdf",
                    "page": 1,
                    "text": "Flat cancellation charge for AC Chair Car is Rs 180 plus GST.",
                }
            ],
        }

    executor = _MockToolExecutor(_handler)
    llm = _make_mock_llm(
        chat_side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="t1", name="query_timetable", arguments={"sql": "SELECT departure ..."}),
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="t2", name="search_documents", arguments={"query": "AC Chair Car cancellation"}),
                ],
            ),
            LLMResponse(content="Train 12002 leaves NDLS at 06:15 and AC Chair Car cancellation costs Rs 180 plus GST."),
        ],
        json_return={
            "answer": "Train 12002 leaves NDLS at 06:15. Cancelling AC Chair Car 5 days ahead costs Rs 180 plus GST.",
            "sources": [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}],
            "dont_know": False,
        },
    )

    assistant = AIAssistant(llm, tool_executor=executor, tool_schemas=MOCK_TOOL_SCHEMAS)
    res = assistant.ask("When does 12002 leave NDLS and what is the AC Chair Car cancellation fee?")

    assert res["dont_know"] is False
    assert res["sources"] == [{"doc": "irctc_refund_rules_jan2026.pdf", "page": 1}]
    assert res["sql"] == "SELECT departure FROM schedules WHERE train_number = '12002' AND station_code = 'NDLS';"
    assert len(executor.calls) == 2


def test_tool_execution_error_handled_gracefully():
    def _failing_tool(name, args):
        raise RuntimeError("SQLite database locked")

    executor = _MockToolExecutor(_failing_tool)
    llm = _make_mock_llm(
        chat_side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="t1", name="query_timetable", arguments={"sql": "SELECT 1"})],
            ),
            LLMResponse(content="I don't know because the timetable lookup failed."),
        ]
    )

    assistant = AIAssistant(llm, tool_executor=executor, tool_schemas=MOCK_TOOL_SCHEMAS)
    res = assistant.ask("Which trains run from JP to UDZ?")

    assert res["dont_know"] is True
    assert res["sources"] == []
    assert res["sql"] is None


def test_empty_tool_results_yield_dont_know():
    executor = _MockToolExecutor(lambda name, args: {"ok": True, "chunks": []})
    llm = _make_mock_llm(
        chat_side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="t1", name="search_documents", arguments={"query": "ticket fare"})],
            ),
            LLMResponse(content="Fare is Rs 1500."),  # LLM tries to hallucinate despite empty chunks
        ]
    )

    assistant = AIAssistant(llm, tool_executor=executor, tool_schemas=MOCK_TOOL_SCHEMAS)
    res = assistant.ask("How much is a 3AC ticket?")

    assert res["dont_know"] is True
    assert res["sources"] == []
    assert "i don't know" in res["answer"].lower()


def test_ungrounded_factual_answer_without_tool_call_is_blocked():
    executor = _MockToolExecutor(lambda name, args: {"ok": True, "chunks": []})
    llm = _make_mock_llm(
        chat_side_effect=[
            LLMResponse(content="Train 12951 departs at 17:00."),  # Skipped tools & invented fact
        ]
    )
    assistant = AIAssistant(llm, tool_executor=executor, tool_schemas=MOCK_TOOL_SCHEMAS)
    res = assistant.ask("When does Mumbai Rajdhani leave?")

    assert res["dont_know"] is True
    assert res["sources"] == []
    assert "i don't know" in res["answer"].lower()


def test_unsafe_write_request_refusal_preserved_with_dont_know():
    executor = _MockToolExecutor(lambda name, args: {"ok": True})
    llm = _make_mock_llm(
        chat_side_effect=[
            LLMResponse(content="I cannot modify the database because only read-only queries are allowed."),
        ]
    )
    assistant = AIAssistant(llm, tool_executor=executor, tool_schemas=MOCK_TOOL_SCHEMAS)
    res = assistant.ask("Change train 12002 departure to 07:00.")

    assert res["dont_know"] is True
    assert "cannot modify" in res["answer"].lower()
    assert executor.calls == []


def test_followup_questions_use_session_memory():
    executor = _MockToolExecutor(
        lambda name, args: {
            "ok": True,
            "chunks": [{"doc": "rb_cc08_2025_tatkal.pdf", "page": 1, "text": "Aadhaar authentication required for Tatkal."}],
        }
    )
    llm = _make_mock_llm(
        chat_side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="t1", name="search_documents", arguments={"query": "Tatkal Aadhaar"})],
            ),
            LLMResponse(
                content='{"answer": "Yes, Aadhaar authentication is required for Tatkal bookings.", "sources": [{"doc": "rb_cc08_2025_tatkal.pdf", "page": 1}], "dont_know": false}'
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="t2", name="search_documents", arguments={"query": "Tatkal Aadhaar effective date"})],
            ),
            LLMResponse(
                content='{"answer": "It applies per Circular 08/2025.", "sources": [{"doc": "rb_cc08_2025_tatkal.pdf", "page": 1}], "dont_know": false}'
            ),
        ]
    )
    memory = MemoryAdapter()
    assistant = AIAssistant(
        llm,
        tool_executor=executor,
        tool_schemas=MOCK_TOOL_SCHEMAS,
        memory_adapter=memory,
    )

    res1 = assistant.ask("Do I need Aadhaar for Tatkal?", session_id="user_42")
    assert res1["dont_know"] is False

    res2 = assistant.ask("Which circular mentions that?", session_id="user_42")
    assert res2["dont_know"] is False

    # Inspect second question's first LLM call messages to verify prior context was injected
    second_turn_messages = llm.chat.call_args_list[2].args[0]
    roles_and_text = [(m["role"], m.get("content", "")) for m in second_turn_messages]
    assert ("user", "Do I need Aadhaar for Tatkal?") in roles_and_text
    assert ("assistant", "Yes, Aadhaar authentication is required for Tatkal bookings.") in roles_and_text


def test_llm_network_error_handled_gracefully():
    llm = _make_mock_llm(chat_side_effect=LLMConnectionError("Offline"))
    assistant = AIAssistant(llm)
    res = assistant.ask("Hello?")

    assert res["dont_know"] is True
    assert res["sources"] == []
    assert res["sql"] is None
    assert "i don't know" in res["answer"].lower()


def test_answer_with_evidence_direct_helper():
    llm = _make_mock_llm(
        chat_side_effect=[],
        json_return={
            "answer": "Train 12430 is Delhi H Nizamuddin-Bangalore Rajdhani Express.",
            "sources": [],
            "dont_know": False,
        },
    )
    assistant = AIAssistant(llm)
    res = assistant.answer_with_evidence(
        question="What is the name of train 12430?",
        db_results=[{"number": "12430", "name": "Delhi H Nizamuddin-Bangalore Rajdhani Express"}],
        sql="SELECT number, name FROM trains WHERE number = '12430';",
        session_id="direct_sess",
    )
    assert res["dont_know"] is False
    assert res["sql"] == "SELECT number, name FROM trains WHERE number = '12430';"
    assert len(assistant.memory_adapter.get_raw_history("direct_sess")) == 2


def test_max_tool_steps_reached_synthesizes_collected_evidence():
    executor = _MockToolExecutor(
        lambda name, args: {
            "ok": True,
            "chunks": [
                {
                    "doc": "rb_cc10_2025_reservation_charts.pdf",
                    "page": 2,
                    "text": "First reservation chart prepared 8 hours before departure.",
                }
            ],
        }
    )
    llm = _make_mock_llm(
        chat_side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="t1", name="search_documents", arguments={"query": "chart"})],
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="t2", name="search_documents", arguments={"query": "chart time"})],
            ),
        ],
        json_return={
            "answer": "First reservation charts are prepared 8 hours before scheduled departure.",
            "sources": [{"doc": "rb_cc10_2025_reservation_charts.pdf", "page": 2}],
            "dont_know": False,
        },
    )
    assistant = AIAssistant(
        llm,
        tool_executor=executor,
        tool_schemas=MOCK_TOOL_SCHEMAS,
        max_tool_steps=2,
    )
    res = assistant.ask("When is the first reservation chart prepared?")
    assert res["dont_know"] is False
    assert res["sources"] == [{"doc": "rb_cc10_2025_reservation_charts.pdf", "page": 2}]
