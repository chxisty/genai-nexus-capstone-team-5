"""
tests/test_cli.py
-----------------
Tests for the main.py terminal CLI application.
Verifies command dispatch, session commands, banners, and format_response.
"""

from __future__ import annotations

import io
import sys
from unittest.mock import patch
import pytest

from helpdesk.pipeline import RailwayHelpDeskPipeline
from main import (
    BANNER,
    format_response,
    main,
    print_history,
    run_interactive,
)


class FakePipeline:
    """Mock pipeline for CLI testing."""

    def __init__(self):
        self.sessions = {}

    def ask(self, question: str, *, session_id: str | None = None) -> dict:
        sid = session_id or "default"
        self.sessions.setdefault(sid, []).append({"role": "user", "content": question})
        ans = {
            "answer": f"Echo: {question}",
            "sources": [{"doc": "test.pdf", "page": 1}],
            "sql": "SELECT 1;",
            "dont_know": False,
        }
        self.sessions[sid].append({"role": "assistant", "content": ans["answer"]})
        return ans

    def get_history(self, session_id: str) -> list[dict]:
        return self.sessions.get(session_id, [])

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


def test_format_response_with_sources_and_sql():
    res = {
        "answer": "Train 12430 is the Bangalore Rajdhani Express.",
        "sources": [{"doc": "railway_rules.pdf", "page": 4}],
        "sql": "SELECT name FROM trains WHERE number = '12430'",
        "dont_know": False,
    }
    output = format_response(res)
    assert "Assistant:\nTrain 12430" in output
    assert "railway_rules.pdf (Page 4)" in output
    assert "SELECT name FROM trains WHERE number = '12430'" in output


def test_format_response_without_sources():
    res = {
        "answer": "Information unavailable.",
        "sources": [],
        "sql": None,
        "dont_know": True,
    }
    output = format_response(res)
    assert "None (No official document cited)" in output
    assert "SQL:\nnull" in output


def test_cli_single_question_argument(capsys):
    with patch("main.get_pipeline", return_value=FakePipeline()):
        with patch.object(sys, "argv", ["main.py", "Which trains go to Mumbai?"]):
            main()

    captured = capsys.readouterr()
    assert "Assistant:\nEcho: Which trains go to Mumbai?" in captured.out
    assert "test.pdf (Page 1)" in captured.out


def test_cli_help_flag(capsys):
    with patch.object(sys, "argv", ["main.py", "/help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "RAILWAY PASSENGER HELP DESK" in captured.out
    assert "/help" in captured.out


def test_interactive_session_exit_command(capsys):
    pipeline = FakePipeline()
    # Simulate user typing /help then /exit
    inputs = iter(["/help", "/exit"])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_interactive(pipeline)

    captured = capsys.readouterr()
    assert "Available Commands:" in captured.out
    assert "Thank you for using Railway Passenger Help Desk!" in captured.out


def test_interactive_session_new_and_history(capsys):
    pipeline = FakePipeline()
    inputs = iter([
        "Find trains from LTT to GWL",
        "/history",
        "/new",
        "/history",
        "/exit"
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_interactive(pipeline)

    captured = capsys.readouterr()
    assert "Echo: Find trains from LTT to GWL" in captured.out
    assert "Started a new conversation session:" in captured.out
    assert "No history recorded for this session yet." in captured.out
