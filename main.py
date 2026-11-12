"""
main.py
-------
Main application entry point for the Railway Passenger Help Desk.

Usage:
    python main.py              # Launch interactive terminal session
    python main.py "question"   # Ask a single question directly
"""

from __future__ import annotations

import os
import sys
import uuid

# Auto-relaunch inside project virtual environment if running with system Python
_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python_win = os.path.join(_dir, ".venv", "Scripts", "python.exe")
_venv_python_unix = os.path.join(_dir, ".venv", "bin", "python")
_venv_python = (
    _venv_python_win
    if os.path.isfile(_venv_python_win)
    else (_venv_python_unix if os.path.isfile(_venv_python_unix) else None)
)

if _venv_python and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower():
    import subprocess
    sys.exit(subprocess.call([_venv_python] + sys.argv))

from helpdesk.pipeline import RailwayHelpDeskPipeline, get_pipeline

BANNER = """============================================
     RAILWAY PASSENGER HELP DESK
============================================

Welcome! Ask questions about Indian Railways.

Commands:
  /help    - Show available commands
  /new     - Start a new conversation
  /history - Show current session history
  /exit    - Exit application"""


def format_response(result: dict) -> str:
    """Format structured answer dictionary for terminal display."""
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    sql = result.get("sql")

    if sources:
        formatted_sources = []
        for i, s in enumerate(sources, 1):
            doc = s.get("doc", "Unknown document")
            page = s.get("page")
            if page is not None:
                formatted_sources.append(f"{i}. {doc} (Page {page})")
            else:
                formatted_sources.append(f"{i}. {doc}")
        sources_str = "\n".join(formatted_sources)
    else:
        sources_str = "None (No official document cited)"

    sql_str = sql if sql else "null"

    return f"\nAssistant:\n{answer}\n\nSources:\n{sources_str}\n\nSQL:\n{sql_str}\n"


def print_history(pipeline: RailwayHelpDeskPipeline, session_id: str) -> None:
    """Display conversation history for the active session."""
    history = pipeline.get_history(session_id)
    if not history:
        print("\nNo history recorded for this session yet.\n")
        return

    print(f"\n--- Session History ({session_id}) ---")
    for msg in history:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            print(f"[{role} Tool Call]: {tool_calls}")
        elif content:
            print(f"[{role}]: {content}")
    print("----------------------------------------\n")


def run_interactive(pipeline: RailwayHelpDeskPipeline) -> None:
    """Run interactive terminal REPL."""
    print(BANNER)
    has_api_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY"))
    if not has_api_key:
        print("\n[Tip: Set GEMINI_API_KEY in .env for full AI reasoning. Direct timetable queries work offline via SQLite.]\n")
    else:
        print()
    session_id = f"sess_{uuid.uuid4().hex[:10]}"

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nThank you for using Railway Passenger Help Desk!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print("\nThank you for using Railway Passenger Help Desk!")
            break

        if cmd == "/help":
            print("""
Available Commands:
  /help    - Show this help message
  /new     - Start a new conversation (fresh session)
  /history - Show history of current session
  /exit    - Exit the application
""")
            continue

        if cmd == "/new":
            session_id = f"sess_{uuid.uuid4().hex[:10]}"
            print(f"\nStarted a new conversation session: {session_id}\n")
            continue

        if cmd == "/history":
            print_history(pipeline, session_id)
            continue

        try:
            result = pipeline.ask(user_input, session_id=session_id)
            print(format_response(result))
        except Exception as exc:
            print(f"\nAssistant:\nAn error occurred while processing your query: {exc}\n")


def main() -> None:
    pipeline = get_pipeline()

    if len(sys.argv) > 1:
        # Non-interactive single question mode
        question = " ".join(sys.argv[1:]).strip()
        if question in ("/help", "--help", "-h"):
            print(BANNER)
            sys.exit(0)
        result = pipeline.ask(question)
        print(format_response(result))
        return

    run_interactive(pipeline)


if __name__ == "__main__":
    main()
