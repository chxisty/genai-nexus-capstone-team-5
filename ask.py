"""
ask.py
------
CLI shortcut bridging to main.py and helpdesk.pipeline.
"""

from __future__ import annotations

import json
import sys

from helpdesk.pipeline import ask
from main import format_response


def main():
    if len(sys.argv) < 2:
        # Delegate to main interactive REPL
        from main import main as main_cli
        main_cli()
        return

    args = sys.argv[1:]
    json_mode = "--json" in args
    question_parts = [a for a in args if a != "--json"]
    question = " ".join(question_parts).strip()
    if not question:
        print("Please provide a question.")
        sys.exit(1)

    result = ask(question)

    if json_mode:
        print(json.dumps(result, indent=2))
    else:
        print(format_response(result))


if __name__ == "__main__":
    main()
