"""
Interactive CLI for manually testing the core (no API required).

Run:  python cli.py
Then enter a summary and description; if the chatbot asks a follow-up
question, answer it.
"""
from __future__ import annotations

import sys

# On Windows consoles, force UTF-8 I/O so Persian text renders correctly.
for _stream in (sys.stdout, sys.stdin, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from src.conversation.manager import ConversationManager


def _print_result(resp: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Status: {resp['status']}   |   Questions asked: {resp['questions_asked']}")
    result = resp["result"]
    if result:
        print("Labels:")
        for layer_id, label in result["labels"].items():
            ev = ", ".join(result["evidence"].get(layer_id, [])) or "—"
            print(f"   • {layer_id}: {label}   (evidence: {ev})")
        print(f"Suggested summary: {result['suggested_summary']}")
        if result["needs_review"]:
            print("⚠️  Low confidence — needs human review.")
    print("=" * 60 + "\n")


def main() -> None:
    print("Ticket Routing Chatbot — press Ctrl+C to exit\n")
    manager = ConversationManager()

    while True:
        try:
            summary = input("Summary> ").strip()
            description = input("Description> ").strip()
            if not summary and not description:
                continue

            resp = manager.start(summary, description)

            # Follow-up question loop
            while resp["status"] == "need_info":
                print(f"\n🤖 Question: {resp['question']}")
                ans = input("Your answer> ").strip()
                resp = manager.answer(resp["session_id"], ans)

            _print_result(resp)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
