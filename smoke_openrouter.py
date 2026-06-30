"""
Initial smoke test against OpenRouter — sends one ticket and prints the detected category.

Purpose: confirm that (1) the OpenRouter key, (2) the DeepSeek model, and (3)
the JSON classification path all work. This script calls the project's real
core (Classifier), i.e. exactly one full classification "round" — with no
follow-up question loop.

Prerequisites (in the .env file):
    DEEPSEEK_API_KEY=sk-or-v1-...                  # OpenRouter key
    DEEPSEEK_BASE_URL=https://openrouter.ai/api/v1
    DEEPSEEK_MODEL=deepseek/deepseek-chat-v3-0324  # or the free tier: ...:free

Run:
    python -m scripts.smoke_openrouter
    python -m scripts.smoke_openrouter "ticket summary" "full problem description"
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the project root (like scripts/evaluate.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from src.classifier.classifier import Classifier  # noqa: E402

# Sample ticket (taken from real data in data/examples.jsonl)
SAMPLE_SUMMARY = "مشکل ثبت ورود و خروج"
SAMPLE_DESCRIPTION = (
    "با سلام، طبق پیوست پانچ ورود و خروج برای تاریخ ۱۹ مرداد ماه در سامانهٔ ERP "
    "به درستی ثبت نشده است. لطفاً بررسی نمایید."
)


def main() -> None:
    # On Windows consoles, force UTF-8 output so Persian text renders correctly.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    argv = sys.argv[1:]
    summary = argv[0] if len(argv) >= 1 else SAMPLE_SUMMARY
    description = argv[1] if len(argv) >= 2 else SAMPLE_DESCRIPTION

    settings.require_api_key()  # if the key is missing, prints a setup hint

    print("=" * 60)
    print("OpenRouter / DeepSeek smoke test")
    print("=" * 60)
    print(f"Endpoint : {settings.deepseek_base_url}")
    print(f"Model    : {settings.model}")
    print(f"Ticket   : {summary} | {description}")
    print("-" * 60)

    clf = Classifier()  # prompt + few-shot are built once
    output, meta = clf.classify(summary, description)

    print("Classification result (one round):")
    for layer in clf.taxonomy.layers:
        lo = output.layers.get(layer.id)
        top = lo.top if lo else None
        label = top.label if top else "—"
        evidence = ", ".join(top.evidence) if (top and top.evidence) else "—"
        flag = "  ⚠️ ambiguous (needs a follow-up question)" if (lo and lo.needs_clarification) else ""
        print(f"  • {layer.id} ({layer.name}): {label}   [evidence: {evidence}]{flag}")

    print(f"\nSuggested summary: {output.suggested_summary}")
    print(
        f"LLM metadata → model={meta.get('model')}  "
        f"latency={meta.get('latency_ms')}ms  usage={meta.get('usage')}"
    )
    print("\n✅ OpenRouter call succeeded and a valid JSON response was received.")


if __name__ == "__main__":
    main()
