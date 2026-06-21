"""
ساخت System/User prompt از روی taxonomy + few-shot.

نکتهٔ کلیدی: همه‌چیز به‌صورت پویا از taxonomy ساخته می‌شود؛ هیچ نام دسته‌ای
در این فایل hard-code نشده. افزودن لایه/دسته در YAML خودکار به prompt می‌آید.
"""
from __future__ import annotations

import json

from src.taxonomy import Taxonomy

_OUTPUT_RULES = """\
You are an expert IT-ticket triage assistant for an HR/ERP ticketing system.
Tickets may be written in Persian, English, or a mix of both. Do NOT translate —
reason in the original language of the ticket.

Your job: for EACH layer below, choose the single best label from that layer's
allowed set, using ONLY concrete evidence found in the ticket text.

Decision principles:
- Every decision must be grounded in specific words/phrases ("evidence") copied
  from the ticket. If a candidate label has no textual support, give it an empty
  evidence list.
- For each layer, return your TOP-2 candidate labels (best first), each with its
  evidence spans.
- A layer "needs clarification" ONLY when the text does not contain enough
  discriminating evidence to choose confidently between its top candidates.
  Set needs_clarification=true for that layer in that case.
- Do NOT ask for clarification just because you feel unsure. Ask only when the
  needed information is genuinely MISSING from the text. If you can cite clear
  evidence, do not ask.
- If ANY layer needs clarification, provide exactly ONE short, targeted
  clarifying_question, written in the SAME language as the ticket, that best
  separates the competing candidates. Prefer one question that helps the most.
- Also produce a clean, short suggested_summary of the underlying issue.
"""


def _layers_block(tax: Taxonomy) -> str:
    parts: list[str] = []
    for layer in tax.layers:
        parts.append(f'== LAYER "{layer.id}" — {layer.name} ==')
        if layer.description:
            parts.append(layer.description.strip())
        for lbl in layer.labels:
            cues = "، ".join(lbl.cues) if lbl.cues else "-"
            parts.append(
                f'  • "{lbl.id}" ({lbl.name}): {lbl.definition.strip()}\n'
                f"    cues: {cues}"
            )
        parts.append("")
    return "\n".join(parts)


def _schema_block(tax: Taxonomy) -> str:
    """نمونهٔ خروجی JSON با کلیدهای لایه‌ها و فهرست برچسب‌های مجاز هر لایه."""
    layers_obj = {}
    allowed_note = []
    for layer in tax.layers:
        allowed_note.append(f'    "{layer.id}" allowed labels: {layer.label_ids}')
        layers_obj[layer.id] = {
            "candidates": [
                {"label": "<one allowed label id>", "evidence": ["<span from text>"]},
                {"label": "<the runner-up label id>", "evidence": []},
            ],
            "needs_clarification": False,
        }
    skeleton = {
        "layers": layers_obj,
        "clarifying_question": None,
        "suggested_summary": "<short clean summary>",
        "reasoning": "<one short sentence, for logs>",
    }
    return (
        "Respond with ONLY a valid JSON object (no markdown, no extra text) of this shape:\n"
        + json.dumps(skeleton, ensure_ascii=False, indent=2)
        + "\n\nConstraints:\n"
        + "\n".join(allowed_note)
        + "\n    Use the layer ids above as the keys of \"layers\"."
    )


def build_system_prompt(tax: Taxonomy, demonstrations: list[dict]) -> str:
    blocks = [
        _OUTPUT_RULES,
        "Classification layers:\n",
        _layers_block(tax),
        _schema_block(tax),
    ]
    if demonstrations:
        ex_lines = ["\n== Examples (input -> ideal JSON output) =="]
        for demo in demonstrations:
            ex_lines.append("INPUT:")
            ex_lines.append(json.dumps(demo["input"], ensure_ascii=False))
            ex_lines.append("OUTPUT:")
            ex_lines.append(json.dumps(demo["output"], ensure_ascii=False))
            ex_lines.append("")
        blocks.append("\n".join(ex_lines))
    return "\n".join(blocks)


def build_user_prompt(
    summary: str,
    description: str,
    clarifications: list[tuple[str, str]] | None = None,
) -> str:
    lines = ["Ticket:", f"Summary: {summary}", f"Description: {description}"]
    if clarifications:
        lines.append("\nClarifications (follow-up Q&A):")
        for q, a in clarifications:
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}")
    lines.append("\nClassify now. Respond with JSON only.")
    return "\n".join(lines)
