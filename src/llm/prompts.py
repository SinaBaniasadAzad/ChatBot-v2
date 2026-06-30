"""
Build the System/User prompts from the taxonomy + few-shot examples.

Key point: everything is generated dynamically from the taxonomy; no label
name is hard-coded in this file. Adding a layer/label in the YAML flows into
the prompt automatically.
"""
from __future__ import annotations

import json

from src.taxonomy import Taxonomy

_OUTPUT_RULES = """\
You are an expert IT-ticket triage assistant for an Incident/ServiceRequest or ERP/Staff ticketing system.
Tickets may be written in Persian, English, or a mix of both. Do NOT translate —
reason in the original language of the ticket.

Your job: for EACH layer below, choose the single best label from that layer's
allowed set, using ONLY concrete evidence found in the ticket text.

Decision principles:
- FIRST, fill the "reasoning" field (one or two sentences): decide (a) which DOMAIN/system
  the ticket belongs to, and (b) whether something that should work is BROKEN/missing
  (Incident) versus the user requesting something NEW or standard to be provided
  (Service Request). Then choose labels that are consistent with this reasoning.
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


def _label_name(tax: Taxonomy, label_id: str) -> str:
    """Find a label's display name in whichever layer it lives in."""
    for layer in tax.layers:
        lbl = layer.get_label(label_id)
        if lbl:
            return lbl.name
    return str(label_id)


def _fmt_kw(item) -> str:
    if isinstance(item, (list, tuple)):
        kw = item[0]
        lvl = item[1] if len(item) > 1 else ""
        return f"{kw}[{lvl}]" if lvl != "" else f"{kw}"
    return str(item)


def build_signals_block(tax: Taxonomy) -> str:
    """Expert keyword map: each keyword determines both Domain and Type at once."""
    sig = getattr(tax, "signals", None) or {}
    if not sig:
        return ""
    lines = [
        "== Expert keyword signals ==",
        "Each keyword below points to BOTH a Domain and a Type at the same time; treat it as "
        "strong evidence for those two labels. The number in [..] is the signal strength/priority "
        "(higher = more decisive). If keywords point to DIFFERENT types, prefer the higher-strength "
        "one. These keywords are hints, not absolute rules — still respect each layer's definition.",
        "",
    ]
    for grp in sig.get("domain_type", []):
        dom = _label_name(tax, grp.get("domain"))
        typ = _label_name(tax, grp.get("type"))
        kws = "، ".join(_fmt_kw(k) for k in grp.get("keywords", []))
        lines.append(f"  {dom}  ·  {typ}:  {kws}")
    for grp in sig.get("type_general", []):
        typ = _label_name(tax, grp.get("type"))
        kws = "، ".join(_fmt_kw(k) for k in grp.get("keywords", []))
        lines.append(f"  General {typ} (any domain):  {kws}")
    rules = sig.get("rules", [])
    if rules:
        lines += ["", "Special rules:"] + [f"  - {r}" for r in rules]
    return "\n".join(lines)


def _schema_block(tax: Taxonomy) -> str:
    """Example JSON output with the layer keys and each layer's allowed labels."""
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
        "reasoning": "<FIRST think: which domain/system, and is something broken (Incident) or a new/standard thing requested (Service Request)>",
        "layers": layers_obj,
        "clarifying_question": None,
        "suggested_summary": "<short clean summary>",
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
        build_signals_block(tax),
        _schema_block(tax),
    ]
    blocks = [b for b in blocks if b]
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
