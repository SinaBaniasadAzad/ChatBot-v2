"""
Output Parser — turn the raw LLM output into a valid, trustworthy structure.

Why is this needed? JSON mode only guarantees JSON *syntax*, not that the
output matches our schema and taxonomy. The model may:
  - return a label that does not exist (hallucination),
  - drop a layer or add an extra key,
  - return the wrong type (e.g. evidence as a string instead of a list).
This module has two layers of defense:
  1) Defensive pre-normalization of the raw dict (coerce types/structure).
  2) Semantic validation against the taxonomy (drop fake labels, fill missing layers).
Golden rule: this function never crashes; in the worst case it marks a layer
as "ambiguous" so the conversation can ask a question, rather than taking the
service down.
"""
from __future__ import annotations

from src.classifier.schema import Candidate, ClassificationOutput, LayerOutput
from src.taxonomy import Taxonomy
from src.utils.logging import get_logger

log = get_logger("output_parser")


def _coerce_raw(raw: dict) -> dict:
    """Defensively coerce common malformed types/structures before Pydantic validation."""
    if not isinstance(raw, dict):
        return {"layers": {}}

    layers_in = raw.get("layers")
    if not isinstance(layers_in, dict):
        layers_in = {}

    norm_layers: dict = {}
    for lid, lo in layers_in.items():
        if not isinstance(lo, dict):
            continue
        cands = lo.get("candidates")
        if isinstance(cands, dict):       # model returned a single candidate instead of a list
            cands = [cands]
        if not isinstance(cands, list):
            cands = []

        norm_cands = []
        for c in cands:
            if isinstance(c, str):        # only the label string was given
                c = {"label": c, "evidence": []}
            if not isinstance(c, dict):
                continue
            ev = c.get("evidence", [])
            if isinstance(ev, str):       # evidence as a string instead of a list
                ev = [ev]
            if not isinstance(ev, list):
                ev = []
            norm_cands.append(
                {"label": str(c.get("label", "")).strip(), "evidence": [str(x) for x in ev]}
            )

        norm_layers[lid] = {
            "candidates": norm_cands,
            "needs_clarification": bool(lo.get("needs_clarification", False)),
        }

    return {
        "layers": norm_layers,
        "clarifying_question": raw.get("clarifying_question") or None,
        "suggested_summary": str(raw.get("suggested_summary") or ""),
        "reasoning": str(raw.get("reasoning") or ""),
    }


def parse_and_validate(raw: dict, tax: Taxonomy) -> ClassificationOutput:
    """
    Convert the raw model output into a valid ClassificationOutput:
    - unknown layers are ignored.
    - labels outside the allowed set (hallucinations) are dropped.
    - every layer defined in the taxonomy is guaranteed at least one valid
      candidate; if none is valid, that layer is marked "ambiguous".
    """
    try:
        out = ClassificationOutput.model_validate(_coerce_raw(raw))
    except Exception as e:  # never crash
        log.warning("Model-output validation failed; assuming all layers ambiguous: %s", e)
        out = ClassificationOutput()

    cleaned: dict[str, LayerOutput] = {}
    for layer in tax.layers:
        allowed = set(layer.label_ids)
        lo = out.layers.get(layer.id) or LayerOutput()
        valid = [c for c in lo.candidates if c.label in allowed]
        needs = lo.needs_clarification

        if not valid:  # no valid label -> ambiguous instead of crashing
            valid = [Candidate(label=lid, evidence=[]) for lid in layer.label_ids]
            needs = True

        cleaned[layer.id] = LayerOutput(candidates=valid, needs_clarification=needs)

    out.layers = cleaned
    return out


def label_key(raw: dict, tax: Taxonomy) -> tuple:
    """Comparison key for self-consistency: the tuple of each layer's top label."""
    try:
        parsed = parse_and_validate(raw, tax)
        return tuple(
            (lid, lo.top.label if lo.top else None) for lid, lo in parsed.layers.items()
        )
    except Exception:
        return ("invalid",)
