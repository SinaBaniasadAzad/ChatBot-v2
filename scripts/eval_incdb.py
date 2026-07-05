"""
Accuracy evaluation on the raw ticket file (INC_DB.jsonl format).

Each line:
  {"Key": "INC-20689", "Application": "ERP", "Summary": "...", "Description": "...",
   "Labels": {"layer_1": "Incident", "layer_2": "ERP"}}

- ground truth = Labels.layer_1 / layer_2 (display names like "Incident"/"ERP").
  Names are auto-mapped to internal ids and "layer_1" is matched to taxonomy "layer1"
  (underscores removed).
- The model runs single-shot (no clarifying questions) = "raw model accuracy".

This module exposes:
  • run_evaluation(...) -> dict     metric computation (for text + dashboard)
  • main()                          CLI text report

Sampling options:
  --balanced N   at most N tickets per (layer1,layer2) combo
  --frac F       fraction F (e.g. 0.2 = 20%) sampled from EACH combo (stratified, seeded)

Professional visual dashboard: scripts/report.py

Run (from project root):
    python -m scripts.eval_incdb data/INC_DB.jsonl --frac 0.2 --workers 6
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.classifier import Classifier  # noqa: E402
from src.classifier.decision import _is_ambiguous, decide  # noqa: E402


def _norm_key(k: str) -> str:
    """"layer_1" / "Layer 1" -> "layer1" (alphanumerics only)."""
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


def _cap(name: str) -> str:
    """Short English display name: "Type / نوع" -> "Type"."""
    return name.split("/")[0].strip() if "/" in name else name.strip()


def _build_maps(tax):
    """name->id maps for layers and labels (accepts display name or id)."""
    layer_key_map: dict[str, str] = {}
    label_map: dict[str, dict[str, str]] = {}
    for layer in tax.layers:
        layer_key_map[_norm_key(layer.id)] = layer.id
        m: dict[str, str] = {}
        for lbl in layer.labels:
            m[lbl.name.strip().lower()] = lbl.id
            m[lbl.id.strip().lower()] = lbl.id
        label_map[layer.id] = m
    return layer_key_map, label_map


def _gt_label(row: dict, layer, label_map) -> str | None:
    """Gold label id of this layer from Labels (or None if missing/unmapped)."""
    labels = row.get("Labels") or {}
    for gk, gv in labels.items():
        if _norm_key(gk) == _norm_key(layer.id):
            return label_map[layer.id].get(str(gv).strip().lower())
    return None


def _combo(row: dict, tax, label_map) -> tuple:
    return tuple(_gt_label(row, layer, label_map) for layer in tax.layers)


def load_rows(
    path: Path,
    limit: int | None,
    balanced: int | None,
    tax,
    frac: float | None = None,
    seed: int = 42,
) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    if balanced or frac:
        # Group by (layer1,layer2) combo, then sample per combo.
        _, label_map = _build_maps(tax)
        buckets: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            buckets[_combo(r, tax, label_map)].append(r)
        rng = random.Random(seed)
        rows = []
        for items in buckets.values():
            if frac:  # fraction (e.g. 20%) of EACH combo, random but reproducible
                k = max(1, round(len(items) * frac))
                rows.extend(rng.sample(items, min(k, len(items))))
            else:     # at most N per combo
                rows.extend(items[:balanced])

    return rows[:limit] if limit else rows


def _field(row: dict, *names: str) -> str:
    for n in names:
        if row.get(n):
            return str(row[n])
    return ""


def compute_cost(tokens: dict, price_in: float, price_cache: float, price_out: float) -> float:
    """Dollar cost from token counts (prices per 1M tokens)."""
    return (
        tokens.get("cache_hit", 0) * price_cache
        + tokens.get("cache_miss", 0) * price_in
        + tokens.get("completion", 0) * price_out
    ) / 1_000_000


def run_evaluation(
    data_path,
    *,
    limit: int | None = None,
    balanced: int | None = None,
    frac: float | None = None,
    seed: int = 42,
    workers: int = 4,
    out_path=None,
    errors_out=None,
    progress: bool = True,
    use_retrieval: bool = True,
) -> dict:
    """Run the model over the tickets and return metrics (no display)."""
    clf = Classifier() if use_retrieval else Classifier(retriever=None)
    tax = clf.taxonomy
    _, label_map = _build_maps(tax)
    rows = load_rows(Path(data_path), limit, balanced, tax, frac=frac, seed=seed)
    total = len(rows)

    if progress:
        print(f"Loaded {total} tickets from {data_path}", file=sys.stderr)
        dist: dict[tuple, int] = defaultdict(int)
        for r in rows:
            dist[_combo(r, tax, label_map)] += 1
        for combo, n in sorted(dist.items(), key=lambda x: str(x[0])):
            names = " + ".join(
                (tax.get_layer(L.id).get_label(cid).name if cid and tax.get_layer(L.id).get_label(cid) else str(cid))
                for L, cid in zip(tax.layers, combo)
            )
            print(f"  {names}: {n}", file=sys.stderr)

    per_layer_total: dict[str, int] = defaultdict(int)
    per_layer_correct: dict[str, int] = defaultdict(int)
    cls_total = {l.id: defaultdict(int) for l in tax.layers}
    cls_correct = {l.id: defaultdict(int) for l in tax.layers}
    confusion = {l.id: defaultdict(lambda: defaultdict(int)) for l in tax.layers}
    full_total = full_correct = errors = 0
    conf = {"confident_total": 0, "confident_correct": 0, "flagged_total": 0, "flagged_correct": 0}
    # همان بازی با گِیتِ جدید (شواهدِ راستی‌آزمایی‌شده + مخالفتِ kNN) — مسیرِ واقعیِ production
    conf_gated = {"confident_total": 0, "confident_correct": 0, "flagged_total": 0, "flagged_correct": 0}
    tok: dict[str, int] = defaultdict(int)
    model_served = None
    t0 = time.perf_counter()
    out_fh = Path(out_path).open("w", encoding="utf-8") if out_path else None
    err_fh = Path(errors_out).open("w", encoding="utf-8") if errors_out else None
    n_wrong = 0

    if clf.retriever is not None:
        print("retrieval: ON (self-key excluded per ticket, near-self >=0.995 dropped)", file=sys.stderr)

    def run_one(row: dict):
        try:
            # گاردِ نشت: خودِ تیکت (و شبه‌خودی‌ها) هرگز به‌عنوانِ سابقهٔ خودش بازیابی نشود.
            out, meta = clf.classify(
                _field(row, "Summary", "summary"),
                _field(row, "Description", "description"),
                exclude_keys=frozenset({str(row.get("Key"))}),
            )
            return row, out, meta, None
        except Exception as e:  # one failing ticket must not kill the whole run
            return row, None, {}, str(e)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, out, meta, err in pool.map(run_one, rows):
            done += 1
            if err:
                errors += 1
            if meta.get("model"):
                model_served = meta["model"]

            u = meta.get("usage") or {}
            pt, ct = u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0
            hit, miss = u.get("prompt_cache_hit_tokens"), u.get("prompt_cache_miss_tokens")
            if hit is None or miss is None:
                hit, miss = 0, pt
            tok["prompt"] += pt
            tok["cache_hit"] += hit
            tok["cache_miss"] += miss
            tok["completion"] += ct

            row_all_ok = True
            row_has_all = True
            rec = {"Key": row.get("Key"), "true": {}, "pred": {}}
            for layer in tax.layers:
                true_id = _gt_label(row, layer, label_map)
                if true_id is None:
                    row_has_all = False
                    continue
                lo = out.layers.get(layer.id) if out else None
                pred_id = lo.top.label if (lo and lo.top) else None
                per_layer_total[layer.id] += 1
                cls_total[layer.id][true_id] += 1
                confusion[layer.id][true_id][pred_id] += 1
                if pred_id == true_id:
                    per_layer_correct[layer.id] += 1
                    cls_correct[layer.id][true_id] += 1
                else:
                    row_all_ok = False
                rec["true"][layer.id] = true_id
                rec["pred"][layer.id] = pred_id
            if row_has_all:
                full_total += 1
                full_correct += int(row_all_ok)
                # Was the model "confident"? (no ambiguous layer -> auto-classifiable)
                flagged = True
                if out is not None:
                    flagged = any(
                        out.layers.get(L.id) is None or _is_ambiguous(out.layers.get(L.id))
                        for L in tax.layers
                    )
                bucket = "flagged" if flagged else "confident"
                conf[bucket + "_total"] += 1
                conf[bucket + "_correct"] += int(row_all_ok)
                # گِیتِ جدید: با max_questions=0 هر ابهامی مستقیم needs_review می‌شود؛
                # این دقیقاً همان تصمیمی است که production (قبل از پرسیدنِ سوال) می‌گیرد.
                gated_flagged = True
                if out is not None:
                    ticket_text = (
                        _field(row, "Summary", "summary")
                        + "\n"
                        + _field(row, "Description", "description")
                    )
                    gated_flagged = decide(
                        out, tax, 0, 0,
                        ticket_text=ticket_text, knn_votes=meta.get("knn_votes"),
                    ).needs_review
                gbucket = "flagged" if gated_flagged else "confident"
                conf_gated[gbucket + "_total"] += 1
                conf_gated[gbucket + "_correct"] += int(row_all_ok)
            if out_fh:
                rec["correct"] = row_all_ok and row_has_all
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Error log: only tickets with at least one wrong layer, with their text.
            wrong_layers = [lid for lid in rec["true"] if rec["pred"].get(lid) != rec["true"][lid]]
            if err_fh and wrong_layers:
                n_wrong += 1
                err_fh.write(
                    json.dumps(
                        {
                            "Key": row.get("Key"),
                            "wrong": wrong_layers,
                            "Summary": _field(row, "Summary", "summary"),
                            "Description": _field(row, "Description", "description"),
                            **{lid: {"true": rec["true"][lid], "pred": rec["pred"].get(lid)} for lid in rec["true"]},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if progress and done % 25 == 0:
                print(f"... {done}/{total}", file=sys.stderr)

    if out_fh:
        out_fh.close()
    if err_fh:
        err_fh.close()
        if progress:
            print(f"Wrote {n_wrong} misclassified tickets to {errors_out}", file=sys.stderr)
    wall = time.perf_counter() - t0

    layers_out = []
    for layer in tax.layers:
        lid = layer.id
        classes = []
        for cid in layer.label_ids:
            t = cls_total[lid].get(cid, 0)
            c = cls_correct[lid].get(cid, 0)
            lbl = layer.get_label(cid)
            classes.append(
                {"id": cid, "name": lbl.name if lbl else cid, "recall": (c / t if t else 0.0), "correct": c, "total": t}
            )
        tot = per_layer_total[lid]
        layers_out.append(
            {
                "id": lid,
                "name": _cap(layer.name),
                "accuracy": (per_layer_correct[lid] / tot if tot else 0.0),
                "correct": per_layer_correct[lid],
                "total": tot,
                "label_ids": list(layer.label_ids),
                "classes": classes,
                "confusion": {t: dict(confusion[lid][t]) for t in confusion[lid]},
            }
        )

    return {
        "n": total,
        "errors": errors,
        "model": model_served,
        "layers": layers_out,
        "overall": {
            "accuracy": (full_correct / full_total if full_total else 0.0),
            "correct": full_correct,
            "total": full_total,
        },
        "tokens": dict(tok),
        "confidence": conf,
        "confidence_gated": conf_gated,
        "retrieval_used": clf.retriever is not None,
        "wall_s": wall,
        "latency_ms_avg": (wall * 1000 / total if total else 0.0),
    }


def print_text_report(res: dict, price_in: float, price_cache: float, price_out: float) -> None:
    cost = compute_cost(res["tokens"], price_in, price_cache, price_out)
    t = res["tokens"]
    print("\n================= Evaluation results (single-shot) =================")
    print(f"Model: {res.get('model')}   |   Tickets: {res['n']}   |   Call errors: {res['errors']}")
    print(f"Tokens -> prompt={t.get('prompt',0):,} (hit={t.get('cache_hit',0):,}/miss={t.get('cache_miss',0):,})  completion={t.get('completion',0):,}")
    print(f"Estimated cost: ${cost:.4f}  (price/1M: in=${price_in}, hit=${price_cache}, out=${price_out} - verify current pricing)")
    print(f"Total time: {res['wall_s']:.0f}s  |  Avg: {res['latency_ms_avg']:.0f} ms/ticket")

    print("\n-- Per-layer accuracy --")
    for L in res["layers"]:
        print(f"  {L['id']} ({L['name']}): {L['accuracy']:.1%}  ({L['correct']}/{L['total']})")
    print(f"\n-- Overall accuracy (both layers correct) --\n  {res['overall']['accuracy']:.1%}  ({res['overall']['correct']}/{res['overall']['total']})")

    c = res.get("confidence", {})
    ct, cc = c.get("confident_total", 0), c.get("confident_correct", 0)
    ft, fc = c.get("flagged_total", 0), c.get("flagged_correct", 0)
    tot = ct + ft
    if tot:
        print("\n-- Accuracy by confidence (legacy gate: self-report + evidence presence) --")
        print(f"  Confident / auto:   coverage {ct/tot:.0%} ({ct})  |  accuracy {cc/ct:.1%}" if ct else "  Confident: 0")
        print(f"  Needs review / ask: {ft/tot:.0%} ({ft})  |  accuracy {fc/ft:.1%}" if ft else "  Needs review: 0")

    g = res.get("confidence_gated", {})
    gct, gcc = g.get("confident_total", 0), g.get("confident_correct", 0)
    gft, gfc = g.get("flagged_total", 0), g.get("flagged_correct", 0)
    gtot = gct + gft
    if gtot:
        print("\n-- Accuracy by NEW confidence gate (verified evidence + kNN precedent) --")
        print(f"  Auto-routable:      coverage {gct/gtot:.0%} ({gct})  |  accuracy {gcc/gct:.1%}" if gct else "  Auto-routable: 0")
        print(f"  Would ask/flag:     {gft/gtot:.0%} ({gft})  |  accuracy {gfc/gft:.1%}" if gft else "  Would ask/flag: 0")

    print("\n-- Per-class recall --")
    for L in res["layers"]:
        print(f"  {L['id']}:")
        for c in L["classes"]:
            print(f"    {c['name']:<18} {c['recall']:6.1%}  ({c['correct']}/{c['total']})")

    for L in res["layers"]:
        ids = L["label_ids"]
        name = {c["id"]: c["name"] for c in L["classes"]}
        print(f"\n-- Confusion '{L['id']}' (row=true, col=pred; none=no prediction) --")
        print("true\\pred".ljust(18) + "".join(name[p][:16].ljust(18) for p in ids) + "none")
        for tr in ids:
            cells = L["confusion"].get(tr, {})
            print(name[tr].ljust(18) + "".join(str(cells.get(p, 0)).ljust(18) for p in ids) + str(cells.get(None, 0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--balanced", type=int, default=None, help="at most N tickets per label combo")
    ap.add_argument("--frac", type=float, default=None, help="fraction per combo, e.g. 0.2 = 20%%")
    ap.add_argument("--seed", type=int, default=42, help="random seed for --frac sampling")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="save all predictions (JSONL)")
    ap.add_argument("--errors", type=Path, default=None, help="save only misclassified tickets + text (JSONL)")
    ap.add_argument("--price-in", type=float, default=0.14)
    ap.add_argument("--price-cache", type=float, default=0.0028)
    ap.add_argument("--price-out", type=float, default=0.28)
    ap.add_argument("--no-retrieval", action="store_true",
                    help="اجرای پایه بدونِ سابقه/گِیتِ kNN (برای مقایسهٔ A/B)")
    args = ap.parse_args()

    res = run_evaluation(
        args.data_path, limit=args.limit, balanced=args.balanced, frac=args.frac, seed=args.seed,
        workers=args.workers, out_path=args.out, errors_out=args.errors,
        use_retrieval=not args.no_retrieval,
    )
    print_text_report(res, args.price_in, args.price_cache, args.price_out)


if __name__ == "__main__":
    main()
