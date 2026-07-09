"""
ممیزیِ کیفیتِ retrieval روی مسیرِ واقعیِ production — «آیا همسایه‌های درست انتخاب می‌شوند؟»

برخلافِ بنچمارکِ آفلاینِ embedding (src/retrieval/bench.py که مدل‌ها را مقایسه
می‌کند)، این اسکریپت *همان TicketRetrieverِ production* را با *همان ایندکس* و
*همان پارامترها* (k_demos, purity_k, sim_floor) صدا می‌زند و می‌سنجد:

  • neighbor agreement@k   سهمِ همسایه‌های تزریق‌شده که هم‌برچسبِ طلاییِ کوئری‌اند
  • kNN vote accuracy      آیا رایِ همسایگی با برچسبِ طلایی یکی است؟
  • purity calibration     در هر سطلِ purity، رای چند درصد درست است؟
                           (اعتبارسنجیِ آستانهٔ KNN_DISAGREE_PURITY=0.80)
  • abstain rate + دلایل   کناره‌گیری (زیرِ sim_floor / کوئریِ خالی)
  • worst offenders        کوئری‌هایی با شباهتِ بالا ولی همسایگیِ غلط → کاندیدِ
                           نویزِ برچسب یا ضعفِ ایندکس؛ قابلِ‌ارسال به صفِ بازبینی

گاردِ نشت: خودِ تیکت (exclude_keys) و شبه‌خودی‌ها (sim>=0.995) حذف می‌شوند —
دقیقاً مثلِ eval_incdb.

نیازمندی: ایندکسِ retrieval و وابستگی‌های embedding (اجرا روی همان ماشینی که
ایندکس ساخته شده). اجرا:
    python -m scripts.eval_retrieval tests/Ticketing_DB.jsonl --frac 0.2 --seed 42 \
        --out retrieval_audit.json --review-out retrieval_worst.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_incdb import _build_maps, _field, _gt_label, load_rows  # noqa: E402
from src.retrieval.retriever import maybe_build_retriever  # noqa: E402
from src.taxonomy import load_taxonomy  # noqa: E402

PURITY_BUCKETS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.01)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_path", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--balanced", type=int, default=None)
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None, help="گزارشِ JSON کامل")
    ap.add_argument("--review-out", type=Path, default=None,
                    help="worst offenders به JSONL (خوراکِ scripts.import_review_items)")
    ap.add_argument("--worst-agreement", type=float, default=0.5,
                    help="آستانهٔ agreement برای worst offender (پیش‌فرض <0.5)")
    ap.add_argument("--enqueue", action="store_true",
                    help="worst offenders را مستقیم واردِ صفِ بازبینی کن")
    args = ap.parse_args()

    retriever = maybe_build_retriever()
    if retriever is None:
        raise SystemExit(
            "بازیاب در دسترس نیست: ایندکس (python -m scripts.build_retrieval_index) و "
            "وابستگی‌های embedding لازم‌اند."
        )

    tax = load_taxonomy()
    _, label_map = _build_maps(tax)
    rows = load_rows(args.data_path, args.limit, args.balanced, tax, frac=args.frac, seed=args.seed)
    layer_ids = [layer.id for layer in tax.layers]
    print(f"{len(rows)} تیکت | ایندکس: {len(retriever.rows)} | مدل: {retriever.model_name}",
          file=sys.stderr)

    agreement_sum = {lid: 0.0 for lid in layer_ids}
    agreement_n = {lid: 0 for lid in layer_ids}
    vote_ok = {lid: 0 for lid in layer_ids}
    vote_n = {lid: 0 for lid in layer_ids}
    purity_cal = {lid: defaultdict(lambda: [0, 0]) for lid in layer_ids}  # bucket -> [ok, n]
    abstains: Counter = Counter()
    sims: list[float] = []
    worst: list[dict] = []
    n_eval = 0
    explain: dict = {}

    for i, row in enumerate(rows, 1):
        key = str(row.get("Key") or "")
        gold = {lid: _gt_label(row, tax.get_layer(lid), label_map) for lid in layer_ids}
        if not all(gold.values()):
            continue
        n_eval += 1
        explain: dict = {}
        res = retriever.retrieve(
            _field(row, "Summary", "summary"),
            _field(row, "Description", "description"),
            exclude_keys=frozenset({key}),
            drop_self_sim=0.995,
            explain=explain,
        )
        if res is None:
            abstains[explain.get("abstain_reason", "unknown")] += 1
            continue
        sims.append(res.top_similarity)

        row_agreements = {}
        for lid in layer_ids:
            labels = [d.get(lid) for d in res.demos if d.get(lid)]
            if labels:
                a = sum(int(l == gold[lid]) for l in labels) / len(labels)
                agreement_sum[lid] += a
                agreement_n[lid] += 1
                row_agreements[lid] = a
            vote = res.votes.get(lid)
            if vote:
                vote_n[lid] += 1
                ok = int(vote.label == gold[lid])
                vote_ok[lid] += ok
                for b in PURITY_BUCKETS:
                    if vote.purity < b:
                        purity_cal[lid][b][0] += ok
                        purity_cal[lid][b][1] += 1
                        break

        min_agree = min(row_agreements.values(), default=1.0)
        if min_agree < args.worst_agreement:
            worst.append({
                "Key": key,
                "Summary": _field(row, "Summary", "summary"),
                "Description": _field(row, "Description", "description"),
                "gold": gold,
                "agreement": {k: round(v, 3) for k, v in row_agreements.items()},
                "top_similarity": round(res.top_similarity, 4),
                "neighbors": [
                    {"key": d["key"], "sim": round(s, 4),
                     **{lid: d.get(lid) for lid in layer_ids}}
                    for d, s in zip(res.demos, res.demo_sims)
                ],
                "votes": res.votes_dict(),
            })
        if i % 50 == 0:
            print(f"... {i}/{len(rows)}", file=sys.stderr)

    n_ret = n_eval - sum(abstains.values())
    report = {
        "n_tickets": n_eval,
        "n_retrieved": n_ret,
        "abstain_rate": round(sum(abstains.values()) / n_eval, 4) if n_eval else 0.0,
        "abstain_reasons": dict(abstains),
        "avg_top_similarity": round(sum(sims) / len(sims), 4) if sims else None,
        "params": {"k_demos": explain.get("k_demos"), "purity_k": explain.get("purity_k"),
                   "sim_floor": explain.get("sim_floor"), "model": retriever.model_name},
        "layers": {},
        "n_worst_offenders": len(worst),
    }
    for lid in layer_ids:
        cal = []
        for b in PURITY_BUCKETS:
            ok, n = purity_cal[lid][b]
            lo = PURITY_BUCKETS[PURITY_BUCKETS.index(b) - 1] if PURITY_BUCKETS.index(b) else 0.0
            cal.append({"purity_bucket": f"[{lo:.1f},{min(b,1.0):.1f})",
                        "n": n, "vote_accuracy": round(ok / n, 4) if n else None})
        report["layers"][lid] = {
            "neighbor_agreement@k": round(agreement_sum[lid] / agreement_n[lid], 4)
            if agreement_n[lid] else None,
            "knn_vote_accuracy": round(vote_ok[lid] / vote_n[lid], 4) if vote_n[lid] else None,
            "purity_calibration": cal,
        }

    print("\n===== ممیزیِ کیفیتِ retrieval =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        args.out.write_text(json.dumps({**report, "worst_offenders": worst},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"گزارشِ کامل: {args.out}")
    if args.review_out and worst:
        with args.review_out.open("w", encoding="utf-8") as f:
            for w in worst:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
        print(f"{len(worst)} worst offender → {args.review_out}")
    if args.enqueue and worst:
        from src.review.store import maybe_build_review_store

        store = maybe_build_review_store()
        if store is None:
            print("صفِ بازبینی غیرفعال است (REVIEW_QUEUE_ENABLED).", file=sys.stderr)
        else:
            n = 0
            for w in worst:
                item_id = store.enqueue(
                    source="retrieval_audit",
                    ticket_key=w["Key"],
                    summary=w["Summary"],
                    description=w["Description"],
                    predicted_labels={},
                    notes=f"neighbor agreement={w['agreement']}, sim={w['top_similarity']}; "
                          f"همسایه‌ها: {[nb['key'] for nb in w['neighbors']]}",
                )
                n += int(item_id is not None)
          python  print(f"{n} مورد واردِ صفِ بازبینی شد (source=retrieval_audit).")


if __name__ == "__main__":
    main()
