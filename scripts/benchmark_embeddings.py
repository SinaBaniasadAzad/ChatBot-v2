"""بنچمارکِ embedding — اجرای هر مدل در یک سلولِ جدا + گزارشِ مقایسه.

اجرا (از ریشهٔ پروژه؛ راهنمای کامل: docs/embedding_benchmark.md):
    python -m scripts.benchmark_embeddings --model bm25
    python -m scripts.benchmark_embeddings --model e5-large
    python -m scripts.benchmark_embeddings --model bge-m3
    python -m scripts.benchmark_embeddings --model qwen3-0.6b
    python -m scripts.benchmark_embeddings --report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.bench import REGISTRY, run_model  # noqa: E402

DEFAULT_DATA = Path("data/retrieval/tickets_clean.jsonl")
DEFAULT_OUT = Path("data/retrieval/results")


def _fmt(v, pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}" if pct else str(v)


def _frontier_at(metrics: dict, layer: str, thr: float) -> str:
    for row in metrics[layer]["purity_frontier"]:
        if abs(row["threshold"] - thr) < 1e-9:
            acc = _fmt(row["accuracy"])
            cov = _fmt(row["coverage"])
            return f"{cov}%@{acc}%"
    return "—"


def print_summary(result: dict) -> None:
    print(f"\n===== {result['model']}  (n={result['n_tickets']}) =====")
    t = result["timing"]
    print(
        f"device={t['device']}  load={t['load_s']}s  encode={t['encode_s']}s "
        f"({t['tickets_per_s']}/s)  query={t['query_latency_ms']}ms  "
        f"vram={t['vram_peak_mb']}MB  dim={t['dim']}"
    )
    for v in result["variants"]:
        m = v["metrics"]
        line = (
            f"  {v['variant']:<12} "
            f"L1 acc@10={_fmt(m['layer1']['knn_acc@10'])}%  "
            f"L2 acc@10={_fmt(m['layer2']['knn_acc@10'])}%  "
            f"combo={_fmt(m['combo']['knn_acc@10'])}%  "
            f"L1 agree@5={_fmt(m['layer1']['agreement']['5'])}%  "
            f"L1 frontier@0.8: {_frontier_at(m, 'layer1', 0.8)}"
        )
        if "contradiction_pairs" in v:
            line += f"  contradictions={v['contradiction_pairs']}"
        print(line)


def report(out_dir: Path) -> None:
    rows = []
    for f in sorted(out_dir.glob("*.json")):
        res = json.loads(f.read_text(encoding="utf-8"))
        for v in res.get("variants", []):
            m = v["metrics"]
            rows.append(
                {
                    "model": res["model"],
                    "variant": v["variant"],
                    "L1_acc@10": m["layer1"]["knn_acc@10"],
                    "L2_acc@10": m["layer2"]["knn_acc@10"],
                    "combo@10": m["combo"]["knn_acc@10"],
                    "L1_agree@5": m["layer1"]["agreement"]["5"],
                    "L2_agree@5": m["layer2"]["agreement"]["5"],
                    "L1_frontier@0.8": _frontier_at(m, "layer1", 0.8),
                    "tickets/s": res["timing"]["tickets_per_s"],
                    "query_ms": res["timing"]["query_latency_ms"],
                    "vram_mb": res["timing"]["vram_peak_mb"],
                }
            )
    if not rows:
        print(f"هیچ نتیجه‌ای در {out_dir} نیست — اول مدل‌ها را اجرا کنید.")
        return

    rows.sort(key=lambda r: -r["combo@10"])
    pct_cols = {"L1_acc@10", "L2_acc@10", "combo@10", "L1_agree@5", "L2_agree@5"}
    formatted = [
        {c: _cell(r[c], as_pct=(c in pct_cols)) for c in r} for r in rows
    ]
    cols = list(formatted[0].keys())
    widths = {c: max(len(c), *(len(r[c]) for r in formatted)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in formatted:
        print("  ".join(r[c].ljust(widths[c]) for c in cols))

    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in formatted:
            f.write(",".join(r[c] for c in cols) + "\n")
    print(f"\n[ok] جدول → {csv_path}")
    print(
        "\nراهنمای انتخاب: معیارِ اصلی combo@10 و L1_acc@10 است (لایهٔ ۱ گلوگاهِ دقت است)؛"
        "\nL1_frontier@0.8 یعنی «چند درصد پوشش با چه دقتی» اگر فقط همسایگی‌های خالص auto شوند."
    )


def _cell(v, as_pct: bool = False) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v * 100:.1f}" if as_pct else f"{v:g}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(REGISTRY), help="اجرای بنچمارک برای یک مدل")
    ap.add_argument("--report", action="store_true", help="جدولِ مقایسهٔ نتایجِ موجود")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="فقط N ردیفِ اول (تستِ دود)")
    args = ap.parse_args()

    if not args.model and not args.report:
        ap.error("--model یا --report لازم است")
    if args.model:
        if not args.data.exists():
            raise SystemExit(
                f"{args.data} یافت نشد — اول: python -m scripts.prepare_retrieval_dataset"
            )
        result = run_model(args.model, args.data, args.out_dir, limit=args.limit)
        print_summary(result)
    if args.report:
        report(args.out_dir)


if __name__ == "__main__":
    main()
