"""
آپلودِ دیتاستِ طلایی به Langfuse (Dataset Management).

همان نمونه‌گیریِ eval_incdb (frac/seed/balanced) را دارد تا دیتاستِ Langfuse دقیقاً
با نمونهٔ ارزیابیِ استانداردِ پروژه (frac=0.2, seed=42) یکی باشد و مقایسهٔ runها
معنادار بماند. برچسب‌های طلایی به idِ داخلیِ taxonomy نگاشت می‌شوند.

اجرا (از ریشهٔ پروژه، با کلیدهای Langfuse در .env):
    python -m scripts.langfuse_dataset tests/Ticketing_DB.jsonl \
        --name ticketing-gold-20pct --frac 0.2 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_incdb import _build_maps, _field, _gt_label, load_rows  # noqa: E402
from src import observability as obs  # noqa: E402
from src.taxonomy import load_taxonomy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_path", type=Path)
    ap.add_argument("--name", required=True, help="نامِ دیتاست در Langfuse")
    ap.add_argument("--description", default="Gold-labeled tickets for triage evaluation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--balanced", type=int, default=None)
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lf = obs.get_client()
    if lf is None:
        raise SystemExit(
            "Langfuse در دسترس نیست: OBSERVABILITY_ENABLED و LANGFUSE_PUBLIC_KEY/SECRET_KEY را در .env تنظیم کنید."
        )

    tax = load_taxonomy()
    _, label_map = _build_maps(tax)
    rows = load_rows(args.data_path, args.limit, args.balanced, tax, frac=args.frac, seed=args.seed)

    lf.create_dataset(
        name=args.name,
        description=args.description,
        metadata={
            "source": str(args.data_path),
            "sampling": {"frac": args.frac, "seed": args.seed,
                         "balanced": args.balanced, "limit": args.limit},
            "n_rows": len(rows),
        },
    )

    n_ok = n_skip = 0
    for row in rows:
        key = str(row.get("Key") or "")
        expected = {}
        for layer in tax.layers:
            gid = _gt_label(row, layer, label_map)
            if gid:
                expected[layer.id] = gid
        if not key or len(expected) < len(tax.layers):
            n_skip += 1  # بدونِ برچسبِ کاملِ طلایی، آیتمِ ارزیابی نمی‌سازیم
            continue
        lf.create_dataset_item(
            dataset_name=args.name,
            id=key,  # upsert با همان Key → اجرای مجدد امن است
            input={
                "summary": _field(row, "Summary", "summary"),
                "description": _field(row, "Description", "description"),
            },
            expected_output=expected,
            metadata={"key": key, "application": row.get("Application")},
        )
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"... {n_ok} آیتم آپلود شد", file=sys.stderr)

    obs.flush()
    print(f"دیتاست «{args.name}»: {n_ok} آیتم آپلود شد، {n_skip} ردِ بدونِ برچسبِ کامل.")


if __name__ == "__main__":
    main()
