"""
ارزیابی دقت روی فایل خام تیکت‌ها (فرمت INC_DB.jsonl).

هر خط:
  {"Key": "INC-20689", "Application": "ERP", "Summary": "...", "Description": "...",
   "Labels": {"layer_1": "Incident", "layer_2": "ERP"}}

- ground truth = Labels.layer_1 / layer_2 (نام نمایشی مثل "Incident"/"ERP").
  این اسکریپت نام‌ها را خودکار به id داخلی نگاشت می‌کند و کلیدِ "layer_1" را با
  "layer1"ِ taxonomy تطبیق می‌دهد (حذفِ underscore).
- مدل به‌صورت single-shot اجرا می‌شود (بدون شبیه‌سازیِ سوال تکمیلی) و بهترین حدسِ
  هر لایه گرفته می‌شود — یعنی «دقتِ خامِ مدل».
- گزارش:
    • دقتِ هر لایه
    • recall هر کلاس  (از Incidentها چند درصد درست؟ از ERP/Staff چند درصد؟)
    • دقتِ کلی = تطبیقِ کاملِ همهٔ لایه‌ها
    • confusion matrix هر لایه
    • توکن و هزینهٔ واقعی (از usageِ DeepSeek، شاملِ cache hit/miss)

اجرا (از ریشهٔ پروژه):
    python -m scripts.eval_incdb data/INC_DB.jsonl
    python -m scripts.eval_incdb data/INC_DB.jsonl --limit 300
    python -m scripts.eval_incdb data/INC_DB.jsonl --balanced 75     # حداکثر ۷۵ از هر ترکیب
    python -m scripts.eval_incdb data/INC_DB.jsonl --workers 6 --out preds.jsonl

قیمت‌ها (به ازای ۱M توکن) با فلگ قابل تنظیم‌اند؛ پیش‌فرض = نرخِ استانداردِ
deepseek-chat (V3). قبل از استناد، نرخِ روز را از platform.deepseek.com بررسی کن.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.classifier import Classifier  # noqa: E402


def _norm_key(k: str) -> str:
    """«layer_1» / «Layer 1» -> «layer1» (فقط حروف و عدد)."""
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


def _build_maps(tax):
    """نگاشت‌های نام->id برای لایه‌ها و برچسب‌ها (تحملِ نام نمایشی یا id)."""
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


def load_rows(path: Path, limit: int | None, balanced: int | None, tax) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    if balanced:
        # حداکثر N نمونه از هر ترکیبِ (layer1, layer2) تا کلاس‌های نادر هم پوشش بگیرند.
        layer_key_map, label_map = _build_maps(tax)
        buckets: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            labels = r.get("Labels") or {}
            combo = []
            for layer in tax.layers:
                val = None
                for gk, gv in labels.items():
                    if _norm_key(gk) == _norm_key(layer.id):
                        val = label_map[layer.id].get(str(gv).strip().lower())
                combo.append(val)
            buckets[tuple(combo)].append(r)
        rows = []
        for items in buckets.values():
            rows.extend(items[:balanced])

    return rows[:limit] if limit else rows


def _field(row: dict, *names: str) -> str:
    for n in names:
        if row.get(n):
            return str(row[n])
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--balanced", type=int, default=None, help="حداکثر N نمونه از هر ترکیب برچسب")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="ذخیرهٔ پیش‌بینی‌ها (JSONL) برای تحلیل خطا")
    # قیمت به ازای ۱M توکن (نرخِ استانداردِ deepseek-chat؛ نرخِ روز را بررسی کن)
    ap.add_argument("--price-in", type=float, default=0.27, help="input cache-miss $/1M")
    ap.add_argument("--price-cache", type=float, default=0.07, help="input cache-hit $/1M")
    ap.add_argument("--price-out", type=float, default=1.10, help="output $/1M")
    args = ap.parse_args()

    clf = Classifier()
    tax = clf.taxonomy
    layer_key_map, label_map = _build_maps(tax)
    rows = load_rows(args.data_path, args.limit, args.balanced, tax)
    total = len(rows)
    print(f"بارگذاری {total} تیکت از {args.data_path}", file=sys.stderr)

    per_layer_total: dict[str, int] = defaultdict(int)
    per_layer_correct: dict[str, int] = defaultdict(int)
    cls_total = {l.id: defaultdict(int) for l in tax.layers}
    cls_correct = {l.id: defaultdict(int) for l in tax.layers}
    confusion = {l.id: defaultdict(lambda: defaultdict(int)) for l in tax.layers}
    full_total = 0
    full_correct = 0
    errors = 0
    tok = defaultdict(int)  # prompt / cache_hit / cache_miss / completion
    t_start = time.perf_counter()
    out_fh = args.out.open("w", encoding="utf-8") if args.out else None

    def run_one(row: dict):
        summary = _field(row, "Summary", "summary")
        description = _field(row, "Description", "description")
        try:
            out, meta = clf.classify(summary, description)
            return row, out, meta, None
        except Exception as e:  # شکستِ کاملِ یک تیکت نباید کلِ اجرا را بکُشد
            return row, None, {}, str(e)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, out, meta, err in pool.map(run_one, rows):
            done += 1
            if err:
                errors += 1

            # توکن/هزینه
            u = meta.get("usage") or {}
            pt = u.get("prompt_tokens", 0) or 0
            ct = u.get("completion_tokens", 0) or 0
            hit = u.get("prompt_cache_hit_tokens")
            miss = u.get("prompt_cache_miss_tokens")
            if hit is None or miss is None:  # اگر تفکیکِ کش نبود، همه را miss فرض کن
                hit, miss = 0, pt
            tok["prompt"] += pt
            tok["cache_hit"] += hit
            tok["cache_miss"] += miss
            tok["completion"] += ct

            labels = row.get("Labels") or {}
            row_all_ok = True
            row_has_all = True
            rec = {"Key": row.get("Key"), "pred": {}, "true": {}}

            for layer in tax.layers:
                # ground truth این لایه
                true_id = None
                for gk, gv in labels.items():
                    if _norm_key(gk) == _norm_key(layer.id):
                        true_id = label_map[layer.id].get(str(gv).strip().lower())
                if true_id is None:
                    row_has_all = False
                    continue

                lo = out.layers.get(layer.id) if out else None
                pred_id = lo.top.label if (lo and lo.top) else None

                per_layer_total[layer.id] += 1
                cls_total[layer.id][true_id] += 1
                confusion[layer.id][true_id][pred_id] += 1
                ok = pred_id == true_id
                if ok:
                    per_layer_correct[layer.id] += 1
                    cls_correct[layer.id][true_id] += 1
                else:
                    row_all_ok = False
                rec["true"][layer.id] = true_id
                rec["pred"][layer.id] = pred_id

            if row_has_all:
                full_total += 1
                full_correct += int(row_all_ok)
            if out_fh:
                rec["correct"] = row_all_ok and row_has_all
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if done % 25 == 0:
                print(f"... {done}/{total}", file=sys.stderr)

    if out_fh:
        out_fh.close()
    wall = time.perf_counter() - t_start

    def name(lid, cid):
        lbl = tax.get_layer(lid).get_label(cid)
        return lbl.name if lbl else str(cid)

    M = 1_000_000
    cost = (
        tok["cache_hit"] * args.price_cache
        + tok["cache_miss"] * args.price_in
        + tok["completion"] * args.price_out
    ) / M

    print("\n================= نتایج ارزیابی (single-shot) =================")
    print(f"تیکت‌های ارزیابی‌شده: {total}   |   خطای فراخوانی: {errors}")
    print(
        f"توکن‌ها → prompt={tok['prompt']:,} "
        f"(cache_hit={tok['cache_hit']:,} / miss={tok['cache_miss']:,})  "
        f"completion={tok['completion']:,}"
    )
    print(f"هزینهٔ تخمینی: ${cost:.4f}   (نرخ/1M: in=${args.price_in}, hit=${args.price_cache}, out=${args.price_out} — نرخِ روز را بررسی کن)")
    if total:
        print(f"زمان کل: {wall:.0f}s   |   میانگین: {wall*1000/total:.0f} ms/تیکت")

    print("\n— دقتِ هر لایه —")
    for layer in tax.layers:
        tot = per_layer_total[layer.id]
        acc = per_layer_correct[layer.id] / tot if tot else 0
        print(f"  {layer.id} ({layer.name.split('/')[0].strip()}): {acc:.1%}  ({per_layer_correct[layer.id]}/{tot})")

    print("\n— recall هر کلاس (از کلاسِ واقعی، چند درصد درست تشخیص داده شد) —")
    for layer in tax.layers:
        print(f"  {layer.id}:")
        for cid in layer.label_ids:
            t = cls_total[layer.id].get(cid, 0)
            c = cls_correct[layer.id].get(cid, 0)
            r = c / t if t else 0
            print(f"    {name(layer.id, cid):<18} {r:6.1%}  ({c}/{t})")

    if full_total:
        print(f"\n— دقتِ کلی (همهٔ لایه‌ها هم‌زمان درست) —\n  {full_correct/full_total:.1%}  ({full_correct}/{full_total})")

    for layer in tax.layers:
        ids = layer.label_ids
        print(f"\n— Confusion «{layer.id}» (سطر=واقعی، ستون=پیش‌بینی؛ ⌀=بدون پیش‌بینی) —")
        print("true\\pred".ljust(18) + "".join(name(layer.id, p)[:16].ljust(18) for p in ids) + "⌀".ljust(6))
        for t in ids:
            none_n = confusion[layer.id][t].get(None, 0)
            row = name(layer.id, t).ljust(18) + "".join(str(confusion[layer.id][t].get(p, 0)).ljust(18) for p in ids)
            print(row + str(none_n).ljust(6))


if __name__ == "__main__":
    main()
