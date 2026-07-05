"""مقایسهٔ آماریِ دو اجرای ارزیابی (A/B) روی همان نمونه — روشِ درستِ سنجشِ بهبود.

به‌جای مقایسهٔ دو عددِ کلیِ دقت (که نویز دارد)، پیش‌بینی‌های جفت‌شده مقایسه می‌شوند:
کدام تیکت‌ها درست شدند (fixed) و کدام خراب شدند (broken) + آزمونِ McNemar
(دوجمله‌ایِ دقیق). p کوچک (≤0.05) یعنی تفاوت احتمالاً واقعی است، نه نویز.

اجرا:
    python -m scripts.eval_incdb ... --out preds_base.jsonl      # اجرای A
    python -m scripts.eval_incdb ... --out preds_new.jsonl       # اجرای B
    python -m scripts.compare_eval_runs preds_base.jsonl preds_new.jsonl
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def _load(path: Path) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["Key"]] = r
    return out


def _mcnemar_p(b: int, c: int) -> float:
    """آزمونِ دقیقِ McNemar (دوجمله‌ای دوطرفه) روی جفت‌های ناسازگار."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def compare_layer(a: dict, b: dict, keys: list[str], layer: str) -> dict:
    fixed, broken = [], []
    for k in keys:
        ta = a[k]["true"].get(layer)
        if ta is None:
            continue
        ok_a = a[k]["pred"].get(layer) == ta
        ok_b = b[k]["pred"].get(layer) == ta
        if not ok_a and ok_b:
            fixed.append(k)
        elif ok_a and not ok_b:
            broken.append(k)
    return {"fixed": fixed, "broken": broken, "p": _mcnemar_p(len(fixed), len(broken))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path, help="preds.jsonl اجرای پایه")
    ap.add_argument("run_b", type=Path, help="preds.jsonl اجرای جدید")
    ap.add_argument("--show", type=int, default=10, help="حداکثر کلیدِ نمایشی از هر فهرست")
    args = ap.parse_args()

    a, b = _load(args.run_a), _load(args.run_b)
    keys = sorted(set(a) & set(b))
    only_a, only_b = len(a) - len(keys), len(b) - len(keys)
    if only_a or only_b:
        print(f"[هشدار] نمونه‌ها کاملاً یکسان نیستند (فقط A: {only_a}، فقط B: {only_b})؛"
              " مقایسه روی اشتراک انجام شد.")
    print(f"تیکت‌های مشترک: {len(keys)}\n")

    layers = sorted({L for r in a.values() for L in r["true"]})
    for layer in layers + ["__overall__"]:
        if layer == "__overall__":
            fixed = [k for k in keys if not a[k]["correct"] and b[k]["correct"]]
            broken = [k for k in keys if a[k]["correct"] and not b[k]["correct"]]
            res = {"fixed": fixed, "broken": broken, "p": _mcnemar_p(len(fixed), len(broken))}
            acc_a = sum(a[k]["correct"] for k in keys) / len(keys)
            acc_b = sum(b[k]["correct"] for k in keys) / len(keys)
            name = f"کل (هر دو لایه)  A={acc_a:.1%} → B={acc_b:.1%}"
        else:
            res = compare_layer(a, b, keys, layer)
            name = layer
        verdict = "معنادار ✓" if res["p"] <= 0.05 else "در حدِ نویز"
        print(f"-- {name} --")
        print(f"  fixed={len(res['fixed'])}  broken={len(res['broken'])}  "
              f"McNemar p={res['p']:.4f}  ({verdict})")
        if res["fixed"]:
            print(f"  نمونهٔ fixed:  {', '.join(res['fixed'][:args.show])}")
        if res["broken"]:
            print(f"  نمونهٔ broken: {', '.join(res['broken'][:args.show])}")
        print()


if __name__ == "__main__":
    main()
