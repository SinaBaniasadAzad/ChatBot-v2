"""ساختِ ایندکسِ retrieval برای مسیرِ production (پیش‌فرض: BGE-M3 dense — برندهٔ بنچمارک).

اجرا (Kaggle GPU: ~۱ دقیقه؛ CPU محلی: ~۲–۵ دقیقه):
    python -m scripts.build_retrieval_index
    python -m scripts.build_retrieval_index --model e5-large   # مدلِ جایگزین از رجیستری

خروجی: data/retrieval/index.npz  (بردارهای fp16 + کلیدها + نامِ مدل)
این فایل را همراهِ data/retrieval/tickets_clean.jsonl روی سرور بگذارید؛ نامِ مدل
داخلِ ایندکس ذخیره می‌شود و بازیاب هنگامِ اجرا همان را بارگذاری می‌کند.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.bench import REGISTRY, _encode_m3, _encode_st  # noqa: E402
from src.retrieval.clean import load_clean  # noqa: E402

DEFAULT_DATA = Path("data/retrieval/tickets_clean.jsonl")
DEFAULT_OUT = Path("data/retrieval/index.npz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bge-m3",
                    choices=[k for k, s in REGISTRY.items() if s.kind in ("st", "m3")])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    spec = REGISTRY[args.model]
    rows = load_clean(args.data)
    texts = [r["embed_text"] for r in rows]
    keys = [r["key"] for r in rows]
    print(f"انکدِ {len(rows)} تیکت با {spec.hf} ...", file=sys.stderr)

    t0 = time.perf_counter()
    if spec.kind == "m3":
        emb, _lexical, timing = _encode_m3(spec, texts)
    else:
        emb, timing = _encode_st(spec, texts)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        emb=emb.astype(np.float16),
        keys=np.array(keys),
        model=np.array(spec.hf),
        kind=np.array(spec.kind),      # backendِ انکودرِ زمانِ اجرا (m3 | st)
        prefix=np.array(spec.prefix),  # پیشوندِ کوئری (مثل "query: " برای e5)
        max_len=np.array(spec.max_len),
    )
    print(
        f"[ok] index → {args.out}  (n={len(rows)}, dim={emb.shape[1]}, "
        f"model={spec.hf}, {time.perf_counter() - t0:.0f}s, device={timing['device']})"
    )


if __name__ == "__main__":
    main()
