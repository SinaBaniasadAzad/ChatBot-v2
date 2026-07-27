"""بنچمارکِ embedding برای بازیابیِ تیکت — مقایسهٔ مدل‌ها روی دادهٔ واقعی.

هر اجرا برای یک مدل چند «واریانت» می‌سنجد:
  dense               فقط شباهتِ کسینوسیِ بردارِ متراکم
  dense+bm25 (RRF)    ادغامِ رتبه‌ای با بیس‌لاینِ واژگانی — کلیدواژه‌های سیستمی
                      (ERP Plus, Jabber, WBS, ...) را جدی می‌گیرد
  m3-sparse / m3-hybrid   فقط BGE-M3: امتیازِ واژگانیِ خودِ مدل و ادغامش با dense

خروجیِ هر مدل: data/retrieval/results/<model>.json + بردارها (npz، float16).
وابستگی‌های سنگین (torch/sentence-transformers/FlagEmbedding/scipy) فقط داخلِ
توابع import می‌شوند تا testها و مسیرِ BM25 روی هر ماشینی اجرا شوند.

نکتهٔ متدولوژی: همهٔ معیارها leave-one-out روی خودِ مخزن‌اند (خودِ سند حذف می‌شود؛
تکراری‌های عینی در مرحلهٔ پاک‌سازی حذف شده‌اند). Qwen3 عمداً بدونِ instruction
انکد می‌شود (بازیابیِ متقارن سند↔سند: هر آیتم هم کوئری است هم سند).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

from src.retrieval import metrics as M
from src.retrieval.bm25 import BM25, tokenize
from src.retrieval.clean import load_clean

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LAYERS = ("layer1", "layer2")
AGREEMENT_KS = (1, 3, 5, 10)
KNN_KS = (5, 10)
PURITY_K = 15
TOP_K = 25  # ≥ همهٔ نیازها (agreement@10، purity@15، contradiction mining)
PURITY_THRESHOLDS = tuple(round(0.60 + 0.05 * i, 2) for i in range(8))  # 0.60..0.95
CONTRADICTION_SIM = 0.95


# ---------------------------------------------------------------------------
# رجیستری مدل‌ها — افزودنِ مدلِ جدید = یک entry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    id: str
    kind: str                      # "bm25" | "st" | "m3"
    hf: str = ""
    prefix: str = ""               # مثل "query: " برای e5 (دو طرف؛ تسکِ متقارن)
    max_len: int = 512
    batch_gpu: int = 64
    batch_cpu: int = 8
    trust_remote_code: bool = False
    tokenizer_kwargs: dict = field(default_factory=dict)
    note: str = ""


REGISTRY: dict[str, ModelSpec] = {
    "bm25": ModelSpec(id="bm25", kind="bm25", note="بیس‌لاینِ واژگانی — بدونِ مدل"),
    "e5-large": ModelSpec(
        id="e5-large", kind="st", hf="intfloat/multilingual-e5-large",
        prefix="query: ", note="مرجعِ آزموده؛ سقفِ ۵۱۲ توکن",
    ),
    "bge-m3": ModelSpec(
        id="bge-m3", kind="m3", hf="BAAI/bge-m3", batch_gpu=32,
        note="dense + sparse بومی؛ MIT",
    ),
    "qwen3-0.6b": ModelSpec(
        id="qwen3-0.6b", kind="st", hf="Qwen/Qwen3-Embedding-0.6B",
        tokenizer_kwargs={"padding_side": "left"},
        note="نسل جدید؛ Apache-2.0؛ نیازمند transformers>=4.51",
    ),
    "gte-base": ModelSpec(
        id="gte-base", kind="st", hf="Alibaba-NLP/gte-multilingual-base",
        trust_remote_code=True, note="گزینهٔ سبک (اختیاری)",
    ),
    # برای آزمودنِ مدلِ قوی‌تر: فقط این خط را اضافه/فعال کنید (VRAM ~۸GB fp16)
    # "qwen3-4b": ModelSpec(id="qwen3-4b", kind="st", hf="Qwen/Qwen3-Embedding-4B",
    #                       batch_gpu=16, tokenizer_kwargs={"padding_side": "left"}),
}


# ---------------------------------------------------------------------------
# انکودرها (import سنگین‌ها فقط این‌جا)
# ---------------------------------------------------------------------------
def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _vram_peak_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass
    return None


def _encode_st(spec: ModelSpec, texts: list[str]) -> tuple[np.ndarray, dict]:
    import torch
    from sentence_transformers import SentenceTransformer

    device = _device()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}

    t0 = time.perf_counter()
    model = SentenceTransformer(
        spec.hf,
        device=device,
        trust_remote_code=spec.trust_remote_code,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=spec.tokenizer_kwargs or None,
    )
    model.max_seq_length = spec.max_len
    load_s = time.perf_counter() - t0

    batch = spec.batch_gpu if device == "cuda" else spec.batch_cpu
    inputs = [spec.prefix + t for t in texts]
    t0 = time.perf_counter()
    emb = model.encode(
        inputs, batch_size=batch, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float32)
    encode_s = time.perf_counter() - t0

    # تاخیرِ تک‌کوئری (مسیرِ production: انکدِ یک تیکتِ تازه)
    lat = []
    for i in range(min(8, len(inputs))):
        t0 = time.perf_counter()
        model.encode([inputs[i]], normalize_embeddings=True, convert_to_numpy=True)
        lat.append((time.perf_counter() - t0) * 1000)

    timing = {
        "device": device, "load_s": round(load_s, 1), "encode_s": round(encode_s, 1),
        "tickets_per_s": round(len(texts) / encode_s, 1),
        "query_latency_ms": round(median(lat), 1),
        "vram_peak_mb": _vram_peak_mb(), "dim": int(emb.shape[1]),
    }
    return emb, timing


def _encode_m3(spec: ModelSpec, texts: list[str]) -> tuple[np.ndarray, list[dict], dict]:
    """BGE-M3: بردارِ متراکم + وزن‌های واژگانی (sparse) در یک پاس."""
    import torch
    from FlagEmbedding import BGEM3FlagModel

    device = _device()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = BGEM3FlagModel(spec.hf, use_fp16=(device == "cuda"))
    load_s = time.perf_counter() - t0

    batch = spec.batch_gpu if device == "cuda" else spec.batch_cpu
    t0 = time.perf_counter()
    out = model.encode(
        texts, batch_size=batch, max_length=spec.max_len,
        return_dense=True, return_sparse=True, return_colbert_vecs=False,
    )
    encode_s = time.perf_counter() - t0
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True).clip(min=1e-9)
    lexical = out["lexical_weights"]

    lat = []
    for i in range(min(8, len(texts))):
        t0 = time.perf_counter()
        model.encode([texts[i]], max_length=spec.max_len, return_dense=True, return_sparse=True)
        lat.append((time.perf_counter() - t0) * 1000)

    timing = {
        "device": device, "load_s": round(load_s, 1), "encode_s": round(encode_s, 1),
        "tickets_per_s": round(len(texts) / encode_s, 1),
        "query_latency_ms": round(median(lat), 1),
        "vram_peak_mb": _vram_peak_mb(), "dim": int(dense.shape[1]),
    }
    return dense, lexical, timing


def _m3_sparse_topk(lexical: list[dict], k: int, block: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """top-k با امتیازِ واژگانیِ M3 (ضربِ ماتریسِ اسپارس، بلوکی)."""
    from scipy import sparse

    vocab: dict[str, int] = {}
    rows, cols, vals = [], [], []
    for i, weights in enumerate(lexical):
        for tok, w in weights.items():
            j = vocab.setdefault(str(tok), len(vocab))
            rows.append(i)
            cols.append(j)
            vals.append(float(w))
    X = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(lexical), max(len(vocab), 1)), dtype=np.float32
    )
    n = X.shape[0]
    idx = np.empty((n, k), dtype=np.int32)
    sim = np.empty((n, k), dtype=np.float32)
    XT = X.T.tocsr()
    for s in range(0, n, block):
        scores = (X[s : s + block] @ XT).toarray()
        r = np.arange(scores.shape[0])
        scores[r, s + r] = -np.inf
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        part_s = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_s, axis=1)
        idx[s : s + block] = np.take_along_axis(part, order, axis=1)
        sim[s : s + block] = np.take_along_axis(part_s, order, axis=1)
    return idx, sim


def bm25_topk(rows: list[dict], k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    tokens = [tokenize(r["bm25_text"]) for r in rows]
    t0 = time.perf_counter()
    bm = BM25(tokens)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx, sim = M.top_k_from_score_fn(lambda i: bm.get_scores(tokens[i]), len(rows), k)
    score_s = time.perf_counter() - t0
    timing = {
        "device": "cpu", "load_s": round(build_s, 2), "encode_s": round(score_s, 1),
        "tickets_per_s": round(len(rows) / max(score_s, 1e-9), 1),
        "query_latency_ms": round(score_s * 1000 / max(len(rows), 1), 2),
        "vram_peak_mb": None, "dim": None,
    }
    return idx, sim, timing


# ---------------------------------------------------------------------------
# ارزیابیِ یک فهرستِ همسایه‌ها
# ---------------------------------------------------------------------------
def evaluate_neighbors(
    nn_idx: np.ndarray, nn_sim: np.ndarray, codes_by_layer: dict[str, tuple[np.ndarray, list[str]]]
) -> dict:
    result: dict = {}
    preds: dict[str, np.ndarray] = {}
    valids: dict[str, np.ndarray] = {}
    for layer, (codes, classes) in codes_by_layer.items():
        layer_res: dict = {
            "agreement": {
                str(k): round(v, 4)
                for k, v in M.agreement_at_k(nn_idx, nn_sim, codes, AGREEMENT_KS).items()
            }
        }
        for k in KNN_KS:
            pred, _ = M.knn_predict(nn_idx, nn_sim, codes, len(classes), k)
            layer_res[f"knn_acc@{k}"] = round(M.accuracy(pred, codes), 4)
            if k == max(KNN_KS):
                preds[layer] = pred
        pred_p, share = M.knn_predict(nn_idx, nn_sim, codes, len(classes), PURITY_K)
        valid = (codes >= 0) & (pred_p >= 0)
        correct = pred_p == codes
        layer_res["purity_frontier"] = M.purity_frontier(
            share, correct, valid, PURITY_THRESHOLDS
        )
        valids[layer] = codes >= 0
        result[layer] = layer_res

    both_valid = np.logical_and.reduce([valids[L] for L in codes_by_layer])
    both_ok = np.logical_and.reduce(
        [
            (preds[L] >= 0) & (preds[L] == codes_by_layer[L][0])
            for L in codes_by_layer
        ]
    )
    result["combo"] = {
        f"knn_acc@{max(KNN_KS)}": round(float(both_ok[both_valid].mean()), 4)
        if both_valid.any()
        else 0.0
    }
    return result


# ---------------------------------------------------------------------------
# اجرای کامل برای یک مدل
# ---------------------------------------------------------------------------
def run_model(
    model_id: str,
    data_path: Path,
    out_dir: Path,
    limit: int | None = None,
) -> dict:
    spec = REGISTRY.get(model_id)
    if spec is None:
        raise SystemExit(f"مدل ناشناخته: {model_id} — گزینه‌ها: {', '.join(REGISTRY)}")

    rows = load_clean(data_path)
    if limit:
        rows = rows[:limit]
    if len(rows) < TOP_K + 1:
        raise SystemExit(f"دادهٔ کافی نیست ({len(rows)} ردیف).")
    texts = [r["embed_text"] for r in rows]
    keys = [r["key"] for r in rows]
    codes_by_layer = {L: M.encode_labels([r.get(L) for r in rows]) for L in LAYERS}
    out_dir.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    lexical = None

    if spec.kind == "bm25":
        nn_idx, nn_sim, timing = bm25_topk(rows, TOP_K)
        variants.append(("bm25", nn_idx, nn_sim))
        np.savez_compressed(out_dir / "bm25_topk.npz", idx=nn_idx, sim=nn_sim, keys=np.array(keys))
    else:
        if spec.kind == "st":
            emb, timing = _encode_st(spec, texts)
        else:
            emb, lexical, timing = _encode_m3(spec, texts)
        np.savez_compressed(
            out_dir / f"emb_{spec.id}.npz", emb=emb.astype(np.float16), keys=np.array(keys)
        )
        nn_idx, nn_sim = M.top_k_neighbors(emb, TOP_K)
        variants.append(("dense", nn_idx, nn_sim))

        # واریانتِ هیبریدِ dense + BM25 (برای همهٔ مدل‌های متراکم)
        bm_idx, bm_sim, _ = bm25_topk(rows, TOP_K)
        h_idx, h_w = M.rrf_fuse([nn_idx, bm_idx], TOP_K)
        variants.append(("dense+bm25", h_idx, h_w))

        if lexical is not None:  # واریانت‌های ویژهٔ M3
            sp_idx, sp_sim = _m3_sparse_topk(lexical, TOP_K)
            variants.append(("m3-sparse", sp_idx, sp_sim))
            hy_idx, hy_w = M.rrf_fuse([nn_idx, sp_idx], TOP_K)
            variants.append(("m3-hybrid", hy_idx, hy_w))

    variant_results = []
    contradiction_file = None
    for name, v_idx, v_sim in variants:
        res = evaluate_neighbors(v_idx, v_sim, codes_by_layer)
        entry = {"variant": name, "metrics": res}
        # استخراجِ تناقض‌ها فقط روی امتیازِ کسینوسی (مقیاسِ قابل‌تفسیر)
        if name == "dense":
            cons = M.find_contradictions(
                v_idx, v_sim, {L: codes_by_layer[L][0] for L in LAYERS}, CONTRADICTION_SIM
            )
            contradiction_file = out_dir / f"{spec.id}_contradictions.jsonl"
            with contradiction_file.open("w", encoding="utf-8") as f:
                for c in cons:
                    f.write(
                        json.dumps(
                            {
                                "similarity": c["similarity"],
                                "differs_on": c["differs_on"],
                                "a": {k: rows[c["i"]][k] for k in ("key", *LAYERS, "embed_text")},
                                "b": {k: rows[c["j"]][k] for k in ("key", *LAYERS, "embed_text")},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            entry["contradiction_pairs"] = len(cons)
        variant_results.append(entry)

    result = {
        "model": spec.id,
        "hf": spec.hf or None,
        "note": spec.note,
        "n_tickets": len(rows),
        "timing": timing,
        "variants": variant_results,
        "params": {
            "top_k": TOP_K, "purity_k": PURITY_K, "knn_ks": list(KNN_KS),
            "contradiction_sim": CONTRADICTION_SIM, "max_len": spec.max_len,
        },
    }
    (out_dir / f"{spec.id}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if contradiction_file:
        result["contradictions_path"] = str(contradiction_file)
    return result
