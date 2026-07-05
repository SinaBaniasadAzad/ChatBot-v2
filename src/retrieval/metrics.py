"""معیارهای کیفیتِ بازیابی برای بنچمارک embedding — فقط numpy (تست‌پذیرِ آفلاین).

معیارهای اصلی و معنای عملیاتی‌شان:
  • label-agreement@k  — چند درصدِ k همسایهٔ برتر هم‌برچسبِ کوئری‌اند؟ پیش‌بینِ
    مستقیمِ «کیفیتِ مثال‌های few-shotِ بازیابی‌شده».
  • kNN LOO accuracy  — رایِ وزن‌دارِ همسایه‌ها به‌عنوانِ یک دسته‌بندِ مستقل
    (leave-one-out). پایهٔ سیگنالِ «نظرِ دومِ kNN» در معماری آینده.
  • purity + frontier — سهمِ برچسبِ اکثریت در همسایگی؛ منحنیِ coverage/accuracy
    در آستانه‌های مختلف = همان KPIِ «نرخِ اتوماسیون در دقتِ هدف».
  • contradictions    — جفت‌های شبه‌تکراری با برچسبِ متفاوت = کاندیدِ نویزِ برچسب.

قرارداد: اندیسِ همسایهٔ -1 یعنی «خالی» (پس از RRF ممکن است) و باید نادیده گرفته شود.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# همسایه‌های نزدیک
# ---------------------------------------------------------------------------
def top_k_neighbors(emb: np.ndarray, k: int, block: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """k همسایهٔ نزدیک (کسینوسی) برای هر ردیف؛ بدونِ خودِ ردیف. بلوکی تا در ۱۰هزار سند هم جا شود.

    ورودی باید L2-normalized باشد. خروجی: (idx[n,k], sim[n,k]) مرتب نزولی.
    """
    n = emb.shape[0]
    if k >= n:
        raise ValueError(f"k={k} باید کوچکتر از تعداد اسناد ({n}) باشد")
    emb = emb.astype(np.float32, copy=False)
    idx = np.empty((n, k), dtype=np.int32)
    sim = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, block):
        e = emb[s : s + block]
        scores = e @ emb.T
        rows = np.arange(e.shape[0])
        scores[rows, s + rows] = -np.inf  # حذفِ خود
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        part_s = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_s, axis=1)
        idx[s : s + block] = np.take_along_axis(part, order, axis=1)
        sim[s : s + block] = np.take_along_axis(part_s, order, axis=1)
    return idx, sim


def top_k_from_score_fn(score_fn, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """همان خروجی برای امتیازدهنده‌های غیرماتریسی (BM25): score_fn(i) -> امتیاز به همهٔ اسناد."""
    idx = np.empty((n, k), dtype=np.int32)
    sim = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        scores = np.asarray(score_fn(i), dtype=np.float32)
        scores[i] = -np.inf
        part = np.argpartition(-scores, k)[:k]
        order = np.argsort(-scores[part])
        idx[i] = part[order]
        sim[i] = scores[part][order]
    return idx, sim


# ---------------------------------------------------------------------------
# برچسب‌ها
# ---------------------------------------------------------------------------
def encode_labels(values: list[str | None]) -> tuple[np.ndarray, list[str]]:
    """برچسب‌های رشته‌ای -> کدِ عددی (None/ناشناخته = -1). خروجی: (codes, classes)."""
    classes = sorted({v for v in values if v})
    to_int = {c: i for i, c in enumerate(classes)}
    codes = np.array([to_int.get(v, -1) for v in values], dtype=np.int32)
    return codes, classes


def _valid_neighbors(nn_idx_row: np.ndarray, nn_sim_row: np.ndarray, codes: np.ndarray):
    """همسایه‌های دارای اندیس و برچسبِ معتبر (به ترتیبِ رتبه)."""
    mask = nn_idx_row >= 0
    ids = nn_idx_row[mask]
    sims = nn_sim_row[mask]
    lab = codes[ids]
    ok = lab >= 0
    return lab[ok], sims[ok]


# ---------------------------------------------------------------------------
# معیارها
# ---------------------------------------------------------------------------
def agreement_at_k(
    nn_idx: np.ndarray, nn_sim: np.ndarray, codes: np.ndarray, ks: tuple[int, ...]
) -> dict[int, float]:
    """میانگینِ سهمِ همسایه‌های هم‌برچسب در kتای برترِ *معتبر*."""
    out = {k: [] for k in ks}
    for i in range(len(codes)):
        if codes[i] < 0:
            continue
        lab, _ = _valid_neighbors(nn_idx[i], nn_sim[i], codes)
        for k in ks:
            if len(lab) == 0:
                continue
            top = lab[:k]
            out[k].append(float(np.mean(top == codes[i])))
    return {k: (float(np.mean(v)) if v else 0.0) for k, v in out.items()}


def knn_predict(
    nn_idx: np.ndarray, nn_sim: np.ndarray, codes: np.ndarray, n_classes: int, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """رایِ وزن‌دار (وزن = امتیازِ شباهت، کفِ صفر) روی k همسایهٔ معتبرِ برتر.

    خروجی: (pred[n] با -1 برای بدونِ رای، vote_share[n] = purity در همان k).
    """
    n = len(codes)
    pred = np.full(n, -1, dtype=np.int32)
    share = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lab, sims = _valid_neighbors(nn_idx[i], nn_sim[i], codes)
        lab, sims = lab[:k], np.maximum(sims[:k], 0.0)
        if len(lab) == 0 or sims.sum() <= 0:
            continue
        w = np.bincount(lab, weights=sims, minlength=n_classes)
        pred[i] = int(w.argmax())
        share[i] = float(w.max() / w.sum())
    return pred, share


def accuracy(pred: np.ndarray, codes: np.ndarray) -> float:
    mask = (codes >= 0) & (pred >= 0)
    return float(np.mean(pred[mask] == codes[mask])) if mask.any() else 0.0


def purity_frontier(
    share: np.ndarray, correct: np.ndarray, valid: np.ndarray, thresholds: tuple[float, ...]
) -> list[dict]:
    """در هر آستانهٔ purity: چه پوششی auto می‌شود و دقتِ آن بخش چقدر است؟"""
    rows = []
    n_valid = int(valid.sum())
    for t in thresholds:
        m = valid & (share >= t)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "coverage": round(float(m.sum() / n_valid), 4) if n_valid else 0.0,
                "accuracy": round(float(correct[m].mean()), 4) if m.any() else None,
                "n": int(m.sum()),
            }
        )
    return rows


def rrf_fuse(
    idx_lists: list[np.ndarray], k_out: int, k_rrf: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """ادغامِ چند فهرستِ رتبه‌بندی با Reciprocal Rank Fusion.

    وزنِ خروجی = امتیازِ RRF (فقط برای وزن‌دهیِ نسبیِ رای معتبر است، شباهت نیست).
    جای خالی با اندیس -1 پر می‌شود.
    """
    n = idx_lists[0].shape[0]
    fused_idx = np.full((n, k_out), -1, dtype=np.int32)
    fused_w = np.zeros((n, k_out), dtype=np.float32)
    for i in range(n):
        scores: dict[int, float] = {}
        for idxs in idx_lists:
            for rank, j in enumerate(idxs[i]):
                if j < 0:
                    continue
                scores[int(j)] = scores.get(int(j), 0.0) + 1.0 / (k_rrf + rank + 1)
        top = sorted(scores.items(), key=lambda x: -x[1])[:k_out]
        for p, (j, w) in enumerate(top):
            fused_idx[i, p] = j
            fused_w[i, p] = w
    return fused_idx, fused_w


def find_contradictions(
    nn_idx: np.ndarray,
    nn_sim: np.ndarray,
    codes_by_layer: dict[str, np.ndarray],
    threshold: float,
) -> list[dict]:
    """جفت‌های شبه‌تکراری (شباهت ≥ آستانه) با برچسبِ متفاوت — کاندیدِ نویزِ برچسب."""
    seen: set[tuple[int, int]] = set()
    out: list[dict] = []
    n = nn_idx.shape[0]
    for i in range(n):
        for j, s in zip(nn_idx[i], nn_sim[i]):
            j = int(j)
            if j < 0 or s < threshold:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            diff = [
                layer
                for layer, c in codes_by_layer.items()
                if c[a] >= 0 and c[b] >= 0 and c[a] != c[b]
            ]
            if diff:
                out.append({"i": a, "j": b, "similarity": round(float(s), 4), "differs_on": diff})
    out.sort(key=lambda r: -r["similarity"])
    return out
