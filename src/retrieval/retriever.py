"""بازیابِ سابقه (precedent retriever) برای مسیرِ production.

دو خدمت به هر دسته‌بندی می‌دهد:
  ۱) demos: k تیکتِ مشابهِ برچسب‌خورده از تاریخچهٔ سازمان → تزریق به پیامِ کاربر
     (پیامِ سیستم ثابت می‌ماند تا prompt cache حفظ شود).
  ۲) votes: رایِ وزن‌دارِ همسایه‌ها به‌ازای هر لایه + خلوص (purity) → «نظرِ دومِ»
     مستقل از LLM برای گِیتِ اطمینان در لایهٔ تصمیم.

اصولِ خطاپذیری:
  - هر خطایی در ساخت/بارگذاری (نبودِ torch/FlagEmbedding، نبودِ ایندکس، ...) فقط
    retrieval را غیرفعال می‌کند؛ مسیرِ اصلیِ دسته‌بندی هرگز نمی‌افتد.
  - اگر بیشینهٔ شباهت زیرِ کف باشد (تیکتِ بی‌سابقه)، retrieval کنار می‌کشد (None)
    تا سابقهٔ بی‌ربط به مدل تزریق نشود.
  - encode با قفل، thread-safe است (ارزیابیِ موازی و uvicorn threadpool).

سازگاریِ ایندکس: بردارِ کوئری باید با همان مدلِ ایندکس ساخته شود؛ نامِ مدل داخلِ
npz ذخیره شده و همان بارگذاری می‌شود (scripts/build_retrieval_index.py).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config.settings import settings
from src.retrieval.clean import clean_text
from src.utils.logging import get_logger

log = get_logger("retriever")

_QUERY_MAX_CHARS = 1200  # هم‌راستا با max_chars پاک‌سازیِ ایندکس
_ENCODE_MAX_LEN = 512    # هم‌راستا با build_retrieval_index / بنچمارک


@dataclass
class LayerVote:
    label: str
    purity: float  # سهمِ وزنیِ برچسبِ اکثریت در همسایگی (0..1)
    n: int         # تعدادِ همسایه‌های رای‌دهنده


@dataclass
class RetrievalResult:
    demos: list[dict]                 # ردیف‌های تیکتِ مشابه (برای few-shot)
    demo_sims: list[float]
    votes: dict[str, LayerVote]       # layer_id -> رای kNN
    top_similarity: float

    def meta(self) -> dict:
        """خلاصهٔ قابلِ‌لاگ (کلیدها و شباهت‌ها؛ بدونِ متنِ کامل)."""
        return {
            "top_similarity": round(self.top_similarity, 4),
            "demo_keys": [d["key"] for d in self.demos],
            "demo_sims": [round(s, 4) for s in self.demo_sims],
            "votes": {
                lid: {"label": v.label, "purity": round(v.purity, 4), "n": v.n}
                for lid, v in self.votes.items()
            },
        }

    def votes_dict(self) -> dict[str, dict]:
        return {lid: {"label": v.label, "purity": v.purity, "n": v.n} for lid, v in self.votes.items()}


class TicketRetriever:
    def __init__(
        self,
        index_path: Path,
        pool_path: Path,
        encode_fn=None,  # برای تست: fn(text) -> بردارِ نرمال‌شده (np.ndarray[dim])
    ) -> None:
        data = np.load(index_path, allow_pickle=False)
        keys = [str(k) for k in data["keys"]]
        emb = data["emb"].astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
        self.model_name = str(data["model"]) if "model" in data else "BAAI/bge-m3"
        # backend و پیشوندِ کوئری داخلِ ایندکس ذخیره می‌شوند تا انکودرِ زمانِ اجرا
        # همیشه با مدلِ سازندهٔ ایندکس یکی باشد (m3 → FlagEmbedding؛ st → sentence-transformers).
        self.model_kind = str(data["kind"]) if "kind" in data else "m3"
        self.query_prefix = str(data["prefix"]) if "prefix" in data else ""

        rows_by_key = {r["key"]: r for r in _load_jsonl(pool_path)}
        keep = [i for i, k in enumerate(keys) if k in rows_by_key]
        if len(keep) < len(keys):
            log.warning("%d بردارِ ایندکس در pool یافت نشد و حذف شد.", len(keys) - len(keep))
        self.emb = emb[keep]
        self.rows = [rows_by_key[keys[i]] for i in keep]
        self.layer_ids = [k for k in ("layer1", "layer2") if self.rows and k in self.rows[0]]

        self._encode_fn = encode_fn
        self._model = None
        self._lock = threading.Lock()
        # پروبِ وابستگیِ انکودر همین‌جا: اگر نصب نیست، همین‌جا خطا بده تا
        # maybe_build_retriever یک‌بار retrieval را غیرفعال کند (نه خطا در هر درخواست).
        if self._encode_fn is None:
            if self.model_kind == "m3":
                import FlagEmbedding  # noqa: F401
                import torch  # noqa: F401
            else:
                import sentence_transformers  # noqa: F401
        log.info("ایندکسِ retrieval بارگذاری شد: %d تیکت، مدل=%s", len(self.rows), self.model_name)

    # ---- encoding ----
    def _encode(self, text: str) -> np.ndarray:
        if self._encode_fn is not None:
            v = np.asarray(self._encode_fn(text), dtype=np.float32)
        elif self.model_kind == "m3":
            with self._lock:
                if self._model is None:
                    t0 = time.perf_counter()
                    import torch
                    from FlagEmbedding import BGEM3FlagModel

                    self._model = BGEM3FlagModel(
                        self.model_name, use_fp16=torch.cuda.is_available()
                    )
                    log.info("مدلِ embedding بارگذاری شد (%.1fs)", time.perf_counter() - t0)
                out = self._model.encode(
                    [text], batch_size=1, max_length=_ENCODE_MAX_LEN,
                    return_dense=True, return_sparse=False,
                )
            v = np.asarray(out["dense_vecs"][0], dtype=np.float32)
        else:  # ایندکسِ ساخته‌شده با sentence-transformers (مثلاً e5-large)
            with self._lock:
                if self._model is None:
                    t0 = time.perf_counter()
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
                    self._model.max_seq_length = _ENCODE_MAX_LEN
                    log.info("مدلِ embedding بارگذاری شد (%.1fs)", time.perf_counter() - t0)
                v = self._model.encode(
                    [self.query_prefix + text], normalize_embeddings=True, convert_to_numpy=True
                )[0].astype(np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    @staticmethod
    def build_query_text(
        summary: str, description: str, clarifications: list[tuple[str, str]] | None = None
    ) -> str:
        """همان پاک‌سازیِ ایندکس روی کوئری + پاسخ‌های شفاف‌سازی (اطلاعاتِ جدیدِ کاربر)."""
        s, _ = clean_text(summary or "")
        d, _ = clean_text(description or "")
        parts = [f"{s}. {d}".strip(". ") if s else d]
        for _q, a in clarifications or []:
            a_clean, _ = clean_text(a)
            if a_clean:
                parts.append(a_clean)
        return " ".join(p for p in parts if p)[:_QUERY_MAX_CHARS]

    # ---- retrieval ----
    def retrieve(
        self,
        summary: str,
        description: str,
        clarifications: list[tuple[str, str]] | None = None,
        *,
        k_demos: int | None = None,
        purity_k: int | None = None,
        sim_floor: float | None = None,
        exclude_keys: frozenset[str] | set[str] | None = None,
        drop_self_sim: float | None = None,  # در ارزیابی: حذفِ شبه‌خودی‌ها (مثلاً 0.995)
    ) -> RetrievalResult | None:
        k_demos = k_demos or settings.retrieval_k_demos
        purity_k = purity_k or settings.retrieval_purity_k
        sim_floor = settings.retrieval_sim_floor if sim_floor is None else sim_floor

        text = self.build_query_text(summary, description, clarifications)
        if not text:
            return None
        q = self._encode(text)
        sims = self.emb @ q

        order = np.argsort(-sims)
        picked: list[int] = []
        excl = {str(k) for k in exclude_keys or ()}
        for i in order:
            if len(picked) >= max(purity_k, k_demos):
                break
            if self.rows[i]["key"] in excl:
                continue
            if drop_self_sim is not None and sims[i] >= drop_self_sim:
                continue
            picked.append(int(i))

        if not picked or float(sims[picked[0]]) < sim_floor:
            return None  # تیکتِ بی‌سابقه → سابقهٔ بی‌ربط تزریق نکن

        votes: dict[str, LayerVote] = {}
        vote_ids = picked[:purity_k]
        for lid in self.layer_ids:
            weights: dict[str, float] = {}
            for i in vote_ids:
                lbl = self.rows[i].get(lid)
                if lbl:
                    weights[lbl] = weights.get(lbl, 0.0) + max(float(sims[i]), 0.0)
            total = sum(weights.values())
            if not weights or total <= 0:
                continue  # هیچ همسایهٔ با وزنِ مثبت — رای معنا ندارد
            best = max(weights, key=weights.get)
            votes[lid] = LayerVote(label=best, purity=weights[best] / total, n=len(vote_ids))

        demo_ids = picked[:k_demos]
        return RetrievalResult(
            demos=[self.rows[i] for i in demo_ids],
            demo_sims=[float(sims[i]) for i in demo_ids],
            votes=votes,
            top_similarity=float(sims[picked[0]]),
        )


def _load_jsonl(path: Path) -> list[dict]:
    import json

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def maybe_build_retriever() -> TicketRetriever | None:
    """ساختِ امنِ بازیاب: هر خطا فقط retrieval را غیرفعال می‌کند، نه سرویس را."""
    if not settings.retrieval_enabled:
        return None
    try:
        if not settings.retrieval_index_path.exists():
            log.warning(
                "ایندکسِ retrieval موجود نیست (%s) — دسته‌بندی بدونِ سابقه ادامه می‌یابد. "
                "ساخت: python -m scripts.build_retrieval_index",
                settings.retrieval_index_path,
            )
            return None
        return TicketRetriever(settings.retrieval_index_path, settings.retrieval_pool_path)
    except Exception as e:
        log.warning("retrieval غیرفعال شد: %s", e)
        return None
