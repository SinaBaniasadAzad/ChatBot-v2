"""BM25 (واریانتِ Lucene) — پیاده‌سازی کوچک و بدونِ وابستگی، به‌عنوانِ بیس‌لاینِ واژگانی.

چرا داخلی و نه rank_bm25؟ (۱) حذفِ یک وابستگی برای تست‌های آفلاین، (۲) idf به فرمِ
Lucene: ln(1 + (N-df+0.5)/(df+0.5)) که همیشه مثبت است و به کلاه‌برداریِ epsilon
نیازی ندارد. برای ۱۶۰۰–۱۰هزار سند کاملاً کافی و سریع است.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9؀-ۿ]+")


def tokenize(normalized_text: str) -> list[str]:
    """ورودی باید از قبل normalize شده باشد (lowercase؛ src/utils/normalize.py)."""
    return _TOKEN.findall(normalized_text or "")


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self.doc_len = np.array([len(t) for t in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        # postings: term -> (اندیس اسناد، فراوانی در هر سند)
        df: Counter = Counter()
        tf_maps: list[Counter] = []
        for tokens in corpus_tokens:
            c = Counter(tokens)
            tf_maps.append(c)
            df.update(c.keys())

        self.idf = {
            term: math.log(1.0 + (self.n_docs - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        tmp: dict[str, tuple[list[int], list[float]]] = {t: ([], []) for t in df}
        for i, c in enumerate(tf_maps):
            for term, tf in c.items():
                tmp[term][0].append(i)
                tmp[term][1].append(float(tf))
        for term, (ids, tfs) in tmp.items():
            self.postings[term] = (np.array(ids, dtype=np.int32), np.array(tfs, dtype=np.float32))

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return scores
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for term in set(query_tokens):
            post = self.postings.get(term)
            if post is None:
                continue
            ids, tf = post
            scores[ids] += self.idf[term] * tf * (self.k1 + 1.0) / (tf + norm[ids])
        return scores
