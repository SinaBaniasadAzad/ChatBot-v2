"""تست‌های آفلاینِ لایهٔ retrieval (پاک‌سازی، BM25، معیارها) — بدونِ مدل/دانلود."""
from __future__ import annotations

import numpy as np

from src.retrieval import metrics as M
from src.retrieval.bm25 import BM25, tokenize
from src.retrieval.clean import clean_dataset, clean_text, light_normalize, strip_boilerplate
from src.taxonomy import load_taxonomy


# ---------------------------------------------------------------------------
# پاک‌سازی
# ---------------------------------------------------------------------------
def test_light_normalize_unifies_chars_and_digits():
    assert light_normalize("كيفيت") == "کیفیت"
    assert light_normalize("۱۲۳ و ٤٥") == "123 و 45"
    assert light_normalize("تایم‌شیت") == "تایم شیت"
    assert light_normalize("SAP  Access") == "SAP Access"  # حروفِ بزرگ حفظ می‌شود


def test_strip_leading_greetings_iterative():
    s = strip_boilerplate("همکار محترم باسلام احترام لطفا قرارداد را به‌روز کنید")
    assert s.startswith("لطفا")
    s2 = strip_boilerplate("با سلام و خسته نباشید لطفا عنوان شغلی را تغییر دهید")
    assert s2.startswith("لطفا")


def test_strip_trailing_thanks():
    s = strip_boilerplate("پانچ من ثبت نشده است. با تشکر")
    assert s.endswith("نشده است")


def test_greeting_boundary_not_overstripped():
    # «سلامت» نباید به‌عنوانِ «سلام» حذف شود
    s = strip_boilerplate("سلامت سیستم بررسی شود")
    assert s.startswith("سلامت")


def test_midtext_request_phrases_preserved():
    s = strip_boilerplate("با سلام خواهشمند است دسترسی ایجاد شود")
    assert "خواهشمند است" in s


def test_attachment_markup_removed_and_flagged():
    text, has_attach = clean_text(
        "Attachments (images): !pastedImage_9_13.png|thumbnail! ماموریت تایید نمی‌شود"
    )
    assert has_attach
    assert "pastedImage" not in text
    assert "ماموریت" in text


def test_clean_dataset_maps_labels_dedupes_and_truncates():
    tax = load_taxonomy()
    row = {
        "Key": "INC-1",
        "Application": "ERP",
        "Summary": "با سلام مشکل پانچ",
        "Description": "پانچ ثبت نشده است. با تشکر",
        "Labels": {"layer_1": "Incident", "layer_2": "ERP"},
    }
    dup = dict(row, Key="INC-2")
    nolabel = dict(row, Key="INC-3", Labels={"layer_1": "???", "layer_2": "ERP"})
    long_row = dict(
        row,
        Key="INC-4",
        Summary="خطا",
        Description="ارور " * 1000,
        Labels={"layer_1": "Incident", "layer_2": "Staff"},
    )
    clean, report = clean_dataset([row, dup, nolabel, long_row], tax, max_chars=100)

    assert [r["key"] for r in clean] == ["INC-1", "INC-4"]
    assert clean[0]["layer1"] == "incident" and clean[0]["layer2"] == "erp"
    assert report["n_exact_duplicates_dropped"] == 1
    assert report["n_missing_labels"] == 1
    assert clean[1]["truncated"] and len(clean[1]["embed_text"]) == 100


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
def test_bm25_ranks_matching_term_higher():
    corpus = [
        tokenize("مشکل پانچ ورود و خروج"),
        tokenize("درخواست وام صندوق"),
        tokenize("دسترسی تایم شیت اپرور"),
    ]
    bm = BM25(corpus)
    scores = bm.get_scores(tokenize("وام صندوق"))
    assert scores.argmax() == 1
    assert scores[1] > 0 and scores[0] == 0.0


def test_bm25_unknown_term_is_zero():
    bm = BM25([tokenize("الف ب"), tokenize("ج د")])
    assert bm.get_scores(tokenize("ناموجود")).sum() == 0.0


# ---------------------------------------------------------------------------
# معیارها (با embedding مصنوعیِ دو خوشه‌ای)
# ---------------------------------------------------------------------------
def _two_clusters(n_per: int = 10, dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = np.tile(np.eye(dim)[0], (n_per, 1)) + rng.normal(0, 0.01, (n_per, dim))
    b = np.tile(np.eye(dim)[1], (n_per, 1)) + rng.normal(0, 0.01, (n_per, dim))
    emb = np.vstack([a, b]).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = ["x"] * n_per + ["y"] * n_per
    return emb, labels


def test_topk_and_agreement_on_separable_clusters():
    emb, labels = _two_clusters()
    idx, sim = M.top_k_neighbors(emb, k=5)
    codes, classes = M.encode_labels(labels)
    agree = M.agreement_at_k(idx, sim, codes, (1, 3, 5))
    assert agree[1] == 1.0 and agree[5] == 1.0
    assert not np.any(idx == np.arange(len(labels))[:, None])  # حذفِ خود


def test_knn_predict_and_frontier():
    emb, labels = _two_clusters()
    idx, sim = M.top_k_neighbors(emb, k=9)
    codes, classes = M.encode_labels(labels)
    pred, share = M.knn_predict(idx, sim, codes, len(classes), k=9)
    assert M.accuracy(pred, codes) == 1.0
    frontier = M.purity_frontier(share, pred == codes, codes >= 0, (0.6, 0.9))
    assert frontier[0]["coverage"] == 1.0 and frontier[0]["accuracy"] == 1.0
    assert frontier[1]["coverage"] <= frontier[0]["coverage"]  # آستانهٔ بالاتر → پوششِ کمتر


def test_rrf_fusion_and_padding_semantics():
    l1 = np.array([[1, 2, 3]])
    l2 = np.array([[2, 1, 4]])
    idx, w = M.rrf_fuse([l1, l2], k_out=4)
    assert set(idx[0][:4]) >= {1, 2, 4}
    assert w[0][0] >= w[0][1] >= w[0][2]
    # آیتم‌های مشترک در هر دو فهرست باید بالاتر از تک‌فهرستی‌ها بیایند
    assert idx[0][0] in (1, 2) and idx[0][1] in (1, 2)


def test_contradiction_mining_finds_near_dup_with_diff_label():
    emb = np.array([[1.0, 0.0], [1.0, 0.001], [0.0, 1.0]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idx, sim = M.top_k_neighbors(emb, k=2)
    codes1, _ = M.encode_labels(["incident", "service_request", "incident"])
    codes2, _ = M.encode_labels(["erp", "erp", "staff"])
    cons = M.find_contradictions(idx, sim, {"layer1": codes1, "layer2": codes2}, 0.95)
    assert len(cons) == 1
    assert cons[0]["differs_on"] == ["layer1"]
    assert {cons[0]["i"], cons[0]["j"]} == {0, 1}
