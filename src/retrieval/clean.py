"""پاک‌سازی دیتاست تیکت‌ها برای بنچمارک embedding و retrieval.

سیاستِ پاک‌سازی (مطابق تحلیلِ معماری):
  - سلام/احوال‌پرسیِ *ابتدای* متن و تشکرِ *انتهای* متن حذف می‌شود — بین کلاس‌ها
    مشترک‌اند و فضای فاصلهٔ embedding را فشرده می‌کنند. عبارت‌های میانی دست
    نمی‌خورند (مثل «خواهشمند است ...» که خودش سیگنالِ نوعِ درخواست است).
  - نشانه‌گذاری پیوستِ Jira (مثل !pastedImage_....png|thumbnail!) نویزِ خالص است و
    حذف می‌شود؛ حضورِ پیوست به‌صورت flag ذخیره می‌شود (برای تحلیلِ «کوریِ پیوست»).
  - نرمال‌سازیِ سبک برای embed_text: یکسان‌سازی حروف عربی→فارسی و ارقام، حذف کشیده،
    نیم‌فاصله→فاصله، فشرده‌سازی فاصله‌ها. بزرگی/کوچکی حروف انگلیسی حفظ می‌شود.
  - bm25_text = نرمال‌سازیِ کامل (lowercase؛ از src/utils/normalize.py) برای تطبیق واژگانی.
  - متنِ خیلی بلند (traceback و ...) به max_chars محدود می‌شود؛ بخشِ آغازین آموزنده‌ترین است.
  - تکراری‌های عینی (bm25_text یکسان) حذف می‌شوند تا معیارهای leave-one-out باد نکنند.

خروجی: data/retrieval/tickets_clean.jsonl + cleaning_report.json — یک‌بار تولید
می‌شود و در مخزن می‌ماند تا پاک‌سازی هر بار تکرار نشود.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from src.taxonomy import Taxonomy, load_taxonomy
from src.utils.normalize import normalize as full_normalize

# ---------------------------------------------------------------------------
# نرمال‌سازی سبک (برای متنِ embedding — حروف بزرگ/کوچک حفظ می‌شود)
# ---------------------------------------------------------------------------
_CHAR_MAP = {
    "ي": "ی", "ك": "ک", "ى": "ی", "ۀ": "ه", "ة": "ه",
    "أ": "ا", "إ": "ا", "ؤ": "و",
}
for _i, _d in enumerate("۰۱۲۳۴۵۶۷۸۹"):
    _CHAR_MAP[_d] = str(_i)
for _i, _d in enumerate("٠١٢٣٤٥٦٧٨٩"):
    _CHAR_MAP[_d] = str(_i)
_TRANS = str.maketrans(_CHAR_MAP)
_WS = re.compile(r"\s+")

# نشانه‌گذاری پیوست به سبک Jira:  !name.png|thumbnail!  یا  !image.png!
_ATTACH_MARKUP = re.compile(r"!\S[^!\n]*\.(?:png|jpe?g|gif|bmp|tiff?)[^!\n]*!", re.IGNORECASE)
_ATTACH_LABEL = re.compile(r"attachments?\s*\(images?\)\s*:?", re.IGNORECASE)
# اشاره به پیوست/تصویر در متن (برای flag، نه حذف)
_ATTACH_MENTION = re.compile(r"پیوست|اسکرین|تصویر|عکس|attach|screenshot", re.IGNORECASE)


def light_normalize(text: str) -> str:
    """نرمال‌سازی برای embedding: یکسان‌سازی نویسه‌ها بدونِ lowercase."""
    if not text:
        return ""
    text = text.translate(_TRANS)
    text = text.replace("ـ", "")   # کشیده
    text = text.replace("‌", " ")  # نیم‌فاصله
    return _WS.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# حذفِ سلام و تشکرِ حاشیه‌ای (فقط ابتدای/انتهای متن)
# ---------------------------------------------------------------------------
_LEAD_PHRASES = sorted(
    [
        "با سلام و احترام", "با سلام و عرض ادب", "با عرض سلام و ادب",
        "با عرض ادب و احترام", "با سلام و خسته نباشید", "سلام و احترام",
        "سلام وقت بخیر", "سلام روز بخیر", "با عرض ادب", "با عرض سلام",
        "با سلام", "باسلام", "سلام", "احتراما", "احترام", "با احترام",
        "همکار محترم", "همکاران محترم", "همکاران گرامی", "همکار گرامی",
        "با درود", "درود", "وقت بخیر", "وقتتون بخیر", "روز بخیر",
        "خسته نباشید", "عرض ادب", "به استحضار میرساند",
        "hi", "hello", "dear support team", "dear support", "dear team", "dear",
    ],
    key=len,
    reverse=True,
)
_TAIL_PHRASES = sorted(
    [
        "با تشکر و احترام", "با تشکر", "باتشکر", "با سپاس", "باسپاس",
        "سپاسگزارم", "سپاسگزار", "سپاس", "ممنونم", "ممنون", "متشکرم",
        "تشکر", "با احترام", "ارادتمند", "قربان شما",
        "thanks in advance", "thank you", "thanks", "best regards", "regards",
    ],
    key=len,
    reverse=True,
)
_SEPARATORS = " \t\r\n.,;:!?،؛؟!-–—_)»\"'"


def _is_boundary(ch: str) -> bool:
    return ch in _SEPARATORS


def strip_boilerplate(text: str) -> str:
    """سلام‌های ابتدای متن و تشکرهای انتهای متن را (تکرارشونده) حذف می‌کند."""
    t = text
    for _ in range(8):  # چند عبارت پشت‌سرهم ("همکار محترم باسلام احترام ...")
        t2 = t.lstrip(_SEPARATORS)
        # حرفِ ربطِ بینِ سلام‌ها ("با سلام و خسته نباشید" پس از حذفِ اولی)
        if t2.startswith("و "):
            t2 = t2[2:].lstrip(_SEPARATORS)
        low = t2.lower()
        for p in _LEAD_PHRASES:
            if low.startswith(p) and (len(t2) == len(p) or _is_boundary(t2[len(p)])):
                t2 = t2[len(p):]
                break
        else:
            t = t2
            break
        t = t2
    for _ in range(4):
        t2 = t.rstrip(_SEPARATORS)
        low = t2.lower()
        for p in _TAIL_PHRASES:
            if low.endswith(p) and (len(t2) == len(p) or _is_boundary(t2[-len(p) - 1])):
                t2 = t2[: -len(p)]
                break
        else:
            t = t2
            break
        t = t2
    return t.strip(_SEPARATORS + " ")


def clean_text(text: str) -> tuple[str, bool]:
    """(متنِ پاک، آیا اشاره به پیوست/تصویر داشت؟)"""
    raw = text or ""
    has_attach = bool(_ATTACH_MARKUP.search(raw) or _ATTACH_MENTION.search(raw))
    t = _ATTACH_LABEL.sub(" ", _ATTACH_MARKUP.sub(" ", raw))
    t = light_normalize(t)
    t = strip_boilerplate(t)
    return t, has_attach


# ---------------------------------------------------------------------------
# نگاشتِ برچسب‌های نمایشی ("Incident"/"ERP") به idهای taxonomy
# ---------------------------------------------------------------------------
def _norm_key(k: str) -> str:
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


def _label_maps(tax: Taxonomy):
    layer_by_norm = {_norm_key(L.id): L for L in tax.layers}
    label_map = {
        L.id: {**{lbl.name.strip().lower(): lbl.id for lbl in L.labels},
               **{lbl.id.strip().lower(): lbl.id for lbl in L.labels}}
        for L in tax.layers
    }
    return layer_by_norm, label_map


def map_labels(row: dict, tax: Taxonomy) -> dict[str, str | None]:
    layer_by_norm, label_map = _label_maps(tax)
    out: dict[str, str | None] = {L.id: None for L in tax.layers}
    for gk, gv in (row.get("Labels") or {}).items():
        layer = layer_by_norm.get(_norm_key(gk))
        if layer:
            out[layer.id] = label_map[layer.id].get(str(gv).strip().lower())
    return out


# ---------------------------------------------------------------------------
# خطِ لولهٔ اصلی
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def clean_dataset(
    rows: list[dict],
    tax: Taxonomy | None = None,
    max_chars: int = 1200,
) -> tuple[list[dict], dict]:
    """خروجی: (ردیف‌های پاک، گزارشِ پاک‌سازی)."""
    tax = tax or load_taxonomy()
    seen: dict[str, str] = {}  # bm25_text -> اولین key
    clean_rows: list[dict] = []
    dupes: list[dict] = []
    n_missing_labels = 0
    n_truncated = 0
    n_attach = 0
    len_before: list[int] = []
    len_after: list[int] = []
    label_dist: Counter = Counter()

    for row in rows:
        key = row.get("Key") or row.get("key") or ""
        summary_raw = str(row.get("Summary") or row.get("summary") or "")
        desc_raw = str(row.get("Description") or row.get("description") or "")
        labels = map_labels(row, tax)

        if any(v is None for v in labels.values()):
            n_missing_labels += 1
            continue  # برای بنچمارکِ برچسب‌محور، ردیفِ بدون برچسبِ کامل بی‌فایده است

        summary, att_s = clean_text(summary_raw)
        description, att_d = clean_text(desc_raw)
        has_attachment = att_s or att_d
        n_attach += int(has_attachment)

        embed_text = f"{summary}. {description}".strip(". ") if summary else description
        len_before.append(len(summary_raw) + len(desc_raw))
        truncated = len(embed_text) > max_chars
        n_truncated += int(truncated)
        if truncated:
            embed_text = embed_text[:max_chars]
        len_after.append(len(embed_text))

        bm25_text = full_normalize(embed_text)
        if not bm25_text:
            n_missing_labels += 0  # متنِ خالی؛ عملاً رخ نمی‌دهد ولی ردش می‌کنیم
            continue
        if bm25_text in seen:
            dupes.append({"key": key, "duplicate_of": seen[bm25_text]})
            continue
        seen[bm25_text] = key

        label_dist[tuple(labels[L.id] for L in tax.layers)] += 1
        clean_rows.append(
            {
                "key": key,
                **{L.id: labels[L.id] for L in tax.layers},
                "application_raw": row.get("Application"),
                "summary": summary,
                "description": description,
                "embed_text": embed_text,
                "bm25_text": bm25_text,
                "has_attachment": has_attachment,
                "truncated": truncated,
            }
        )

    report = {
        "n_input": len(rows),
        "n_kept": len(clean_rows),
        "n_missing_labels": n_missing_labels,
        "n_exact_duplicates_dropped": len(dupes),
        "duplicates": dupes,
        "n_truncated": n_truncated,
        "n_with_attachment_refs": n_attach,
        "avg_chars_before": round(sum(len_before) / max(len(len_before), 1), 1),
        "avg_chars_after": round(sum(len_after) / max(len(len_after), 1), 1),
        "label_distribution": {" + ".join(k): v for k, v in sorted(label_dist.items())},
        "max_chars": max_chars,
    }
    return clean_rows, report


def save_clean(rows: list[dict], report: dict, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report_path = out_path.parent / "cleaning_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_clean(path: Path) -> list[dict]:
    return load_raw(path)
