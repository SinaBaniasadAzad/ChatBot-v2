"""
داشبوردِ دقتِ حرفه‌ای (برای ارائه) — روی نتایجِ eval_incdb.run_evaluation.

نمایش:
  • کارت‌های KPI: دقتِ کل (هر دو لایه) + دقتِ هر لایه
  • نمودارِ recall هر کلاس (Incident / Service Request / ERP / Staff)
  • Confusion matrix هر لایه (heatmap)
  • نوار پایین: مدل، تعداد نمونه، توکن/هزینه، تاریخ

استفاده روی Kaggle (یک سلول):
    from scripts.report import evaluate_and_report
    res, fig = evaluate_and_report(
        "data/INC_DB.jsonl", balanced=75, workers=6,
        save_path="/kaggle/working/accuracy_report.png",
    )

خروجی هم inline نمایش داده می‌شود، هم PNGِ باکیفیت برای اسلاید ذخیره می‌شود.
"""
from __future__ import annotations

from datetime import date

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from scripts.eval_incdb import print_text_report, run_evaluation
from src.reporting.cost import Pricing, breakdown_from_eval

# پالت
_INK = "#0f172a"
_MUTE = "#64748b"
_GRID = "#e2e8f0"
_LAYER_COLORS = ["#4f46e5", "#0d9488", "#b45309", "#9333ea"]  # برای لایه‌های ۱،۲،…

# رنگِ اجزای هزینه (هم‌خوان با گزارشِ HTML)
_C_HIT = "#0d9488"     # ورودیِ cache-hit
_C_MISS = "#d97706"    # ورودیِ cache-miss
_C_OUT = "#4f46e5"     # خروجی/completion
_C_SAVE = "#16a34a"    # صرفه‌جویی


def _grade(v: float) -> str:
    """رنگِ نمره: خوب/متوسط/ضعیف."""
    if v >= 0.90:
        return "#0d9488"
    if v >= 0.80:
        return "#d97706"
    return "#dc2626"


def _tint(hex_color: str, f: float = 0.12) -> tuple:
    """نسخهٔ روشن (آمیخته با سفید)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


def _draw_kpis(ax, res: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ov = res["overall"]
    tiles = [("Overall accuracy\n(both layers correct)", ov["accuracy"], f"{ov['correct']} / {ov['total']}", True)]
    for i, L in enumerate(res["layers"], 1):
        tiles.append((f"{L['name']}  ·  Layer {i}", L["accuracy"], f"{L['correct']} / {L['total']}", False))

    n = len(tiles)
    gap = 0.025
    w = (1 - gap * (n - 1)) / n
    for i, (label, value, sub, primary) in enumerate(tiles):
        x = i * (w + gap)
        accent = _grade(value)
        if primary:  # سرتیترِ خنثیٰ و مقتدر (سرمه‌ای) + نقطهٔ رنگیِ نمره
            face, edge, lw = "#1f2937", "#1f2937", 0
            num_color, lbl_color, sub_color = "white", "white", "#cbd5e1"
        else:  # کارتِ هر لایه: ته‌رنگِ نمره + عددِ هم‌رنگ
            face, edge, lw = _tint(accent, 0.14), accent, 1.4
            num_color, lbl_color, sub_color = accent, _INK, _MUTE
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.06), w, 0.88,
                boxstyle="round,pad=0,rounding_size=0.035",
                linewidth=lw, edgecolor=edge, facecolor=face, mutation_aspect=0.5,
            )
        )
        ax.text(x + w / 2, 0.78, label, ha="center", va="center", fontsize=11,
                color=lbl_color, fontweight="bold", linespacing=1.25)
        ax.text(x + w / 2, 0.45, f"{value*100:.1f}%", ha="center", va="center",
                fontsize=33, color=num_color, fontweight="bold")
        if primary:  # نقطهٔ رنگیِ نمره روی کارتِ سرمه‌ای
            ax.scatter([x + 0.06], [0.78], s=90, color=accent, zorder=5)
        ax.text(x + w / 2, 0.16, sub, ha="center", va="center", fontsize=11, color=sub_color)


def _draw_recall(ax, res: dict) -> None:
    rows = []  # (class_name, recall, correct, total, color)
    for li, L in enumerate(res["layers"]):
        color = _LAYER_COLORS[li % len(_LAYER_COLORS)]
        for c in L["classes"]:
            rows.append((c["name"], c["recall"], c["correct"], c["total"], color, L["name"]))
    rows.reverse()  # اولین کلاس بالا

    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[4] for r in rows], height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11.5, color=_INK)
    for yi, r in zip(y, rows):
        ax.text(min(r[1] + 0.015, 1.0), yi, f"{r[1]*100:.1f}%  ({r[2]}/{r[3]})",
                va="center", ha="left", fontsize=10.5, color=_INK, fontweight="bold")
    ax.axvline(0.90, color="#94a3b8", ls="--", lw=1, zorder=2)
    ax.text(0.90, len(rows) - 0.35, " target 90%", color="#94a3b8", fontsize=9, va="bottom")
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10, color=_MUTE)
    ax.set_title("Per-class recall  ·  share of each true class predicted correctly",
                 fontsize=13, fontweight="bold", color=_INK, loc="left", pad=10)
    ax.xaxis.grid(True, color=_GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(length=0)

    # افسانهٔ لایه‌ها
    handles = [plt.Rectangle((0, 0), 1, 1, color=_LAYER_COLORS[i % len(_LAYER_COLORS)]) for i in range(len(res["layers"]))]
    ax.legend(
    handles,
    [f"{L['name']} (layer {L['id']})" for L in res["layers"]],
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    frameon=False,
    fontsize=9.5
)


def _draw_confusion(ax, L: dict) -> None:
    ids = L["label_ids"]
    name = {c["id"]: c["name"] for c in L["classes"]}
    has_none = any(L["confusion"].get(t, {}).get(None, 0) for t in ids)
    cols = list(ids) + ([None] if has_none else [])
    col_labels = [name[c] if c is not None else "∅ none" for c in cols]

    M = np.array([[L["confusion"].get(tr, {}).get(p, 0) for p in cols] for tr in ids], dtype=float)
    row_sums = M.sum(axis=1, keepdims=True)
    norm = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)

    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9.5, rotation=20, ha="right", color=_INK)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([name[t] for t in ids], fontsize=9.5, color=_INK)
    for i in range(len(ids)):
        for j in range(len(cols)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if norm[i, j] > 0.5 else _INK, fontweight="bold")
    ax.set_title(f"Confusion — {L['name']}", fontsize=12, fontweight="bold", color=_INK, pad=8)
    ax.set_xlabel("Predicted", fontsize=10, color=_MUTE)
    ax.set_ylabel("True", fontsize=10, color=_MUTE)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_cost_tokens(ax, b) -> None:
    """نوارِ افقیِ انباشتهٔ ترکیبِ توکن: cached / uncached input + output."""
    segs = [
        ("Cached input", b.cache_hit_tokens, _C_HIT),
        ("Uncached input", b.cache_miss_tokens, _C_MISS),
        ("Output", b.completion_tokens, _C_OUT),
    ]
    total = b.total_tokens or 1
    left = 0.0
    for _, val, color in segs:
        ax.barh(0, val, left=left, color=color, height=0.8, zorder=3)
        left += val
    ax.set_xlim(0, total)
    ax.set_ylim(-2.95, 0.6)
    ax.axis("off")
    ax.set_title("Token composition  ·  cached vs uncached input + output",
                 fontsize=13, fontweight="bold", color=_INK, loc="left", pad=10)
    # افسانه با شمارش و درصد — هر آیتم در یک خطِ جدا (بدونِ هم‌پوشانی)
    for i, (name, val, color) in enumerate(segs):
        pct = val / total * 100
        y = -0.95 - i * 0.62
        ax.scatter([total * 0.006], [y], s=70, marker="s", color=color, zorder=4)
        ax.text(total * 0.02, y, f"{name}", va="center", ha="left",
                fontsize=10.5, color=_MUTE)
        ax.text(total * 0.42, y, f"{val:,.0f}  ·  {pct:.1f}%", va="center", ha="left",
                fontsize=10.5, color=_INK, fontweight="bold")


def _tiles(ax, items, ncol: int) -> None:
    """شبکهٔ کاشی‌های KPI روی یک محور. items = [(label, value, accent), ...]."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nrow = -(-len(items) // ncol)  # ceil
    gx, gy = 0.035, 0.12
    w = (1 - gx * (ncol - 1)) / ncol
    h = (1 - gy * (nrow - 1)) / nrow
    for idx, (label, val, accent) in enumerate(items):
        r, c = divmod(idx, ncol)
        x = c * (w + gx)
        y = 1 - (r + 1) * h - r * gy
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                linewidth=1.4, edgecolor=accent, facecolor=_tint(accent, 0.13), mutation_aspect=0.42,
            )
        )
        ax.text(x + w / 2, y + h * 0.70, label, ha="center", va="center",
                fontsize=10, color=_INK, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.33, val, ha="center", va="center",
                fontsize=19, color=accent, fontweight="bold")


def _draw_cost_kpis(ax, b) -> None:
    save_label = f"Saved by cache  (−{b.cache_savings_pct*100:.0f}%)"
    _tiles(
        ax,
        [
            ("Total cost", f"${b.cost_total:,.2f}", _C_OUT),
            ("Cost / ticket", f"${b.cost_per_ticket:.4f}", _C_HIT),
            ("Cache-hit rate", f"{b.cache_hit_rate*100:.0f}%", _C_MISS),
            (save_label, f"${b.cache_savings:,.2f}", _C_SAVE),
        ],
        ncol=2,
    )


def render_dashboard(res: dict, *, dataset_name: str = "", prices=(0.14, 0.0028, 0.28)):
    layers = res["layers"]
    b = breakdown_from_eval(res, pricing=Pricing.from_tuple(prices))

    fig = plt.figure(figsize=(13.0, 15.8), facecolor="white")
    gs = fig.add_gridspec(
        4, 1, height_ratios=[0.70, 1.05, 1.12, 0.74], hspace=0.55,
        left=0.07, right=0.95, top=0.885, bottom=0.055,
    )

    # سرتیتر
    fig.text(0.07, 0.952, "Ticket Classification — Accuracy & Cost Report",
             fontsize=22, fontweight="bold", color=_INK)
    sub = f"Model: {res.get('model') or '—'}    ·    Tickets evaluated: {res['n']}"
    if dataset_name:
        sub += f"    ·    Dataset: {dataset_name}"
    sub += f"    ·    {date.today().isoformat()}"
    fig.text(0.07, 0.917, sub, fontsize=11.5, color=_MUTE)

    _draw_kpis(fig.add_subplot(gs[0]), res)
    _draw_recall(fig.add_subplot(gs[1]), res)

    sub_gs = gs[2].subgridspec(1, len(layers), wspace=0.32)
    for i, L in enumerate(layers):
        _draw_confusion(fig.add_subplot(sub_gs[0, i]), L)

    # ردیفِ هزینه/توکن: چپ = ترکیبِ توکن، راست = کاشی‌های KPIِ هزینه
    cost_gs = gs[3].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.16)
    _draw_cost_tokens(fig.add_subplot(cost_gs[0, 0]), b)
    _draw_cost_kpis(fig.add_subplot(cost_gs[0, 1]), b)

    # نوار پایین — مفروضاتِ قیمت
    p = b.pricing
    foot = (
        f"Single-shot evaluation (no clarifying questions)   ·   "
        f"tokens: input {b.prompt_tokens:,} (hit {b.cache_hit_tokens:,} / miss {b.cache_miss_tokens:,}) / "
        f"output {b.completion_tokens:,}   ·   "
        f"pricing /1M: in ${p.input_per_m} · hit ${p.cache_hit_per_m} · out ${p.output_per_m} "
        f"(verify current pricing)   ·   avg latency {res['latency_ms_avg']:.0f} ms/ticket"
    )
    fig.text(0.07, 0.022, foot, fontsize=9.5, color=_MUTE)
    return fig


def evaluate_and_report(
    data_path,
    *,
    limit=None,
    balanced=None,
    frac=None,
    seed=42,
    workers=4,
    out_path=None,
    errors_out=None,
    save_path=None,
    html_path=None,
    dataset_name=None,
    prices=(0.14, 0.0028, 0.28),
    show=True,
):
    """
    یک اجرای واقعی → همهٔ خروجی‌ها (بدونِ مصرفِ دوبارهٔ API). خروجی: (res, fig).

    آرگومان‌های ذخیره‌سازی (هرکدام داده شود نوشته می‌شود):
      • out_path    : همهٔ پیش‌بینی‌ها (JSONL)
      • errors_out  : فقط تیکت‌های اشتباه + متن (JSONL)
      • save_path   : داشبوردِ دقت+هزینه (PNG)
      • html_path   : گزارشِ HTMLِ مستقلِ هزینه/توکن
    """
    res = run_evaluation(
        data_path, limit=limit, balanced=balanced, frac=frac, seed=seed, workers=workers,
        out_path=out_path, errors_out=errors_out,
    )
    name = dataset_name or str(data_path)
    fig = render_dashboard(res, dataset_name=name, prices=prices)
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
        print("saved:", save_path)
    if html_path:
        # importِ تنبل تا وابستگیِ متقابل و بارِ اضافه نباشد.
        from pathlib import Path

        from scripts.cost_report import render_html

        breakdown = breakdown_from_eval(res, pricing=Pricing.from_tuple(prices))
        Path(html_path).write_text(
            render_html(breakdown, dataset_name=Path(name).name), encoding="utf-8"
        )
        print("saved:", html_path)
    if show:
        plt.show()
    print_text_report(res, *prices)
    return res, fig


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("data_path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--balanced", type=int, default=None)
    ap.add_argument("--frac", type=float, default=None, help="fraction per combo, e.g. 0.2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--errors", default=None, help="ذخیرهٔ تیکت‌های اشتباه + متن (JSONL)")
    ap.add_argument("--save", default="accuracy_report.png")
    a = ap.parse_args()
    evaluate_and_report(
        a.data_path, limit=a.limit, balanced=a.balanced, frac=a.frac, workers=a.workers,
        out_path=a.out, errors_out=a.errors, save_path=a.save, show=False,
    )
