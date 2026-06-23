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

from scripts.eval_incdb import compute_cost, print_text_report, run_evaluation

# پالت
_INK = "#0f172a"
_MUTE = "#64748b"
_GRID = "#e2e8f0"
_LAYER_COLORS = ["#4f46e5", "#0d9488", "#b45309", "#9333ea"]  # برای لایه‌های ۱،۲،…


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
    ax.legend(handles, [f"{L['name']} (layer {L['id']})" for L in res["layers"]],
              loc="lower right", frameon=False, fontsize=9.5)


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


def render_dashboard(res: dict, *, dataset_name: str = "", prices=(0.27, 0.07, 1.10)):
    layers = res["layers"]
    fig = plt.figure(figsize=(13.0, 13.6), facecolor="white")
    gs = fig.add_gridspec(
        3, 1, height_ratios=[0.78, 1.18, 1.25], hspace=0.42,
        left=0.07, right=0.95, top=0.86, bottom=0.11,
    )

    # سرتیتر
    fig.text(0.07, 0.945, "Ticket Classification — Accuracy Report",
             fontsize=22, fontweight="bold", color=_INK)
    sub = f"Model: {res.get('model') or '—'}    ·    Tickets evaluated: {res['n']}"
    if dataset_name:
        sub += f"    ·    Dataset: {dataset_name}"
    sub += f"    ·    {date.today().isoformat()}"
    fig.text(0.07, 0.905, sub, fontsize=11.5, color=_MUTE)

    _draw_kpis(fig.add_subplot(gs[0]), res)
    _draw_recall(fig.add_subplot(gs[1]), res)

    sub_gs = gs[2].subgridspec(1, len(layers), wspace=0.32)
    for i, L in enumerate(layers):
        _draw_confusion(fig.add_subplot(sub_gs[0, i]), L)

    # نوار پایین
    t = res["tokens"]
    cost = compute_cost(t, *prices)
    foot = (
        f"Single-shot evaluation (no clarifying questions)   ·   "
        f"tokens: prompt {t.get('prompt',0):,} / completion {t.get('completion',0):,}   ·   "
        f"est. cost ${cost:.3f} (verify current pricing)   ·   "
        f"avg latency {res['latency_ms_avg']:.0f} ms/ticket"
    )
    fig.text(0.07, 0.035, foot, fontsize=9.5, color=_MUTE)
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
    dataset_name=None,
    prices=(0.14, 0.0028, 0.28),
    show=True,
):
    """اجرا + رندرِ داشبورد. خروجی: (res, fig)."""
    res = run_evaluation(
        data_path, limit=limit, balanced=balanced, frac=frac, seed=seed, workers=workers,
        out_path=out_path, errors_out=errors_out,
    )
    fig = render_dashboard(res, dataset_name=dataset_name or str(data_path), prices=prices)
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
        print("saved:", save_path)
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
