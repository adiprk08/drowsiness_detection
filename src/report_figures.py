"""Generate report figures and tables for the combined deployment model.

The comparison table (`src.compare`) and the threshold sweep (`src.calibrate`)
already cover part of the report's evidence. This script fills the gaps with
four artefacts the written report needs:

1. **Confusion matrix** — MobileNetV2 combined, on the held-out test set,
   one panel per evaluation domain (DDD / UTA / combined). Reads the
   confusion counts straight out of `calibration.json`.
2. **Training curves** — train vs. validation macro-F1 and loss across
   epochs, from `history.json`. Shows the train/val gap and where early
   stopping picked the best epoch.
3. **Dataset summary table** — frame and subject counts for DDD and
   UTA-RLDD, plus the subject-disjoint train/val/test split. Written as
   Markdown for pasting into the report.
4. **Sample face crops** — a montage of alert/drowsy crops from each
   dataset, so the report can show what the model actually sees.

Everything is read-only on the data; outputs land in `artifacts/`.

Usage
-----
    py -m src.report_figures                       # all four
    py -m src.report_figures --figures confusion   # just one
    py -m src.report_figures --figures confusion,curves,table,crops
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ART = Path("artifacts")
COMBINED_DIR = ART / "mobilenet_v2_combined"
DB = Path("data/drowsiness.db")
UTA = Path("data/uta_rldd_frames")

LABELS = ["alert", "drowsy"]
DOMAIN_TITLES = {
    "ddd": "DDD (cabin camera)",
    "uta": "UTA-RLDD (webcam)",
    "combined": "Combined",
}


def _mpl():
    """Import matplotlib with the non-interactive Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# 1. Confusion matrices — combined model, per evaluation domain
# ---------------------------------------------------------------------------

def _plot_confusion(ax, cm: np.ndarray, title: str) -> None:
    """Draw one 2x2 confusion panel: counts + row-normalised percentages."""
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = cm / np.maximum(row_sums, 1)
    ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1], labels=LABELS)
    ax.set_yticks([0, 1], labels=LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j] * 100:.1f}%",
                    ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.5 else "black")


def figure_confusion(out: Path | None = None) -> Path:
    cal_path = COMBINED_DIR / "calibration.json"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"{cal_path} not found — run `py -m src.calibrate "
            f"--model mobilenet_v2 --combined` first."
        )
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    op = cal["operating_points"]["default_0.5"]
    thr = op["threshold"]

    plt = _mpl()
    domains = ["ddd", "uta", "combined"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, d in zip(axes, domains):
        c = op["test"][d]["confusion"]
        cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
        mf1 = op["test"][d]["macro_f1"]
        _plot_confusion(ax, cm, f"{DOMAIN_TITLES[d]}\nmacro-F1 = {mf1:.3f}")
    fig.suptitle(
        f"MobileNetV2 (DDD+UTA combined) - held-out test confusion, "
        f"threshold {thr:.2f}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = out or COMBINED_DIR / "confusion_test.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[report_figures] confusion matrix -> {out}")
    return out


# ---------------------------------------------------------------------------
# 2. Training curves — train vs. val macro-F1 and loss
# ---------------------------------------------------------------------------

def figure_training_curves(out: Path | None = None) -> Path:
    hist_path = COMBINED_DIR / "history.json"
    if not hist_path.exists():
        raise FileNotFoundError(f"{hist_path} not found.")
    h = json.loads(hist_path.read_text(encoding="utf-8"))
    hist = h["history"]
    best = h.get("best_epoch")

    epochs = [e["epoch"] for e in hist]
    train_f1 = [e["train"]["macro_f1"] for e in hist]
    train_loss = [e["train"]["loss"] for e in hist]
    val_f1 = {d: [e["val"][d]["macro_f1"] for e in hist]
              for d in ("ddd", "uta", "combined")}
    val_loss = [e["val"]["combined"]["loss"] for e in hist]

    plt = _mpl()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))

    a1.plot(epochs, train_f1, "o-", color="#1f77b4", label="train")
    a1.plot(epochs, val_f1["combined"], "s-", color="#d62728",
            label="val (combined)")
    a1.plot(epochs, val_f1["ddd"], "^--", color="#2ca02c", alpha=0.8,
            label="val (DDD)")
    a1.plot(epochs, val_f1["uta"], "v--", color="#ff7f0e", alpha=0.8,
            label="val (UTA)")
    if best is not None:
        a1.axvline(best, color="grey", ls=":", label=f"best epoch ({best})")
    a1.set_xlabel("epoch")
    a1.set_ylabel("macro-F1")
    a1.set_title("Macro-F1 per epoch")
    a1.set_ylim(0, 1)
    a1.grid(alpha=0.3)
    a1.legend(loc="lower right", fontsize=8)

    a2.plot(epochs, train_loss, "o-", color="#1f77b4", label="train")
    a2.plot(epochs, val_loss, "s-", color="#d62728", label="val (combined)")
    if best is not None:
        a2.axvline(best, color="grey", ls=":", label=f"best epoch ({best})")
    a2.set_xlabel("epoch")
    a2.set_ylabel("loss")
    a2.set_title("Loss per epoch")
    a2.grid(alpha=0.3)
    a2.legend(loc="upper right", fontsize=8)

    fig.suptitle("MobileNetV2 (DDD+UTA combined) - training history",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = out or COMBINED_DIR / "training_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    gap = train_f1[best - 1] - val_f1["combined"][best - 1] if best else 0.0
    print(f"[report_figures] training curves -> {out}  "
          f"(train-val macro-F1 gap at best epoch: {gap:+.3f})")
    return out


# ---------------------------------------------------------------------------
# 3. Dataset summary table
# ---------------------------------------------------------------------------

def _ddd_counts() -> dict:
    """Per-split, per-label DDD face-frame counts from the SQLite bundle."""
    if not DB.exists():
        raise FileNotFoundError(f"{DB} not found.")
    con = sqlite3.connect(str(DB))
    try:
        rows = con.execute(
            "SELECT split, label, COUNT(*) FROM samples "
            "WHERE source='ddd' AND stream='face' GROUP BY split, label"
        ).fetchall()
        n_subj = con.execute(
            "SELECT COUNT(DISTINCT subject_id) FROM samples "
            "WHERE source='ddd' AND stream='face'"
        ).fetchone()[0]
    finally:
        con.close()
    counts = {s: {0: 0, 1: 0} for s in ("train", "val", "test")}
    for split, label, n in rows:
        if split in counts:
            counts[split][int(label)] = n
    return {"counts": counts, "subjects": int(n_subj)}


def _uta_counts() -> dict:
    """Per-split, per-label UTA frame counts, reproducing the training split."""
    from .uta_rldd import index_uta_rldd, split_uta_subjects
    samples = index_uta_rldd(UTA)
    train_s, val_s, test_s = split_uta_subjects(
        UTA, val_frac=0.15, test_frac=0.15, seed=42)
    split_of = {}
    for sid in train_s:
        split_of[sid] = "train"
    for sid in val_s:
        split_of[sid] = "val"
    for sid in test_s:
        split_of[sid] = "test"
    counts = {s: {0: 0, 1: 0} for s in ("train", "val", "test")}
    for s in samples:
        split = split_of.get(s.subject_id)
        if split:
            counts[split][int(s.label)] += 1
    n_subj = len({s.subject_id for s in samples})
    return {"counts": counts, "subjects": n_subj,
            "split_subjects": {"train": len(train_s), "val": len(val_s),
                               "test": len(test_s)}}


def figure_dataset_table(out: Path | None = None) -> Path:
    ddd = _ddd_counts()
    uta = _uta_counts()

    def tot(d, lbl=None):
        c = d["counts"]
        if lbl is None:
            return sum(c[s][0] + c[s][1] for s in c)
        return sum(c[s][lbl] for s in c)

    ddd_total = tot(ddd)
    uta_total = tot(uta)

    lines = []
    lines.append("# Dataset summary - combined deployment model\n")
    lines.append("Frame counts feeding the MobileNetV2 DDD+UTA model. All\n"
                 "splits are subject-disjoint: no person appears in more\n"
                 "than one split.\n")

    # Table 1 — datasets
    lines.append("## Datasets\n")
    lines.append("| Dataset | Camera domain | Subjects | Alert | Drowsy | Total |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| DDD | cabin camera | {ddd['subjects']} | "
                 f"{tot(ddd, 0):,} | {tot(ddd, 1):,} | {ddd_total:,} |")
    lines.append(f"| UTA-RLDD | webcam / phone | {uta['subjects']} | "
                 f"{tot(uta, 0):,} | {tot(uta, 1):,} | {uta_total:,} |")
    lines.append(f"| **Total** | - | {ddd['subjects'] + uta['subjects']} | "
                 f"{tot(ddd, 0) + tot(uta, 0):,} | "
                 f"{tot(ddd, 1) + tot(uta, 1):,} | "
                 f"{ddd_total + uta_total:,} |")
    lines.append("")
    lines.append("DDD \"subjects\" are per-video pseudo-subject groups used "
                 "for leak-free splitting; UTA subjects are the 48 recorded "
                 "individuals.\n")

    # Table 2 — split
    lines.append("## Train / validation / test split\n")
    lines.append("| Split | DDD frames | UTA frames | Total | "
                 "Drowsy share |")
    lines.append("|---|---|---|---|---|")
    for s in ("train", "val", "test"):
        d_n = ddd["counts"][s][0] + ddd["counts"][s][1]
        u_n = uta["counts"][s][0] + uta["counts"][s][1]
        d_drow = ddd["counts"][s][1] + uta["counts"][s][1]
        total = d_n + u_n
        share = (d_drow / total * 100) if total else 0.0
        lines.append(f"| {s} | {d_n:,} | {u_n:,} | {total:,} | {share:.1f}% |")
    d_all = ddd_total
    u_all = uta_total
    all_drow = tot(ddd, 1) + tot(uta, 1)
    lines.append(f"| **Total** | {d_all:,} | {u_all:,} | "
                 f"{d_all + u_all:,} | "
                 f"{all_drow / (d_all + u_all) * 100:.1f}% |")
    lines.append("")
    ss = uta["split_subjects"]
    lines.append(f"UTA subject split: {ss['train']} train / {ss['val']} val "
                 f"/ {ss['test']} test (seed 42).\n")

    out = out or ART / "dataset_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_figures] dataset table -> {out}")
    print(f"  DDD {ddd_total:,} frames / {ddd['subjects']} groups | "
          f"UTA {uta_total:,} frames / {uta['subjects']} subjects")
    return out


# ---------------------------------------------------------------------------
# 4. Sample face crops — alert / drowsy from each dataset
# ---------------------------------------------------------------------------

def _ddd_sample_images(per_label: int = 2) -> dict[int, list]:
    import cv2
    con = sqlite3.connect(str(DB))
    out: dict[int, list] = {0: [], 1: []}
    try:
        for label in (0, 1):
            # OFFSET past the first frames so we don't always grab frame 0.
            rows = con.execute(
                "SELECT image_bytes FROM samples "
                "WHERE source='ddd' AND stream='face' AND label=? "
                "ORDER BY id LIMIT ? OFFSET 200", (label, per_label)
            ).fetchall()
            for (data,) in rows:
                img = cv2.imdecode(np.frombuffer(data, np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is not None:
                    out[label].append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    finally:
        con.close()
    return out


def _uta_sample_images(per_label: int = 2) -> dict[int, list]:
    import cv2
    from .uta_rldd import index_uta_rldd
    samples = index_uta_rldd(UTA)
    out: dict[int, list] = {0: [], 1: []}
    seen_subjects: dict[int, set] = {0: set(), 1: set()}
    for s in samples:
        lbl = int(s.label)
        if len(out[lbl]) >= per_label:
            continue
        # One crop per subject so the montage shows different people.
        if s.subject_id in seen_subjects[lbl]:
            continue
        img = cv2.imread(str(s.path), cv2.IMREAD_COLOR)
        if img is not None:
            out[lbl].append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            seen_subjects[lbl].add(s.subject_id)
    return out


def figure_sample_crops(out: Path | None = None) -> Path:
    plt = _mpl()
    ddd = _ddd_sample_images(per_label=2)
    uta = _uta_sample_images(per_label=2)

    # 2 rows (DDD, UTA) x 4 cols (alert, alert, drowsy, drowsy)
    rows = [("DDD", ddd), ("UTA-RLDD", uta)]
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for r, (name, imgs) in enumerate(rows):
        ordered = imgs[0] + imgs[1]            # alert crops then drowsy crops
        col_label = ["alert", "alert", "drowsy", "drowsy"]
        for c in range(4):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(ordered):
                ax.imshow(ordered[c])
            ax.set_title(f"{name} - {col_label[c]}", fontsize=9)
        axes[r][0].set_ylabel(name, fontsize=11)
    fig.suptitle("Sample face crops - model input (224x224)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = out or ART / "sample_crops.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[report_figures] sample crops -> {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_FIGURES = {
    "confusion": figure_confusion,
    "curves": figure_training_curves,
    "table": figure_dataset_table,
    "crops": figure_sample_crops,
}


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate report figures for the combined model.")
    p.add_argument("--figures", default="all",
                   help="Comma-separated subset of: "
                        f"{', '.join(_FIGURES)} (default: all).")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.figures.strip().lower() == "all":
        wanted = list(_FIGURES)
    else:
        wanted = [f.strip() for f in args.figures.split(",") if f.strip()]
    unknown = [f for f in wanted if f not in _FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s): {unknown}. "
                         f"choose from {list(_FIGURES)}")
    for name in wanted:
        _FIGURES[name]()
    print(f"[report_figures] done ({len(wanted)} artefact(s)).")


if __name__ == "__main__":
    main()
