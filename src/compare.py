"""Generate a side-by-side comparison of all trained models from saved metrics.

Reads ``artifacts/<model>/test_metrics.json`` for each model we trained and
prints a markdown table to stdout. Also writes ``artifacts/comparison.md`` so
the same numbers are easy to drop straight into the report.

For the three single-stream models we additionally pull each model's
*safety-threshold* operating point (largest threshold that keeps val drowsy
recall ≥ 0.95) from ``calibration.json``, because that's the operating
point a real cabin deployment would actually use — false-negatives are
much more expensive than false-positives in this domain.

The two-stream model gets two rows because its eye and face branches are
evaluated on different test subsets (MRL eye crops vs DDD face crops) —
the numbers aren't directly comparable to a single combined "fused" model
result, so reporting them side-by-side is the honest thing.

Usage
-----
    py -m src.compare                   # print + save with default artifacts/
    py -m src.compare --artifacts ART   # custom artifacts dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


# Display order on the table — matches the report narrative
# (baseline → classical TL → modern TL → fusion).
SINGLE_STREAM = ["baseline_cnn", "alexnet", "mobilenet_v2"]


def _fmt(x: float | int | None, width: int = 6, prec: int = 4) -> str:
    if x is None:
        return "  --  "
    if isinstance(x, int):
        return f"{x:>{width}d}"
    return f"{x:>{width}.{prec}f}"


def _row(name: str, m: dict, *, threshold: str = "0.5") -> dict:
    """Pull the headline numbers we want to display for one model."""
    return {
        "model": name,
        "threshold": threshold,
        "n": m.get("n", 0),
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "f1_drowsy": m.get("f1_drowsy"),
        "recall_drowsy": m.get("recall_drowsy"),
        "precision_drowsy": m.get("precision_drowsy"),
        "roc_auc": m.get("roc_auc"),
    }


def _read_single_stream(art: Path, name: str) -> list[dict]:
    """Default-threshold row plus, if available, the safety-threshold row."""
    rows: list[dict] = []
    test_json = art / name / "test_metrics.json"
    if not test_json.exists():
        return rows
    test = json.loads(test_json.read_text())
    rows.append(_row(name, test, threshold="0.50"))

    cal_json = art / name / "calibration.json"
    if cal_json.exists():
        cal = json.loads(cal_json.read_text())
        safety = cal.get("operating_points", {}).get("recall_\u2265_0.95_on_val")
        if safety is not None:
            t = safety.get("threshold")
            rows.append(_row(
                f"{name} @safety",
                safety["test"],
                threshold=f"{t:.2f}" if isinstance(t, (int, float)) else str(t),
            ))
    return rows


def _read_two_stream(art: Path) -> list[dict]:
    test_json = art / "two_stream" / "test_metrics.json"
    if not test_json.exists():
        return []
    test = json.loads(test_json.read_text())
    rows: list[dict] = []
    if "eye_branch" in test:
        rows.append(_row("two_stream (eye)", test["eye_branch"], threshold="0.50"))
    if "face_branch" in test:
        rows.append(_row("two_stream (face)", test["face_branch"], threshold="0.50"))
    return rows


def _markdown_table(rows: list[dict]) -> str:
    headers = ["Model", "Thr", "N", "Acc", "Macro-F1",
               "F1 (drowsy)", "Recall (drowsy)", "Prec. (drowsy)", "ROC-AUC"]
    sep = "|" + "|".join("---" for _ in headers) + "|"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for r in rows:
        lines.append("| " + " | ".join([
            r["model"],
            r["threshold"],
            f"{r['n']:d}",
            _fmt(r["accuracy"]),
            _fmt(r["macro_f1"]),
            _fmt(r["f1_drowsy"]),
            _fmt(r["recall_drowsy"]),
            _fmt(r["precision_drowsy"]),
            _fmt(r["roc_auc"]),
        ]) + " |")
    return "\n".join(lines)


def _plain_table(rows: list[dict]) -> str:
    """Fixed-width text table for pasting into the terminal / a code block."""
    headers = ["Model", "Thr", "N", "Acc", "mF1",
               "F1d", "Rd", "Pd", "AUC"]
    widths = [22, 6, 6, 7, 7, 7, 7, 7, 7]
    out = []
    line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    out.append(line)
    out.append("-" * len(line))
    for r in rows:
        cells = [
            f"{r['model']:<22}",
            f"{r['threshold']:>6}",
            f"{r['n']:>6d}",
            _fmt(r["accuracy"], width=7, prec=4),
            _fmt(r["macro_f1"], width=7, prec=4),
            _fmt(r["f1_drowsy"], width=7, prec=4),
            _fmt(r["recall_drowsy"], width=7, prec=4),
            _fmt(r["precision_drowsy"], width=7, prec=4),
            _fmt(r["roc_auc"], width=7, prec=4),
        ]
        out.append("  ".join(cells))
    return "\n".join(out)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare trained models side-by-side.")
    p.add_argument("--artifacts", default="artifacts",
                   help="Artifacts directory (default: artifacts)")
    p.add_argument("--out", default=None,
                   help="Markdown output path (default: <artifacts>/comparison.md)")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    art = Path(args.artifacts)

    rows: list[dict] = []
    for name in SINGLE_STREAM:
        rows.extend(_read_single_stream(art, name))
    rows.extend(_read_two_stream(art))

    if not rows:
        print("[compare] no test_metrics.json found under "
              f"{art} — train + eval first")
        return

    print("\n[compare] held-out test results -- single source of truth:\n")
    print(_plain_table(rows))
    print("\nLegend: Thr=decision threshold, N=#samples, mF1=macro-F1, "
          "F1d=F1 on drowsy class, Rd=recall on drowsy, Pd=precision on drowsy.")
    print("'@safety' rows use the largest threshold that keeps val drowsy "
          "recall >= 0.95 (the deployment operating point).\n")

    md_path = Path(args.out) if args.out else art / "comparison.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md = (
        "# Model comparison — held-out test set\n\n"
        "All numbers from `artifacts/<model>/test_metrics.json`. The\n"
        "`@safety` rows pull the safety operating point from\n"
        "`artifacts/<model>/calibration.json` — the largest threshold\n"
        "that keeps validation drowsy recall ≥ 0.95.\n\n"
        f"{_markdown_table(rows)}\n\n"
        "**Two-stream caveat.** The eye and face branches are evaluated on\n"
        "different held-out subsets (MRL eye crops vs DDD face crops)\n"
        "because no test sample carries both modalities — the rows are not\n"
        "directly comparable to each other or to a single fused model.\n"
    )
    md_path.write_text(md, encoding="utf-8")
    print(f"[compare] wrote {md_path}")


if __name__ == "__main__":
    main()
