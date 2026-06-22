#!/usr/bin/env python3
"""List false negatives / false positives from a dataset + cached scores.

Powered by the scores cache written by run_pint_benchmark.py — no API calls.

Usage:
  python scripts/error_analysis.py \\
    --dataset notebooks/data/external/deepset.yaml \\
    --scores-cache notebooks/data/external/results/deepset-sentinel-hf.json.scores.json \\
    --threshold 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pint_common import load_dataset


def _snippet(text: str, n: int = 120) -> str:
    flat = " ".join(str(text).split())
    return flat[:n] + ("…" if len(flat) > n else "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FN/FP error analysis from cached scores")
    p.add_argument("--dataset", required=True)
    p.add_argument("--scores-cache", required=True, help="JSON list of per-row scores")
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--limit", type=int, default=0, help="Max rows to print per bucket (0 = all)")
    p.add_argument("--out", default=None, help="Optional JSON dump of the errors")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(Path(args.dataset))
    scores = json.loads(Path(args.scores_cache).read_text(encoding="utf-8"))
    if len(scores) != len(df):
        print(f"scores ({len(scores)}) != rows ({len(df)})", file=sys.stderr)
        sys.exit(1)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(bool).tolist()
    cats = df["category"].astype(str).tolist() if "category" in df.columns else [""] * len(df)

    fns, fps = [], []
    for text, label, score, cat in zip(texts, labels, scores, cats):
        pred = float(score) >= args.threshold
        if label and not pred:
            fns.append({"score": round(float(score), 4), "category": cat, "snippet": _snippet(text)})
        elif not label and pred:
            fps.append({"score": round(float(score), 4), "category": cat, "snippet": _snippet(text)})

    fns.sort(key=lambda r: r["score"], reverse=True)   # near-misses first
    fps.sort(key=lambda r: r["score"], reverse=True)   # most confident FPs first

    def _show(title: str, rows):
        print(f"\n{title} ({len(rows)})")
        print("-" * 78)
        shown = rows[: args.limit] if args.limit else rows
        for r in shown:
            print(f"  {r['score']:.3f}  [{r['category']:<16}] {r['snippet']}")

    print("=" * 78)
    print(f"ERROR ANALYSIS @ threshold={args.threshold}  (dataset={Path(args.dataset).name})")
    print("=" * 78)
    _show("FALSE NEGATIVES (attacks scored below threshold)", fns)
    _show("FALSE POSITIVES (benign scored at/above threshold)", fps)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"threshold": args.threshold, "false_negatives": fns, "false_positives": fps}, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
