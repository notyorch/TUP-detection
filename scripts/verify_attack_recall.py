#!/usr/bin/env python3
"""Verify minimum attack recall on the PINT proxy dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pint_common import resolve_dataset_path


def main() -> None:
    p = argparse.ArgumentParser(description="Verify attack recall target")
    p.add_argument("--results", required=True, help="Benchmark JSON from run_pint_benchmark.py")
    p.add_argument("--engine", default="TUP Layer 1+Classifier")
    p.add_argument("--min-recall-pct", type=float, default=99.2)
    args = p.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    hit = next((r for r in data["results"] if r["engine"] == args.engine), None)
    if not hit:
        print(f"Engine not found: {args.engine}", file=sys.stderr)
        sys.exit(1)

    recall = hit["attack_detection_pct"]
    print(f"Engine: {args.engine}")
    print(f"Attack recall: {recall:.2f}% (TP={hit['tp']} FN={hit['fn']})")
    print(f"PINT balanced: {hit['pint_balanced_pct']:.2f}%")

    if recall + 1e-9 < args.min_recall_pct:
        print(f"FAIL: below target {args.min_recall_pct:.1f}%", file=sys.stderr)
        sys.exit(1)
    print(f"PASS: >= {args.min_recall_pct:.1f}% attack recall")


if __name__ == "__main__":
    main()
