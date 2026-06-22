#!/usr/bin/env python3
"""Print comparison table from benchmark JSON result files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_result(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Compare PINT benchmark JSON results")
    p.add_argument("files", nargs="+", help="Benchmark JSON paths")
    p.add_argument("--engine", default="TUP Layer 1+Classifier", help="Engine name to compare")
    args = p.parse_args()

    rows: List[Dict[str, Any]] = []
    for fp in args.files:
        data = load_result(Path(fp))
        hit = next((r for r in data.get("results", []) if r.get("engine") == args.engine), None)
        if not hit:
            engines = [r.get("engine") for r in data.get("results", [])]
            print(f"WARN: {fp} missing engine {args.engine!r}; have {engines}")
            continue
        overall_acc = hit.get("overall_accuracy_pct", 0.0)
        rows.append({
            "dataset": Path(data.get("dataset", fp)).name,
            "rows": data.get("rows"),
            "overall_accuracy_pct": overall_acc,
            "pint_balanced_pct": hit["pint_balanced_pct"],
            "attack_detection_pct": hit["attack_detection_pct"],
            "benign_pass_pct": hit["benign_pass_pct"],
            "tp": hit["tp"],
            "fn": hit["fn"],
            "fp": hit["fp"],
            "file": str(fp),
        })

    print(f"\nEngine: {args.engine}\n")
    print(f"{'Dataset':<28} {'Rows':>6} {'Overall':>10} {'Balanced':>10} {'Recall':>8} {'Benign':>8} {'FN':>4} {'FP':>4}")
    print("-" * 88)
    for r in rows:
        print(
            f"{r['dataset']:<28} {r['rows']:>6} {r['overall_accuracy_pct']:>9.2f}% "
            f"{r['pint_balanced_pct']:>9.2f}% "
            f"{r['attack_detection_pct']:>7.1f}% {r['benign_pass_pct']:>7.1f}% "
            f"{r['fn']:>4} {r['fp']:>4}"
        )


if __name__ == "__main__":
    main()
