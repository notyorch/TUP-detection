#!/usr/bin/env python3
"""Calibrate injection thresholds per dataset using score cache.

Usage:
  python scripts/calibrate_thresholds.py \\
    --dataset notebooks/data/external/deepset.yaml \\
    --scores-cache notebooks/data/scores/deepset-scores.json \\
    --out notebooks/data/calibration/deepset-calib.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pint_common import load_dataset, pint_score


def sweep_thresholds(
    scores: List[float],
    labels: List[bool],
    step: float = 0.01,
) -> Dict[float, Dict[str, Any]]:
    """Sweep thresholds and compute metrics at each point."""
    results: Dict[float, Dict[str, Any]] = {}

    for threshold in tqdm.tqdm(
        [i * step for i in range(int(1.0 / step) + 1)],
        desc="Threshold sweep",
    ):
        preds = [s >= threshold for s in scores]
        scored = pd.DataFrame({"label": labels, "prediction": preds})
        scored["correct"] = scored["prediction"] == scored["label"]

        bal = pint_score(scored, "balanced")
        imb = pint_score(scored, "imbalanced")
        overall = float(scored["correct"].mean())
        attacks = scored[scored["label"]]
        benign = scored[~scored["label"]]

        results[round(threshold, 3)] = {
            "threshold": round(threshold, 3),
            "overall_accuracy": round(overall, 4),
            "pint_balanced": round(bal, 4),
            "pint_imbalanced": round(imb, 4),
            "attack_recall": round(float(attacks["prediction"].mean()), 4) if len(attacks) else 0.0,
            "benign_specificity": round(
                float((1 - benign["prediction"].mean()), 4) if len(benign) else 1.0
            ),
            "tp": int((attacks["prediction"]).sum()),
            "fn": int((~attacks["prediction"]).sum()),
            "fp": int((benign["prediction"]).sum()),
            "tn": int((~benign["prediction"]).sum()),
        }
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate injection thresholds per dataset")
    p.add_argument("--dataset", required=True, help="Dataset path (yaml/json)")
    p.add_argument("--scores-cache", required=True, help="JSON list of classifier scores")
    p.add_argument("--out", required=True, help="Output calibration JSON")
    p.add_argument("--step", type=float, default=0.01, help="Threshold sweep step (default 0.01)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load dataset
    df = load_dataset(Path(args.dataset))
    print(f"Loaded {len(df)} rows from {args.dataset}")

    # Load cached scores
    scores_path = Path(args.scores_cache)
    if not scores_path.is_file():
        print(f"Error: scores cache not found: {args.scores_cache}", file=sys.stderr)
        sys.exit(1)
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    if len(scores) != len(df):
        print(f"Error: scores length {len(scores)} != rows {len(df)}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(scores)} scores from {args.scores_cache}")

    labels = df["label"].tolist()

    # Sweep
    print(f"Sweeping thresholds (step={args.step})...")
    results = sweep_thresholds(scores, labels, step=args.step)

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(args.dataset),
        "rows": len(df),
        "attacks": int(df["label"].sum()),
        "benign": int((~df["label"]).sum()),
        "sweep_step": args.step,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved calibration to {out}")

    # Find optimal thresholds for different objectives
    print("\n" + "=" * 70)
    print("RECOMMENDED THRESHOLDS:")
    print("=" * 70)

    best_balanced = max(
        results.items(),
        key=lambda x: x[1]["pint_balanced"],
    )
    print(f"Max PINT balanced (0.5 attack/benign):    {best_balanced[0]} → {best_balanced[1]['pint_balanced']:.2%}")

    best_overall = max(
        results.items(),
        key=lambda x: x[1]["overall_accuracy"],
    )
    print(f"Max Overall Accuracy:                      {best_overall[0]} → {best_overall[1]['overall_accuracy']:.2%}")

    # High-recall point (catch >=80% attacks)
    high_recall = [
        (t, m) for t, m in results.items()
        if m["attack_recall"] >= 0.80
    ]
    if high_recall:
        best_hr = max(high_recall, key=lambda x: x[1]["benign_specificity"])
        print(f"High recall (≥80% attacks, max specificity): {best_hr[0]} → "
              f"recall={best_hr[1]['attack_recall']:.2%} specificity={best_hr[1]['benign_specificity']:.2%}")


if __name__ == "__main__":
    main()
