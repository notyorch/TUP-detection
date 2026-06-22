#!/usr/bin/env python3
import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tup-manager"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pint_common import INJECTION_ATTACK_RULES, load_dataset, metrics_from_predictions, resolve_dataset_path
from tup_manager.injection_classifier import InjectionClassifier
from tup_manager.rules_engine import TUPRulesEngine
from tup_manager.text_normalize import enrich_prompt_fields


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate INJECTION_THRESHOLD on PINT dev split")
    p.add_argument("--dataset", default=None)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--dev-ratio", type=float, default=0.2)
    p.add_argument("--min-threshold", type=float, default=0.25)
    p.add_argument("--max-threshold", type=float, default=0.75)
    p.add_argument("--step", type=float, default=0.025)
    p.add_argument("--scores-out", default=str(REPO_ROOT / "notebooks" / "data" / "pint-classifier-scores.json"))
    p.add_argument("--export", default=str(REPO_ROOT / "notebooks" / "data" / "pint-threshold-calibration.json"))
    p.add_argument("--write-env", default=None, help="Optional path to write INJECTION_THRESHOLD=...")
    return p.parse_args()


def layer1_hit(rules_engine: TUPRulesEngine, event: dict) -> bool:
    alerts = rules_engine.evaluate(event, session_events=None)
    return any(a.get("rule_id") in INJECTION_ATTACK_RULES for a in alerts)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    df = load_dataset(dataset_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    classifier = InjectionClassifier(enabled=True)
    if not classifier.available:
        print(f"Classifier unavailable: {classifier.init_error}", file=sys.stderr)
        sys.exit(1)

    rules_engine = TUPRulesEngine(str(REPO_ROOT / "policies" / "rules"))
    prompts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()

    print(f"Scoring {len(prompts)} prompts...")
    scores = classifier.score_batch(prompts)
    scores_path = Path(args.scores_out)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(scores), encoding="utf-8")

    l1_flags: List[bool] = []
    for prompt in tqdm.tqdm(prompts, desc="Layer 1"):
        ev = enrich_prompt_fields({
            "event_id": str(uuid.uuid4()),
            "prompt": prompt,
            "response": "",
        })
        l1_flags.append(layer1_hit(rules_engine, ev))

    split_at = max(1, int(len(df) * (1 - args.dev_ratio)))
    dev_scores = scores[split_at:]
    dev_l1 = l1_flags[split_at:]
    dev_labels = labels[split_at:]

    sweep = []
    best = None
    threshold = args.min_threshold
    while threshold <= args.max_threshold + 1e-9:
        preds = [l1 or (s >= threshold) for l1, s in zip(dev_l1, dev_scores)]
        m = metrics_from_predictions(dev_labels, preds)
        row = {"threshold": round(threshold, 4), **m}
        sweep.append(row)
        if best is None or m["pint_balanced_pct"] > best["pint_balanced_pct"]:
            best = row
        threshold += args.step

    payload = {
        "dataset": str(dataset_path),
        "dev_rows": len(dev_labels),
        "dev_attacks": int(sum(dev_labels)),
        "scores_cache": str(scores_path),
        "best": best,
        "sweep": sweep,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(args.export)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 70)
    print("THRESHOLD CALIBRATION (dev split)")
    print("=" * 70)
    print(f"Best threshold: {best['threshold']}")
    print(f"  PINT balanced:   {best['pint_balanced_pct']:.2f}%")
    print(f"  Attack detection: {best['attack_detection_pct']:.1f}%")
    print(f"  Benign pass:      {best['benign_pass_pct']:.1f}%")
    print(f"  TP={best['tp']} FN={best['fn']} FP={best['fp']} TN={best['tn']}")
    print(f"Scores cache: {scores_path}")
    print(f"Saved: {out}")

    if args.write_env:
        Path(args.write_env).write_text(
            f"INJECTION_THRESHOLD={best['threshold']}\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.write_env}")


if __name__ == "__main__":
    main()
