#!/usr/bin/env python3
"""Compare TUP Layer 1 vs Sentinel v2 vs combined stack on a dataset."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tup-manager"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pint_common import INJECTION_ATTACK_RULES, load_dataset, pint_score, resolve_dataset_path
from tup_manager.benign_guard import should_suppress_classifier_hit
from tup_manager.injection_classifier import InjectionClassifier
from tup_manager.prompt_segments import enrich_with_segments
from tup_manager.rules_engine import TUPRulesEngine
from tup_manager.text_normalize import enrich_prompt_fields


def effective_threshold(mode: str, cli_threshold: float) -> float:
    if mode in ("strict", "benchmark"):
        return float(os.environ.get("INJECTION_THRESHOLD_STRICT", "0.15"))
    return cli_threshold


def make_event(prompt: str, category: str | None = None) -> Dict[str, Any]:
    event = enrich_prompt_fields({
        "event_id": str(uuid.uuid4()),
        "session_id": "stack-ablation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "stack-ablation",
        "model_id": "benchmark",
        "prompt": prompt,
        "response": "",
    })
    if category:
        event["category"] = category
    enrich_with_segments(event)
    return event


def summarize(name: str, labels: List[bool], preds: List[bool]) -> Dict[str, Any]:
    import pandas as pd

    df = pd.DataFrame({"label": labels, "prediction": preds})
    df["correct"] = df["label"] == df["prediction"]
    attacks = df[df["label"]]
    benign = df[~df["label"]]
    return {
        "stack": name,
        "overall_accuracy_pct": round(float(df["correct"].mean() * 100), 2),
        "pint_balanced_pct": round(pint_score(df, "balanced") * 100, 2),
        "attack_detection_pct": round(float(attacks["prediction"].mean() * 100), 2) if len(attacks) else 0.0,
        "benign_pass_pct": round(float((1 - benign["prediction"].mean()) * 100), 2) if len(benign) else 100.0,
        "tp": int(attacks["prediction"].sum()),
        "fn": int((~attacks["prediction"]).sum()),
        "fp": int(benign["prediction"].sum()),
        "tn": int((~benign["prediction"]).sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TUP / Sentinel / combined stack ablation")
    p.add_argument("--dataset", default=None)
    p.add_argument("--results-out", default=str(REPO_ROOT / "notebooks" / "data" / "external" / "results" / "stack-ablation-synthetic.json"))
    p.add_argument("--scores-cache", default=None)
    p.add_argument("--save-scores-cache", default=None)
    p.add_argument("--injection-model", default=os.environ.get("INJECTION_MODEL", "rogue-security/prompt-injection-jailbreak-sentinel-v2"))
    p.add_argument("--injection-threshold", type=float, default=float(os.environ.get("INJECTION_THRESHOLD", "0.5")))
    p.add_argument("--detection-mode", default=os.environ.get("DETECTION_MODE", "benchmark"),
                   choices=["production", "strict", "benchmark"])
    p.add_argument("--classifier-batch-size", type=int, default=int(os.environ.get("INJECTION_BATCH_SIZE", "32")))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    threshold = effective_threshold(args.detection_mode, args.injection_threshold)
    dataset_path = resolve_dataset_path(args.dataset)
    df = load_dataset(dataset_path)

    rules = TUPRulesEngine(str(REPO_ROOT / "policies" / "rules"))
    clf = InjectionClassifier(
        enabled=True,
        threshold=args.injection_threshold,
        model_id=args.injection_model,
        batch_size=args.classifier_batch_size,
    )
    if not clf.available:
        print(f"Classifier unavailable: {clf.init_error}", file=sys.stderr)
        sys.exit(1)

    print("Warmup...")
    _ = clf.score("warmup")

    events = [make_event(str(r["text"]), str(r.get("category") or "")) for _, r in df.iterrows()]
    labels = df["label"].tolist()
    prompts = [ev["prompt"] for ev in events]

    t0 = time.time()
    cache_path = Path(args.scores_cache) if args.scores_cache else None
    if cache_path and cache_path.is_file():
        scores = [float(x) for x in json.loads(cache_path.read_text(encoding="utf-8"))]
        if len(scores) != len(prompts):
            print(f"Cache length mismatch: {len(scores)} vs {len(prompts)}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded scores from {cache_path}")
    else:
        print("Scoring with Sentinel v2...")
        if args.detection_mode in ("strict", "benchmark"):
            scores = clf.score_variants_batch(prompts, progress=True)
        else:
            scores = clf.score_batch(prompts)
        out_cache = Path(args.save_scores_cache or str(args.results_out) + ".scores.json")
        out_cache.parent.mkdir(parents=True, exist_ok=True)
        out_cache.write_text(json.dumps(scores), encoding="utf-8")
        print(f"Saved scores cache: {out_cache}")

    scoring_seconds = round(time.time() - t0, 1)

    def layer1_pred(ev: Dict[str, Any]) -> bool:
        alerts = rules.evaluate(ev, session_events=None)
        return any(a.get("rule_id") in INJECTION_ATTACK_RULES for a in alerts)

    def sentinel_pred(i: int) -> bool:
        return scores[i] >= threshold

    def combined_pred(ev: Dict[str, Any], i: int) -> bool:
        if layer1_pred(ev):
            return True
        if scores[i] < threshold:
            return False
        return not should_suppress_classifier_hit(
            category=ev.get("category"),
            classifier_score=scores[i],
            threshold=threshold,
        )

    stacks = {
        "TUP Layer 1 only": lambda ev, i: layer1_pred(ev),
        "Sentinel v2 only": lambda ev, i: sentinel_pred(i),
        "TUP + Sentinel v2": lambda ev, i: combined_pred(ev, i),
    }

    results = []
    per_row = []
    for name, fn in stacks.items():
        preds = [fn(ev, i) for i, ev in enumerate(tqdm.tqdm(events, desc=name[:28]))]
        results.append(summarize(name, labels, preds))
        if name == "TUP + Sentinel v2":
            per_row = [
                {
                    "text": ev["prompt"],
                    "label": bool(lab),
                    "category": ev.get("category"),
                    "sentinel_score": float(scores[i]),
                    "tup_l1": layer1_pred(ev),
                    "sentinel_detected": sentinel_pred(i),
                    "combined_detected": preds[i],
                }
                for i, (ev, lab) in enumerate(zip(events, labels))
            ]

    attacks = [(r, lab) for r, lab in zip(per_row, labels) if lab]
    complementarity = {
        "tup_only_catches": sum(1 for r, _ in attacks if r["tup_l1"] and not r["sentinel_detected"]),
        "sentinel_only_catches": sum(1 for r, _ in attacks if r["sentinel_detected"] and not r["tup_l1"]),
        "both_catch": sum(1 for r, _ in attacks if r["tup_l1"] and r["sentinel_detected"]),
        "neither_fn": sum(1 for r, _ in attacks if not r["tup_l1"] and not r["sentinel_detected"]),
        "combined_tp": sum(1 for r, _ in attacks if r["combined_detected"]),
    }

    payload = {
        "benchmark": "stack_ablation",
        "dataset": str(dataset_path),
        "rows": len(df),
        "attacks": int(df["label"].sum()),
        "benign": int((~df["label"]).sum()),
        "injection_model": args.injection_model,
        "effective_threshold": threshold,
        "detection_mode": args.detection_mode,
        "scoring_seconds": scoring_seconds,
        "api_calls": clf.api_calls,
        "scoring_failures": clf.scoring_failures,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "complementarity": complementarity,
        "results": results,
        "samples": per_row,
    }

    out = Path(args.results_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for row in results:
        print(f"{row['stack']}")
        print(f"  Overall: {row['overall_accuracy_pct']:.1f}% | PINT: {row['pint_balanced_pct']:.1f}%")
        print(f"  Attack:  {row['attack_detection_pct']:.1f}% | Benign: {row['benign_pass_pct']:.1f}%")
    print("=" * 60)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
