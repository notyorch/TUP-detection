#!/usr/bin/env python3
import argparse
import json
import re
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tup-manager"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pint_common import INJECTION_ATTACK_RULES, load_dataset, resolve_dataset_path
from tup_manager.benign_guard import should_suppress_classifier_hit
from tup_manager.injection_classifier import InjectionClassifier
from tup_manager.rules_engine import TUPRulesEngine
from tup_manager.text_normalize import enrich_prompt_fields

TAG_PATTERNS: Dict[str, re.Pattern] = {
    "non_english": re.compile(r"[^\x00-\x7F]"),
    "zero_width": re.compile(r"[\u200b-\u200d\ufeff]"),
    "base64_blob": re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),
    "delimiter_padding": re.compile(r"={5,}|[-_*]{8,}"),
    "indirect_exfil": re.compile(r"(?i)(first letter|each letter|one letter at a time|spell out|shout)"),
    "roleplay_jailbreak": re.compile(r"(?i)(act as|you are (?:the|an)|conversation between|from now known as|STAN)"),
    "context_discovery": re.compile(r"(?i)(repeat (?:all )?(?:prior|previous)|what (?:are |is )?your instructions|reconfirm|rephrase)"),
    "password_exfil": re.compile(r"(?i)(password|secret|passeord|credentials)"),
    "typo_obfuscation": re.compile(r"(?i)(instrcution|anwweisung|instruct\b.*\binstruct)"),
}


def tag_sample(text: str) -> List[str]:
    tags: List[str] = []
    for name, pattern in TAG_PATTERNS.items():
        if pattern.search(text):
            tags.append(name)
    if not tags:
        tags.append("unclassified")
    return tags


def make_event(prompt: str, category: Optional[str] = None) -> Dict[str, Any]:
    ev = {
        "event_id": str(uuid.uuid4()),
        "session_id": "pint-gap-analysis",
        "prompt": prompt,
        "response": "",
        "source": "pint-gap-analysis",
        "model_id": "analysis",
    }
    if category:
        ev["category"] = category
    return enrich_prompt_fields(ev)


def layer1_hit(rules_engine: TUPRulesEngine, event: Dict[str, Any]) -> bool:
    alerts = rules_engine.evaluate(event, session_events=None)
    return any(a.get("rule_id") in INJECTION_ATTACK_RULES for a in alerts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine PINT false negatives/positives for Phase 2 tuning")
    p.add_argument("--dataset", default=None)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--engine", choices=("layer1", "classifier", "layer1_classifier"), default="layer1_classifier")
    p.add_argument("--injection-threshold", type=float, default=float(__import__("os").environ.get("INJECTION_THRESHOLD", "0.5")))
    p.add_argument("--scores-cache", default=None, help="JSON list of floats (classifier scores, dataset order)")
    p.add_argument("--export", default=str(REPO_ROOT / "notebooks" / "data" / "pint-gap-analysis.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    df = load_dataset(dataset_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    rules_engine = TUPRulesEngine(str(REPO_ROOT / "policies" / "rules"))
    classifier = InjectionClassifier(enabled=args.engine in ("classifier", "layer1_classifier"), threshold=args.injection_threshold)

    prompts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()
    categories = df["category"].tolist() if "category" in df.columns else [None] * len(df)

    scores: List[float]
    if args.scores_cache and Path(args.scores_cache).is_file():
        scores = json.loads(Path(args.scores_cache).read_text())
        if len(scores) != len(prompts):
            print(f"scores-cache length {len(scores)} != rows {len(prompts)}", file=sys.stderr)
            sys.exit(1)
    elif classifier.available:
        print(f"Scoring {len(prompts)} prompts with classifier...")
        scores = classifier.score_batch(prompts)
    else:
        scores = [0.0] * len(prompts)
        if args.engine != "layer1":
            print(f"Classifier unavailable: {classifier.init_error}", file=sys.stderr)
            sys.exit(1)

    rows: List[Dict[str, Any]] = []
    for idx, (prompt, label, category) in enumerate(zip(prompts, labels, categories)):
        event = make_event(prompt, category)
        l1 = layer1_hit(rules_engine, event)
        score = scores[idx]
        clf = score >= args.injection_threshold
        if clf and should_suppress_classifier_hit(
            category=category,
            classifier_score=score,
            threshold=args.injection_threshold,
        ):
            clf = False
        if args.engine == "layer1":
            pred = l1
        elif args.engine == "classifier":
            pred = clf
        else:
            pred = l1 or clf
        rows.append({
            "index": idx,
            "label": bool(label),
            "category": category,
            "prediction": pred,
            "layer1": l1,
            "classifier": clf,
            "classifier_score": round(scores[idx], 4),
            "text_preview": prompt[:240],
            "tags": tag_sample(prompt),
        })

    fn = [r for r in rows if r["label"] and not r["prediction"]]
    fp = [r for r in rows if not r["label"] and r["prediction"]]
    tp = [r for r in rows if r["label"] and r["prediction"]]
    tn = [r for r in rows if not r["label"] and not r["prediction"]]

    fn_tags = Counter(tag for r in fn for tag in r["tags"])
    fp_tags = Counter(tag for r in fp for tag in r["tags"])

    payload = {
        "dataset": str(dataset_path),
        "engine": args.engine,
        "injection_threshold": args.injection_threshold,
        "rows": len(rows),
        "attacks": int(sum(labels)),
        "benign": int(len(labels) - sum(labels)),
        "summary": {
            "tp": len(tp),
            "fn": len(fn),
            "fp": len(fp),
            "tn": len(tn),
            "fn_tag_counts": dict(fn_tags.most_common()),
            "fp_tag_counts": dict(fp_tags.most_common()),
        },
        "false_negatives": fn[:200],
        "false_positives": fp[:200],
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.export)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 70)
    print("PINT GAP ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {dataset_path} ({len(rows)} rows)")
    print(f"Engine:  {args.engine} @ threshold={args.injection_threshold}")
    print(f"TP={len(tp)} FN={len(fn)} FP={len(fp)} TN={len(tn)}")
    if fn_tags:
        print("\nTop FN tags:")
        for tag, count in fn_tags.most_common(8):
            print(f"  {tag}: {count}")
    if fp_tags:
        print("\nTop FP tags:")
        for tag, count in fp_tags.most_common(8):
            print(f"  {tag}: {count}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
