#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tup-manager"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pint_common import INJECTION_ATTACK_RULES, load_dataset, pint_score, resolve_dataset_path
from tup_manager.benign_guard import should_suppress_classifier_hit
from tup_manager.detection_engine import DetectionEngine
from tup_manager.injection_classifier import InjectionClassifier
from tup_manager.nvidia_judge_engine import NvidiaJudgeEngine
from tup_manager.prompt_segments import enrich_with_segments
from tup_manager.rules_engine import TUPRulesEngine
from tup_manager.text_normalize import enrich_prompt_fields


class BenchmarkRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rules_engine = TUPRulesEngine(str(REPO_ROOT / "policies" / "rules"))
        self.injection_classifier = InjectionClassifier(
            enabled=True,
            threshold=args.injection_threshold,
            model_id=args.injection_model,
            batch_size=args.classifier_batch_size,
        )
        self.judge = NvidiaJudgeEngine(enabled=args.enable_judge)
        self.detection_engine = DetectionEngine(
            str(REPO_ROOT / "policies" / "rules"),
            classifier=self.injection_classifier,
            judge=self.judge,
            injection_threshold=args.injection_threshold,
            use_judge=args.enable_judge,
            detection_mode=args.detection_mode,
            gray_zone_low=args.gray_zone_low,
            gray_zone_high=args.gray_zone_high,
            use_gray_zone_judge=args.enable_gray_zone_judge,
        )
        self._classifier_scores: List[float] = []

    @property
    def effective_threshold(self) -> float:
        if self.args.detection_mode in ("strict", "benchmark"):
            return float(os.environ.get("INJECTION_THRESHOLD_STRICT", "0.15"))
        return self.args.injection_threshold

    def make_event(self, prompt: str, response: str = "", category: Optional[str] = None) -> Dict[str, Any]:
        event = enrich_prompt_fields({
            "event_id": str(uuid.uuid4()),
            "session_id": "pint-benchmark",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "pint-benchmark",
            "model_id": "benchmark",
            "prompt": prompt,
            "response": response,
        })
        if category:
            event["category"] = category
        enrich_with_segments(event)
        return event

    def precompute_classifier_scores(self, events: List[Dict[str, Any]]) -> None:
        if not self.injection_classifier.available:
            self._classifier_scores = [0.0] * len(events)
            return
        prompts = [ev.get("prompt") or "" for ev in events]
        if self.args.detection_mode in ("strict", "benchmark"):
            # Multi-variant max score, with global dedup + parallel HTTP.
            self._classifier_scores = self.injection_classifier.score_variants_batch(
                prompts, progress=True
            )
            return
        self._classifier_scores = self.injection_classifier.score_batch(prompts)

    def layer1_detect(self, event: Dict[str, Any]) -> bool:
        alerts = self.rules_engine.evaluate(event, session_events=None)
        return any(a.get("rule_id") in INJECTION_ATTACK_RULES for a in alerts)

    def classifier_detect(self, idx: int) -> bool:
        if idx >= len(self._classifier_scores):
            return False
        return self._classifier_scores[idx] >= self.effective_threshold

    def layer1_classifier_detect(self, event: Dict[str, Any], idx: int) -> bool:
        if self.layer1_detect(event):
            return True
        if idx >= len(self._classifier_scores):
            return False
        score = self._classifier_scores[idx]
        if score < self.effective_threshold:
            return False
        if should_suppress_classifier_hit(
            category=event.get("category"),
            classifier_score=score,
            threshold=self.effective_threshold,
        ):
            return False
        return True

    def judge_detect(self, event: Dict[str, Any]) -> bool:
        prompt = event.get("prompt") or ""
        return self.judge.is_malicious(prompt, event.get("response") or "")

    def full_stack_detect(self, event: Dict[str, Any], idx: int) -> bool:
        score = self._classifier_scores[idx] if idx < len(self._classifier_scores) else None
        return self.detection_engine.is_attack(event, classifier_score=score)

    def evaluate_engine(
        self,
        name: str,
        fn: Callable[[Dict[str, Any], int], bool],
        events: List[Dict[str, Any]],
        labels: List[bool],
    ) -> Dict[str, Any]:
        preds: List[bool] = []
        for idx, event in enumerate(tqdm.tqdm(events, desc=name[:32])):
            preds.append(fn(event, idx))
        scored = pd.DataFrame({"label": labels, "prediction": preds})
        scored["correct"] = scored["prediction"] == scored["label"]
        bal = pint_score(scored, "balanced")
        imb = pint_score(scored, "imbalanced")
        overall = float(scored["correct"].mean())
        attacks = scored[scored["label"]]
        benign = scored[~scored["label"]]
        return {
            "engine": name,
            "overall_accuracy_pct": round(overall * 100, 4),
            "pint_balanced_pct": round(bal * 100, 4),
            "pint_imbalanced_pct": round(imb * 100, 4),
            "attack_detection_pct": round(float(attacks["prediction"].mean() * 100), 2) if len(attacks) else 0,
            "benign_pass_pct": round(float((1 - benign["prediction"].mean()) * 100), 2) if len(benign) else 100,
            "tp": int((attacks["prediction"]).sum()),
            "fn": int((~attacks["prediction"]).sum()),
            "fp": int((benign["prediction"]).sum()),
            "tn": int((~benign["prediction"]).sum()),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TUP PINT benchmark (Phases 2–3)")
    p.add_argument("--dataset", default=None)
    p.add_argument("--max-samples", type=int, default=int(os.environ.get("PINT_MAX_SAMPLES", "0")))
    p.add_argument("--phase", type=int, choices=(2, 3), default=2, help="2=L1+clf; 3=adds L3+Full")
    p.add_argument("--engines", default=os.environ.get("PINT_ENGINES", "all"))
    p.add_argument("--results-out", default=None)
    p.add_argument("--injection-model", default=os.environ.get("INJECTION_MODEL", "rogue-security/prompt-injection-jailbreak-sentinel-v2"))
    p.add_argument("--injection-threshold", type=float, default=float(os.environ.get("INJECTION_THRESHOLD", "0.5")))
    p.add_argument("--classifier-batch-size", type=int, default=int(os.environ.get("INJECTION_BATCH_SIZE", "32")))
    p.add_argument("--scores-cache", default=None, help="JSON list of precomputed classifier scores (load if present)")
    p.add_argument("--save-scores-cache", default=None, help="Path to write per-row scores (default: <results>.scores.json)")
    p.add_argument("--no-save-scores-cache", action="store_true", help="Disable automatic scores-cache writing")
    p.add_argument("--enable-judge", action="store_true", default=None, help="Force NVIDIA judge on")
    p.add_argument("--disable-judge", action="store_true", help="Skip NVIDIA judge engines")
    p.add_argument("--detection-mode", default=os.environ.get("DETECTION_MODE", "production"),
                  choices=["production", "strict", "benchmark"], help="Detection mode: production (0.71), strict (0.15), benchmark (0.15)")
    p.add_argument("--gray-zone-low", type=float, default=float(os.environ.get("GRAY_ZONE_LOW", "0.15")),
                  help="Gray zone lower bound for judge fallback")
    p.add_argument("--gray-zone-high", type=float, default=float(os.environ.get("GRAY_ZONE_HIGH", "0.85")),
                  help="Gray zone upper bound for judge fallback")
    p.add_argument("--enable-gray-zone-judge", action="store_true", default=os.environ.get("DETECTION_JUDGE_GRAY_ZONE_ONLY") in ("1", "true"),
                  help="Use judge only in gray zone (not below threshold)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.disable_judge:
        args.enable_judge = False
    elif args.enable_judge is None:
        args.enable_judge = args.phase >= 3

    if args.results_out is None:
        suffix = f"phase{args.phase}-smoke" if args.max_samples > 0 else f"phase{args.phase}"
        args.results_out = str(REPO_ROOT / "notebooks" / "data" / f"pint-benchmark-{suffix}.json")

    dataset_path = resolve_dataset_path(args.dataset)
    runner = BenchmarkRunner(args)

    phase_label = (
        "Phase 2 (normalize + FN rules + Sentinel v2)"
        if args.phase == 2
        else "Phase 3 (+ NVIDIA judge L3 + Full stack)"
    )
    print("=" * 70)
    print(f"TUP PINT BENCHMARK — {phase_label}")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Classifier model: {args.injection_model}")
    print(f"Detection mode: {args.detection_mode}")
    effective_threshold = args.injection_threshold
    if args.detection_mode in ("strict", "benchmark"):
        effective_threshold = float(os.environ.get("INJECTION_THRESHOLD_STRICT", "0.15"))
    print(f"Classifier threshold: {args.injection_threshold} (effective: {effective_threshold})")
    if args.enable_gray_zone_judge:
        print(f"Gray zone: [{args.gray_zone_low}, {args.gray_zone_high}] -> judge fallback enabled")

    if runner.injection_classifier.available:
        print(f"Classifier: ready ({runner.injection_classifier.model_id})")
    elif args.scores_cache and Path(args.scores_cache).is_file():
        print(f"Classifier: using scores cache ({args.scores_cache})")
    else:
        print(f"Classifier: UNAVAILABLE — {runner.injection_classifier.init_error}")
        sys.exit(1)

    if args.enable_judge:
        if runner.judge.available:
            print(f"Judge: ready ({runner.judge.model_id}, threshold={runner.judge.threshold})")
        else:
            print(f"Judge: UNAVAILABLE — {runner.judge.init_error or 'disabled'}")

    # Warmup for scale-to-zero (HF Endpoint)
    if runner.injection_classifier.available and runner.injection_classifier.backend in ("hf", "hf_inference", "huggingface"):
        print("Warmup: sending test request to HF Endpoint (scale-to-zero)...")
        try:
            _ = runner.injection_classifier.score("Warmup test")
            print("Warmup: OK")
        except Exception as exc:
            print(f"Warmup: WARNING — {exc}")

    df = load_dataset(dataset_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    print(f"Rows: {len(df)} | attacks={int(df['label'].sum())} benign={int((~df['label']).sum())}")

    prompts = df["text"].astype(str).tolist()
    categories = df["category"].astype(str).tolist() if "category" in df.columns else [None] * len(prompts)
    events = [runner.make_event(p, category=c) for p, c in zip(prompts, categories)]

    t0 = time.time()
    loaded_from_cache = False
    if args.scores_cache and Path(args.scores_cache).is_file():
        cached = json.loads(Path(args.scores_cache).read_text(encoding="utf-8"))
        if len(cached) != len(prompts):
            print(f"scores-cache length {len(cached)} != rows {len(prompts)}", file=sys.stderr)
            sys.exit(1)
        runner._classifier_scores = [float(x) for x in cached]
        loaded_from_cache = True
        print(f"Loaded classifier scores from {args.scores_cache}")
    else:
        print("Scoring prompts with classifier (dedup + parallel HTTP)...")
        runner.precompute_classifier_scores(events)
    scoring_seconds = round(time.time() - t0, 1)
    print(f"Classifier scores ready in {scoring_seconds}s "
          f"(api_calls={runner.injection_classifier.api_calls}, "
          f"failures={runner.injection_classifier.scoring_failures})")

    # Auto-save per-row scores for fast re-runs / re-calibration without API.
    if not loaded_from_cache and not args.no_save_scores_cache:
        cache_path = Path(args.save_scores_cache) if args.save_scores_cache else Path(
            str(args.results_out) + ".scores.json"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(runner._classifier_scores), encoding="utf-8")
        print(f"Saved scores cache: {cache_path}")

    all_engines: Dict[str, Callable[[Dict[str, Any], int], bool]] = {
        "TUP Layer 1 (regex)": lambda ev, idx: runner.layer1_detect(ev),
        "TUP Layer 1+Classifier": lambda ev, idx: runner.layer1_classifier_detect(ev, idx),
        "TUP Classifier only": lambda ev, idx: runner.classifier_detect(idx),
    }
    if args.phase >= 3 and args.enable_judge and runner.judge.available:
        all_engines["TUP Layer 3 (NVIDIA judge)"] = lambda ev, idx: runner.judge_detect(ev)
        all_engines["TUP Full (L1+Clf->L3)"] = lambda ev, idx: runner.full_stack_detect(ev, idx)

    wanted = args.engines.strip().lower()
    if wanted != "all":
        keys = {k.strip() for k in wanted.split(",")}
        engines = {
            k: v for k, v in all_engines.items()
            if any(k.lower().startswith(x) or x in k.lower() for x in keys)
        }
    else:
        engines = all_engines

    labels = df["label"].tolist()
    results = []
    print("\n" + "=" * 70)
    for name, fn in engines.items():
        row = runner.evaluate_engine(name, fn, events, labels)
        results.append(row)
        print(f"\n{name}")
        print(f"  Overall Accuracy: {row['overall_accuracy_pct']:.2f}%")
        print(f"  PINT balanced:    {row['pint_balanced_pct']:.2f}%")
        print(f"  PINT imbalanced:  {row['pint_imbalanced_pct']:.2f}%")
        print(f"  Attack detection: {row['attack_detection_pct']:.1f}% | Benign pass: {row['benign_pass_pct']:.1f}%")
        print(f"  TP={row['tp']} FN={row['fn']} FP={row['fp']} TN={row['tn']}")

    clf = runner.injection_classifier
    payload = {
        "phase": args.phase,
        "dataset": str(dataset_path),
        "rows": len(df),
        "attacks": int(df["label"].sum()),
        "benign": int((~df["label"]).sum()),
        "injection_model": args.injection_model,
        "injection_threshold": args.injection_threshold,
        "effective_threshold": effective_threshold,
        "detection_mode": args.detection_mode,
        "classifier_backend": clf.backend,
        "classifier_endpoint": clf._hf_model_ref,
        "scoring_seconds": scoring_seconds,
        "scoring_from_cache": loaded_from_cache,
        "api_calls": clf.api_calls,
        "scoring_failures": clf.scoring_failures,
        "fail_open": clf.fail_open,
        "gray_zone_low": args.gray_zone_low,
        "gray_zone_high": args.gray_zone_high,
        "gray_zone_judge_enabled": args.enable_gray_zone_judge,
        "judge_model": runner.judge.model_id if runner.judge.available else None,
        "judge_threshold": runner.judge.threshold if runner.judge.available else None,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    out = Path(args.results_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"Saved: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
