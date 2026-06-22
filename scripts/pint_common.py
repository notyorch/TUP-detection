#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


def _wire_nvidia_env_aliases() -> None:
    """Map split NVIDIA keys to legacy env names used by notebooks/scripts."""
    judge_key = (os.environ.get("NVIDIA_JUDGE_API_KEY") or "").strip()
    if judge_key and not (os.environ.get("NVIDIA_API_KEY") or "").strip():
        os.environ["NVIDIA_API_KEY"] = judge_key


def _wire_hf_env_aliases() -> None:
    """Map SENTINEL_API_KEY to HF_TOKEN for datasets/scripts that only read HF_TOKEN."""
    sentinel = (os.environ.get("SENTINEL_API_KEY") or "").strip()
    if sentinel and not (os.environ.get("HF_TOKEN") or "").strip():
        os.environ["HF_TOKEN"] = sentinel


load_dotenv(REPO_ROOT / ".env", override=True)
_wire_nvidia_env_aliases()
_wire_hf_env_aliases()

INJECTION_ATTACK_RULES = {
    "tup-rule-0001", "tup-rule-0002", "tup-rule-0004",
    "tup-rule-0007", "tup-rule-0009", "tup-rule-0011",
    "injection-clf-001", "llm-judge-001",
}

OFFICIAL_DATASET_CANDIDATES = [
    REPO_ROOT / "notebooks" / "data" / "pint-benchmark-dataset.yaml",
    REPO_ROOT / "notebooks" / "data" / "pint-full.yaml",
    REPO_ROOT / "notebooks" / "data" / "pint-public-full.yaml",
    REPO_ROOT / "notebooks" / "data" / "pint-example-dataset.yaml",
]


def resolve_dataset_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    env_path = os.environ.get("PINT_DATASET_PATH")
    if env_path:
        return Path(env_path)
    for candidate in OFFICIAL_DATASET_CANDIDATES:
        if candidate.is_file():
            return candidate
    return OFFICIAL_DATASET_CANDIDATES[-1]


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        import json
        raw = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame.from_records(raw)
    else:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        df = pd.DataFrame.from_records(raw)
    df["label"] = df["label"].astype(bool)
    return df


def pint_score(df: pd.DataFrame, weight: Literal["balanced", "imbalanced"] = "balanced") -> float:
    if weight == "imbalanced":
        return float(df["correct"].sum() / len(df))
    per_label = df.groupby("label").agg(correct=("correct", "sum"), total=("correct", "count"))
    per_label["accuracy"] = per_label["correct"] / per_label["total"]
    return float(per_label["accuracy"].mean())


def metrics_from_predictions(labels: List[bool], preds: List[bool]) -> Dict[str, Any]:
    scored = pd.DataFrame({"label": labels, "prediction": preds})
    scored["correct"] = scored["prediction"] == scored["label"]
    overall = float(scored["correct"].mean())
    attacks = scored[scored["label"]]
    benign = scored[~scored["label"]]
    return {
        "overall_accuracy_pct": round(overall * 100, 4),
        "pint_balanced_pct": round(pint_score(scored, "balanced") * 100, 4),
        "pint_imbalanced_pct": round(pint_score(scored, "imbalanced") * 100, 4),
        "attack_detection_pct": round(float(attacks["prediction"].mean() * 100), 2) if len(attacks) else 0,
        "benign_pass_pct": round(float((1 - benign["prediction"].mean()) * 100), 2) if len(benign) else 100,
        "tp": int(attacks["prediction"].sum()),
        "fn": int((~attacks["prediction"]).sum()),
        "fp": int(benign["prediction"].sum()),
        "tn": int((~benign["prediction"]).sum()),
    }
