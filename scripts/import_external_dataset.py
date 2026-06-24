#!/usr/bin/env python3
"""Download public HF datasets and convert to PINT YAML (text, category, label)."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

PRESETS: Dict[str, Dict[str, str]] = {
    "owasp-v2": {
        "hf_id": "PointGuardAI/Prompt-Injection-OWASP-Benchmark-V2",
        "split": "train",
    },
    "antijection": {
        "hf_id": "Antijection/prompt-injection-dataset-v1",
        "split": "train",
    },
    "deepset": {
        "hf_id": "deepset/prompt-injections",
        "split": "all",
    },
}


def _extract_user_prompt(text: str) -> str:
    """Antijection wraps context/user in special tokens; prefer user segment."""
    if "<|user|>" in text:
        parts = text.split("<|user|>", 1)
        if len(parts) > 1:
            user = parts[1].split("<|assistant|>", 1)[0]
            return user.strip()
    if "[USER]" in text and "[/USER]" in text:
        m = re.search(r"\[USER\](.*?)\[/USER\]", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return text.strip()


def _row_to_pint(row: Dict[str, Any], preset: str) -> Optional[Dict[str, Any]]:
    if preset == "owasp-v2":
        text = str(row.get("text") or "").strip()
        if not text:
            return None
        trigger = row.get("should_trigger")
        if trigger is None:
            trigger = 1 if str(row.get("expected_action", "")).lower() == "block" else 0
        label = bool(int(trigger))
        subtype = str(row.get("subtype") or "prompt_injection")
        category = "jailbreak" if "jailbreak" in subtype.lower() else "prompt_injection"
        if not label:
            category = str(row.get("metadata", {}) or {}).get("benign_control_type") or "hard_negatives"
            if isinstance(row.get("metadata"), dict):
                bt = row["metadata"].get("benign_control_type")
                if bt:
                    category = str(bt)
            else:
                category = "hard_negatives"
        return {"text": text, "category": category, "label": label}

    if preset == "antijection":
        text = _extract_user_prompt(str(row.get("prompt") or row.get("text") or ""))
        if not text:
            return None
        raw_label = str(row.get("label") or "").lower()
        label = raw_label in ("malicious", "attack", "injection", "1", "true", "yes")
        category = str(row.get("attack_category") or ("prompt_injection" if label else "chat"))
        if not label:
            category = "hard_negatives"
        return {"text": text, "category": category, "label": label}

    if preset == "deepset":
        text = str(row.get("text") or "").strip()
        if not text:
            return None
        label = bool(int(row.get("label", 0)))
        category = "prompt_injection" if label else "chat"
        return {"text": text, "category": category, "label": label}

    raise ValueError(f"unknown preset: {preset}")


def load_hf_rows(hf_id: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("install datasets: pip install datasets") from exc
    if split == "all":
        bundle = load_dataset(hf_id)
        rows: List[Dict[str, Any]] = []
        for part in bundle.keys():
            rows.extend(dict(row) for row in bundle[part])
        return rows
    ds = load_dataset(hf_id, split=split)
    return [dict(row) for row in ds]


def convert_rows(rows: List[Dict[str, Any]], preset: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        rec = _row_to_pint(row, preset)
        if rec:
            out.append(rec)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import public dataset -> PINT YAML")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), help="Built-in HF preset")
    p.add_argument("--hf-id", help="Override HuggingFace dataset id")
    p.add_argument("--split", default=None)
    p.add_argument("--out", required=True, help="Output path (.yaml or .json)")
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.preset:
        cfg = PRESETS[args.preset]
        hf_id = args.hf_id or cfg["hf_id"]
        split = args.split if args.split is not None else cfg["split"]
        preset = args.preset
    elif args.hf_id:
        hf_id = args.hf_id
        split = args.split
        preset = "owasp-v2" if "owasp" in hf_id.lower() else "antijection"
    else:
        print("Specify --preset or --hf-id", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {hf_id} split={split} ...")
    rows = load_hf_rows(hf_id, split)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    pint_rows = convert_rows(rows, preset)
    attacks = sum(1 for r in pint_rows if r["label"])
    benign = len(pint_rows) - attacks

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        import json
        out.write_text(json.dumps(pint_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        out.write_text(
            yaml.safe_dump(pint_rows, allow_unicode=True, sort_keys=False, width=120),
            encoding="utf-8",
        )
    print(f"Saved {len(pint_rows)} rows -> {out}")
    print(f"  attacks={attacks} benign={benign}")


if __name__ == "__main__":
    main()
