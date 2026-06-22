"""Optional ensemble with Llama Prompt Guard 2 (lightweight secondary classifier)."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

_log = logging.getLogger("tup.ensemble_classifier")

DEFAULT_SECONDARY_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"


class EnsembleClassifier:
    """Combine Sentinel v2 (primary) with Llama Prompt Guard 2 (secondary).

    Ensemble rule: attack if primary >= t1 OR secondary >= t2.
    Feature-flagged: ENSEMBLE_ENABLED=false by default (no impact on baseline).
    """

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        primary_threshold: Optional[float] = None,
        secondary_model: Optional[str] = None,
        secondary_threshold: Optional[float] = None,
        backend: Optional[str] = None,
    ) -> None:
        raw_enabled = enabled if enabled is not None else os.getenv("ENSEMBLE_ENABLED", "false")
        self.enabled = str(raw_enabled).lower() in ("1", "true", "yes")
        self.primary_threshold = (
            primary_threshold
            if primary_threshold is not None
            else float(os.getenv("INJECTION_THRESHOLD", "0.5"))
        )
        self.secondary_model = secondary_model or os.getenv(
            "ENSEMBLE_SECONDARY_MODEL", DEFAULT_SECONDARY_MODEL
        )
        self.secondary_threshold = secondary_threshold or float(
            os.getenv("ENSEMBLE_SECONDARY_THRESHOLD", "0.5")
        )
        self.backend = (backend or os.getenv("ENSEMBLE_BACKEND", "local")).lower()
        self._tokenizer = None
        self._model = None
        self._malicious_index: Optional[int] = None
        self._available = False
        self._init_error: Optional[str] = None
        self._lock = threading.Lock()
        if self.enabled:
            self._ensure_loaded()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def _ensure_loaded(self) -> None:
        if self._available or self._init_error:
            return
        with self._lock:
            if self._available or self._init_error:
                return
            if self.backend != "local":
                self._init_error = f"unsupported backend: {self.backend}"
                _log.warning("EnsembleClassifier disabled: %s", self._init_error)
                return
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                from tup_manager.injection_classifier import resolve_classifier_api_key

                token = resolve_classifier_api_key()
                kwargs: Dict[str, Any] = {}
                if token:
                    kwargs["token"] = token

                self._tokenizer = AutoTokenizer.from_pretrained(self.secondary_model, **kwargs)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.secondary_model, **kwargs
                )
                self._model.eval()
                self._malicious_index = self._resolve_malicious_index(self._model.config.id2label)
                self._available = True
                _log.info(
                    "EnsembleClassifier ready model=%s malicious_index=%s threshold=%.2f",
                    self.secondary_model,
                    self._malicious_index,
                    self.secondary_threshold,
                )
            except ImportError:
                self._init_error = "install transformers and torch: pip install transformers torch"
                _log.warning("EnsembleClassifier disabled: %s", self._init_error)
            except Exception as exc:
                self._init_error = str(exc)
                _log.warning("EnsembleClassifier init failed: %s", exc)

    @staticmethod
    def _resolve_malicious_index(id2label: Dict[Any, Any]) -> int:
        malicious_labels = frozenset({
            "INJECTION", "INJECT", "MALICIOUS", "JAILBREAK", "ATTACK", "LABEL_1", "1",
        })
        for idx, label in id2label.items():
            if str(label).upper() in malicious_labels:
                return int(idx)
        if len(id2label) >= 2:
            return max(int(k) for k in id2label.keys())
        return 1

    def score_secondary(self, text: str) -> float:
        """Score text with secondary model."""
        scores = self.score_batch_secondary([text])
        return scores[0] if scores else 0.0

    def score_batch_secondary(self, texts: List[str]) -> List[float]:
        """Batch score with secondary model."""
        if not self.enabled or not self._available:
            return [0.0] * len(texts)

        import torch

        out: List[float] = []
        idx = self._malicious_index or 1

        with self._lock:
            for text in texts:
                inputs = self._tokenizer(
                    [text],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)[:, idx]
                out.append(float(probs[0].item()))
        return out

    def is_attack_ensemble(self, primary_score: float, secondary_score: float) -> bool:
        """Ensemble rule: attack if primary >= t1 OR secondary >= t2."""
        if not self.enabled:
            return primary_score >= self.primary_threshold
        return primary_score >= self.primary_threshold or secondary_score >= self.secondary_threshold
