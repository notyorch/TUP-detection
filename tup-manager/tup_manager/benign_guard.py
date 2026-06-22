import os
from typing import Optional

# Benign dataset categories where mid-confidence classifier scores are often FP.
_BENIGN_CATEGORIES = frozenset({"documents", "hard_negatives"})


def _high_confidence_cutoff() -> float:
    return float(os.getenv("INJECTION_HIGH_CONFIDENCE", "0.995"))


def _benign_guard_enabled() -> bool:
    raw = os.getenv("BENIGN_GUARD_ENABLED", "true")
    return str(raw).lower() in ("1", "true", "yes", "on")


def should_suppress_classifier_hit(
    *,
    category: Optional[str],
    classifier_score: float,
    threshold: float,
) -> bool:
    """Suppress borderline classifier hits on clearly benign content types.

    Conservative mode: only suppress in 'documents' and 'hard_negatives', not 'chat'.
    Can be disabled via BENIGN_GUARD_ENABLED env var or DETECTION_MODE=strict|benchmark.
    """
    if not _benign_guard_enabled():
        return False
    if not category or category not in _BENIGN_CATEGORIES:
        return False
    if classifier_score < threshold:
        return False
    if classifier_score >= _high_confidence_cutoff():
        return False
    return True
