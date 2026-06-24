import os
from typing import Any, Dict, List, Optional, Set

from tup_manager.benign_guard import should_suppress_classifier_hit
from tup_manager.injection_classifier import InjectionClassifier
from tup_manager.nvidia_judge_engine import NvidiaJudgeEngine
from tup_manager.prompt_segments import enrich_with_segments
from tup_manager.rules_engine import TUPRulesEngine

INJECTION_ATTACK_RULES: Set[str] = {
    "tup-rule-0001", "tup-rule-0002", "tup-rule-0004",
    "tup-rule-0007", "tup-rule-0009", "tup-rule-0011",
    "injection-clf-001", "llm-judge-001",
}


class DetectionEngine:
    """L1 regex -> Sentinel/HF classifier -> optional NVIDIA judge L3 fallback."""

    def __init__(
        self,
        rules_dir: str,
        *,
        classifier: Optional[InjectionClassifier] = None,
        judge: Optional[NvidiaJudgeEngine] = None,
        injection_threshold: Optional[float] = None,
        use_judge: Optional[bool] = None,
        detection_mode: Optional[str] = None,
        gray_zone_low: Optional[float] = None,
        gray_zone_high: Optional[float] = None,
        use_gray_zone_judge: Optional[bool] = None,
    ) -> None:
        self.rules_engine = TUPRulesEngine(rules_dir)
        threshold = injection_threshold if injection_threshold is not None else float(
            os.getenv("INJECTION_THRESHOLD", "0.5")
        )
        self.classifier = classifier or InjectionClassifier(threshold=threshold)
        self.judge = judge if judge is not None else NvidiaJudgeEngine()
        raw_judge = use_judge if use_judge is not None else os.getenv("DETECTION_JUDGE_ENABLED", "auto")
        mode = str(raw_judge).lower()
        if mode in ("0", "false", "no", "off"):
            self.use_judge = False
        elif mode in ("1", "true", "yes", "on"):
            self.use_judge = True
        else:
            self.use_judge = self.judge.available
        self.injection_threshold = threshold

        # Detection mode and adaptive thresholds
        self.detection_mode = (
            detection_mode or os.getenv("DETECTION_MODE", "production")
        ).lower()
        if self.detection_mode not in ("production", "strict", "benchmark"):
            self.detection_mode = "production"

        # In strict/benchmark modes, use lower threshold if provided
        if self.detection_mode in ("strict", "benchmark"):
            threshold_env = os.getenv("INJECTION_THRESHOLD_STRICT", "0.15")
            self.injection_threshold = float(threshold_env)

        # Gray zone for fallback to judge
        raw_low = gray_zone_low or os.getenv("GRAY_ZONE_LOW", "0.15")
        raw_high = gray_zone_high or os.getenv("GRAY_ZONE_HIGH", "0.85")
        self.gray_zone_low = float(raw_low)
        self.gray_zone_high = float(raw_high)
        raw_gz = use_gray_zone_judge if use_gray_zone_judge is not None else os.getenv(
            "DETECTION_JUDGE_GRAY_ZONE_ONLY", "0"
        )
        self.use_gray_zone_judge = str(raw_gz).lower() in ("1", "true", "yes", "on")

    def health_check(self) -> bool:
        """Quick health check: verify classifier and judge are available."""
        if not self.classifier.available:
            return False
        if self.use_judge and not self.judge.available:
            return False
        return True

    def warmup(self) -> bool:
        """Warmup for scale-to-zero (HF Endpoint). Test 1 simple prompt."""
        if not self.classifier.available:
            return False
        try:
            _score = self.classifier.score("test prompt")
            return _score is not None
        except Exception:
            return False

    def _layer1_alerts(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts = self.rules_engine.evaluate(event, session_events=None)
        return [a for a in alerts if a.get("rule_id") in INJECTION_ATTACK_RULES]

    def _classifier_score(self, event: Dict[str, Any], score: Optional[float] = None) -> float:
        if score is not None:
            return score
        if not self.classifier.enabled:
            return 0.0

        # Enrich with segments if not already done
        if "prompt_segments" not in event:
            enrich_with_segments(event)

        prompt = event.get("prompt") or event.get("prompt_normalized") or ""
        if not prompt:
            return 0.0

        # For strict/benchmark modes, score variants; for production, score single text
        if self.detection_mode in ("strict", "benchmark"):
            return self.classifier.score_text_variants(prompt)
        return self.classifier.score(prompt)

    def _is_suppressed(self, event: Dict[str, Any], score: float) -> bool:
        """benign_guard: suppress borderline hits on clearly benign content types."""
        category = event.get("category") or event.get("pint_category")
        return should_suppress_classifier_hit(
            category=category,
            classifier_score=score,
            threshold=self.injection_threshold,
        )

    def _classifier_hit(self, event: Dict[str, Any], score: Optional[float] = None) -> bool:
        resolved = self._classifier_score(event, score)
        if resolved < self.injection_threshold:
            return False
        return not self._is_suppressed(event, resolved)

    def detect(
        self,
        event: Dict[str, Any],
        *,
        classifier_score: Optional[float] = None,
        skip_judge: bool = False,
    ) -> List[Dict[str, Any]]:
        l1 = self._layer1_alerts(event)
        if l1:
            return l1

        # Score once, reuse for the decision, benign_guard, and the alert.
        score = self._classifier_score(event, classifier_score)
        if score >= self.injection_threshold and not self._is_suppressed(event, score):
            return self.classifier.analyze(
                event, score=score, threshold=self.injection_threshold
            )

        # Gray zone: borderline scores get judge fallback
        if (
            self.use_judge
            and not skip_judge
            and self.judge.available
            and self.gray_zone_low <= score <= self.gray_zone_high
        ):
            judge_alerts = self.judge.analyze(event)
            if judge_alerts:
                return judge_alerts

        # Non-gray-zone; judge only if gray_zone mode disabled
        if (
            self.use_judge
            and not skip_judge
            and self.judge.available
            and not self.use_gray_zone_judge
            and score < self.injection_threshold
        ):
            judge_alerts = self.judge.analyze(event)
            if judge_alerts:
                return judge_alerts

        return []

    def is_attack(
        self,
        event: Dict[str, Any],
        *,
        classifier_score: Optional[float] = None,
        skip_judge: bool = False,
    ) -> bool:
        return bool(self.detect(event, classifier_score=classifier_score, skip_judge=skip_judge))
