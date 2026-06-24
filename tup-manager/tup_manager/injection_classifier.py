import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_log = logging.getLogger("tup.injection_classifier")

# Best open cross-benchmark classifier (Sentinel v2 paper: ~0.96 avg F1 vs ~0.75 DeBERTa).
DEFAULT_MODEL = "rogue-security/prompt-injection-jailbreak-sentinel-v2"
MALICIOUS_LABELS = frozenset({
    "INJECTION", "INJECT", "MALICIOUS", "JAILBREAK", "ATTACK", "LABEL_1", "1",
    "PROMPT_INJECTION", "UNSAFE", "HARMFUL",
})
BENIGN_LABELS = frozenset({
    "BENIGN", "SAFE", "NEGATIVE", "LABEL_0", "0", "NOT_INJECTION", "NOT_JAILBREAK",
})
HF_BACKENDS = frozenset({"hf", "hf_inference", "huggingface", "inference"})


def _is_hf_space_url(url: str) -> bool:
    return ".hf.space" in (url or "").lower()


def _prepare_text(text: str, max_chars: int = 4000) -> str:
    return " ".join((text or "").split())[:max_chars]


def resolve_classifier_api_key() -> Optional[str]:
    """API key for Sentinel / HF Inference (dedicated var first, then HF fallbacks)."""
    for env_name in (
        "SENTINEL_API_KEY",
        "INJECTION_CLASSIFIER_API_KEY",
        "HF_INFERENCE_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
    ):
        value = (os.getenv(env_name) or "").strip()
        if value:
            return value
    return None


def _hf_token() -> Optional[str]:
    return resolve_classifier_api_key()


def _malicious_score_from_hf_labels(items: List[Any]) -> float:
    """Extract P(malicious) from HF text-classification response."""
    if items is None:
        return 0.0
    if not isinstance(items, list):
        items = [items]
    best_malicious = 0.0
    best_other = 0.0
    for item in items:
        label = str(getattr(item, "label", "") or (item.get("label") if isinstance(item, dict) else "")).upper()
        score = float(getattr(item, "score", None) or (item.get("score") if isinstance(item, dict) else 0.0))
        if label in MALICIOUS_LABELS:
            best_malicious = max(best_malicious, score)
        elif label not in BENIGN_LABELS:
            best_other = max(best_other, score)
        else:
            best_other = max(best_other, 1.0 - score)
    if best_malicious > 0:
        return best_malicious
    return best_other


class InjectionClassifier:
    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        model_id: Optional[str] = None,
        threshold: Optional[float] = None,
        backend: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        raw_enabled = enabled if enabled is not None else os.getenv("INJECTION_CLASSIFIER_ENABLED", "true")
        self.enabled = str(raw_enabled).lower() in ("1", "true", "yes")
        self.model_id = model_id or os.getenv("INJECTION_MODEL", DEFAULT_MODEL)
        self.threshold = threshold if threshold is not None else float(os.getenv("INJECTION_THRESHOLD", "0.5"))
        self.backend = (backend or os.getenv("INJECTION_CLASSIFIER_BACKEND", "hf")).lower()
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        # Fail policy when inference errors out after retries:
        #   fail-open (default): return 0.0 (benign) — avoids alert storms on flaky endpoint
        #   fail-closed: return 1.0 (malicious) — security-safe, surfaces a hit on failure
        self.fail_open = str(os.getenv("INJECTION_FAIL_OPEN", "true")).lower() in ("1", "true", "yes", "on")
        self.scoring_failures = 0
        self.api_calls = 0
        self._counter_lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._client = None
        self._malicious_index: Optional[int] = None
        self._available = False
        self._init_error: Optional[str] = None
        self._lock = threading.Lock()
        self._hf_model_ref: Optional[str] = None
        self._use_hf_space_api = False
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
            if self.backend in HF_BACKENDS:
                self._ensure_hf_client()
            elif self.backend == "local":
                self._ensure_local_model()
            else:
                self._init_error = f"unsupported backend: {self.backend} (use hf or local)"
                _log.warning("InjectionClassifier disabled: %s", self._init_error)

    def _ensure_hf_client(self) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            self._init_error = "install huggingface_hub: pip install huggingface_hub"
            _log.warning("InjectionClassifier disabled: %s", self._init_error)
            return

        token = resolve_classifier_api_key()
        if not token:
            self._init_error = (
                "SENTINEL_API_KEY (or HF_TOKEN) required for cloud inference backend"
            )
            _log.warning("InjectionClassifier disabled: %s", self._init_error)
            return

        endpoint = (os.getenv("HF_INFERENCE_ENDPOINT") or "").strip()

        # FAIL FAST: Exigir endpoint si backend explícitamente es 'hf' (no serverless)
        if not endpoint and self.backend == "hf":
            self._init_error = (
                "HF_INFERENCE_ENDPOINT required for backend='hf' (Sentinel v2 scale-to-zero endpoint). "
                "Use backend='local' for local model or 'huggingface' for serverless fallback."
            )
            _log.warning("InjectionClassifier disabled: %s", self._init_error)
            return

        provider = (os.getenv("HF_INFERENCE_PROVIDER") or "auto").strip() or "auto"
        timeout_raw = os.getenv("HF_INFERENCE_TIMEOUT", "120")
        try:
            timeout = float(timeout_raw)
        except ValueError:
            timeout = 120.0

        # Dedicated Endpoint URL takes precedence; otherwise route by model id.
        model_ref = endpoint if endpoint else self.model_id
        self._hf_model_ref = model_ref
        self._use_hf_space_api = bool(endpoint and _is_hf_space_url(endpoint))

        if self._use_hf_space_api:
            self._available = True
            _log.info(
                "InjectionClassifier ready backend=hf-space model=%s endpoint=%s threshold=%.2f",
                self.model_id,
                model_ref,
                self.threshold,
            )
            self._warmup_hf_space()
            return

        try:
            kwargs: Dict[str, Any] = {"token": token, "timeout": timeout}
            if endpoint:
                kwargs["model"] = endpoint
            else:
                kwargs["model"] = self.model_id
                if provider and provider != "auto":
                    kwargs["provider"] = provider

            self._client = InferenceClient(**kwargs)
            self._available = True
            _log.info(
                "InjectionClassifier ready backend=hf model=%s endpoint=%s threshold=%.2f",
                self.model_id,
                "dedicated" if endpoint else "fallback",
                self.threshold,
            )

            # WARMUP: 1 request para escala-a-cero si es endpoint dedicado
            if endpoint:
                self._warmup_hf()
        except Exception as exc:
            self._init_error = str(exc)
            _log.warning("InjectionClassifier HF init failed: %s", exc)

    def _ensure_local_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            token = _hf_token()
            kwargs: Dict[str, Any] = {"trust_remote_code": True}
            if token:
                kwargs["token"] = token

            dtype_name = (os.getenv("INJECTION_TORCH_DTYPE") or "float32").lower()
            if dtype_name in ("float16", "fp16", "half"):
                kwargs["torch_dtype"] = torch.float16

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id, **kwargs)
            self._model.eval()
            self._malicious_index = self._resolve_malicious_index(self._model.config.id2label)
            self._available = True
            _log.info(
                "InjectionClassifier ready backend=local model=%s malicious_index=%s threshold=%.2f",
                self.model_id,
                self._malicious_index,
                self.threshold,
            )
        except ImportError:
            self._init_error = "install transformers and torch: pip install transformers torch"
            _log.warning("InjectionClassifier disabled: %s", self._init_error)
        except Exception as exc:
            self._init_error = str(exc)
            _log.warning("InjectionClassifier local init failed: %s", exc)

    @staticmethod
    def _resolve_malicious_index(id2label: Dict[Any, Any]) -> int:
        for idx, label in id2label.items():
            if str(label).upper() in MALICIOUS_LABELS:
                return int(idx)
        if len(id2label) >= 2:
            return max(int(k) for k in id2label.keys())
        return 1

    def _warmup_hf_space(self) -> None:
        try:
            self._score_hf_space("Warmup test")
            _log.debug("HF Space warmup request sent")
        except Exception as exc:
            _log.debug("HF Space warmup failed (ok if cold start): %s", exc)

    def _warmup_hf(self) -> None:
        """Warmup request para escala-a-cero. Fire-and-forget, sin re-intentos."""
        try:
            self._client.text_classification("Warmup test", model=self._hf_model_ref)
            _log.debug("HF warmup request sent")
        except Exception as exc:
            _log.debug("HF warmup failed (ok if scale-to-zero): %s", exc)

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Check if error code is 503/504/524 (scale-to-zero, overload)."""
        exc_str = str(exc)
        return any(code in exc_str for code in ("503", "504", "524"))

    def _score_hf_space(self, text: str) -> float:
        import requests

        prepared = _prepare_text(text)
        if not prepared or not self._hf_model_ref:
            return 0.0
        url = self._hf_model_ref.rstrip("/") + "/"
        token = resolve_classifier_api_key()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        retries = max(1, int(os.getenv("HF_INFERENCE_RETRIES", "5")))
        delay = float(os.getenv("HF_INFERENCE_RETRY_DELAY", "5.0"))
        timeout = float(os.getenv("HF_INFERENCE_TIMEOUT", "120"))
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                resp = requests.post(
                    url,
                    json={"inputs": prepared},
                    headers=headers,
                    timeout=timeout,
                )
                if resp.status_code in (502, 503, 504, 524):
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                return _malicious_score_from_hf_labels(resp.json())
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < retries and self._is_retryable_error(exc):
                    backoff = delay * min(3 ** attempt, 10)
                    _log.warning(
                        "HF Space error (retryable, attempt %d/%d): %s, waiting %.1fs",
                        attempt + 1,
                        retries,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                elif attempt + 1 < retries:
                    _log.warning("HF Space non-retryable error: %s", exc)
                    break
        return self._on_scoring_failure(retries, last_exc)

    def _score_hf(self, text: str) -> float:
        with self._counter_lock:
            self.api_calls += 1
        if self._use_hf_space_api:
            return self._score_hf_space(text)
        if not self._client:
            return 0.0
        prepared = _prepare_text(text)
        if not prepared:
            return 0.0
        model_kw: Dict[str, Any] = {}
        if self._hf_model_ref and self._hf_model_ref != self.model_id:
            model_kw["model"] = self._hf_model_ref
        elif not os.getenv("HF_INFERENCE_ENDPOINT"):
            model_kw["model"] = self.model_id

        retries = max(1, int(os.getenv("HF_INFERENCE_RETRIES", "5")))
        delay = float(os.getenv("HF_INFERENCE_RETRY_DELAY", "5.0"))
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                result = self._client.text_classification(prepared, **model_kw)
                return _malicious_score_from_hf_labels(result)
            except Exception as exc:
                last_exc = exc
                is_retryable = self._is_retryable_error(exc)
                if attempt + 1 < retries and is_retryable:
                    # Backoff exponencial: 5s, 15s, 30s, ...
                    backoff = delay * min(3 ** attempt, 10)
                    _log.warning(
                        "HF inference error (retryable, attempt %d/%d): %s, waiting %.1fs",
                        attempt + 1,
                        retries,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                elif attempt + 1 < retries and not is_retryable:
                    # Non-retryable: fail fast
                    _log.warning("HF inference non-retryable error: %s", exc)
                    break
        return self._on_scoring_failure(retries, last_exc)

    def _on_scoring_failure(self, retries: int, last_exc: Optional[Exception]) -> float:
        """Apply fail policy when scoring exhausts retries. ERROR-level, not silent."""
        with self._counter_lock:
            self.scoring_failures += 1
        if self.fail_open:
            _log.error(
                "HF inference failed after %s tries (fail-OPEN -> scoring as benign 0.0): %s",
                retries, last_exc,
            )
            return 0.0
        _log.error(
            "HF inference failed after %s tries (fail-CLOSED -> scoring as malicious 1.0): %s",
            retries, last_exc,
        )
        return 1.0

    def score(self, text: str) -> float:
        scores = self.score_batch([text])
        return scores[0] if scores else 0.0

    def score_text_variants(self, text: str) -> float:
        from tup_manager.text_normalize import prompt_variants
        variants = prompt_variants(text)
        if not variants:
            return 0.0
        scores = self.score_batch(variants)
        return max(scores) if scores else 0.0

    def score_variants_batch(self, prompts: List[str], *, progress: bool = False) -> List[float]:
        """Multi-variant max score per prompt, with global variant dedup.

        Many prompts share identical variants (raw == normalized when already clean),
        so we score the *unique* set once and map back — big win over per-row scoring.
        """
        from tup_manager.text_normalize import prompt_variants

        per_prompt = [prompt_variants(p or "") for p in prompts]
        unique: Dict[str, None] = {}
        for variants in per_prompt:
            for v in variants:
                unique.setdefault(v, None)
        unique_texts = list(unique.keys())
        scores = self._score_unique(unique_texts, progress=progress)
        lookup = dict(zip(unique_texts, scores))
        out: List[float] = []
        for variants in per_prompt:
            out.append(max((lookup.get(v, 0.0) for v in variants), default=0.0))
        return out

    def _score_unique(self, texts: List[str], *, progress: bool = False) -> List[float]:
        if not self.enabled:
            return [0.0] * len(texts)
        self._ensure_loaded()
        if not self._available or not texts:
            return [0.0] * len(texts)
        if self.backend in HF_BACKENDS:
            return self._score_hf_parallel(texts, progress=progress)
        return self.score_batch(texts)

    def _score_hf_parallel(self, texts: List[str], *, progress: bool = False) -> List[float]:
        """Score texts over the HF backend with a thread pool (I/O-bound HTTP)."""
        workers = max(1, int(os.getenv("HF_INFERENCE_WORKERS", "8")))
        results: List[float] = [0.0] * len(texts)

        def _iter(it):
            if not progress:
                return it
            try:
                import tqdm
                return tqdm.tqdm(it, total=len(texts), desc="HF scoring", unit="req")
            except ImportError:
                return it

        if workers == 1:
            for i, t in _iter(enumerate(texts)):
                results[i] = self._score_hf(t)
            return results

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._score_hf, t): i for i, t in enumerate(texts)}
            for fut in _iter(as_completed(futures)):
                results[futures[fut]] = fut.result()
        return results

    def score_batch(self, texts: List[str]) -> List[float]:
        if not self.enabled:
            return [0.0] * len(texts)
        self._ensure_loaded()
        if not self._available or not texts:
            return [0.0] * len(texts)

        if self.backend in HF_BACKENDS:
            return self._score_hf_parallel(texts)

        import torch

        prepared = [_prepare_text(t) for t in texts]
        out: List[float] = []
        idx = self._malicious_index or 1

        with self._lock:
            for start in range(0, len(prepared), self.batch_size):
                chunk = prepared[start : start + self.batch_size]
                inputs = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)[:, idx]
                out.extend(float(x) for x in probs.tolist())
        return out

    def is_malicious(self, text: str) -> bool:
        return self.score(text) >= self.threshold

    def analyze(
        self,
        event: Dict[str, Any],
        *,
        score: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Build a classifier alert for an event.

        If ``score`` is provided, it is reused as-is (no re-inference) — the caller
        already decided this is a hit (avoids double scoring + threshold mismatch).
        ``threshold`` overrides ``self.threshold`` for the gating check.
        """
        text = event.get("prompt") or event.get("prompt_normalized") or ""
        if not text.strip():
            return []
        resolved = score if score is not None else self.score(text)
        gate = threshold if threshold is not None else self.threshold
        if resolved < gate:
            return []
        alert = self._to_alert(resolved, event)
        return [alert] if alert else []

    def _to_alert(self, score: float, event: Dict[str, Any]) -> Dict[str, Any]:
        severity = min(15, max(7, round(score * 15)))
        snippet = _prepare_text(event.get("prompt") or "", 200)
        backend_label = "hf-inference" if self.backend in HF_BACKENDS else "classifier"
        return {
            "alert_id": str(uuid.uuid4()),
            "event_id": str(event.get("event_id", "")),
            "rule_id": "injection-clf-001",
            "title": f"Prompt Injection Classifier (score={score:.2f})",
            "severity": severity,
            "owasp_mapping": "LLM01",
            "mitre_atlas": "AML.T0054",
            "matched_field": "prompt",
            "matched_value": snippet,
            "action": "log",
            "source": event.get("source"),
            "model_id": event.get("model_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "behavior": "prompt_injection",
            "detection_source": backend_label,
            "classifier_score": round(score, 4),
            "classifier_model": self.model_id,
            "iocs": [],
        }
