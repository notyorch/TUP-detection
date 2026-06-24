"""Tests for dedup + parallel multi-variant scoring (perf path)."""
import threading
import time

import pytest

from tup_manager.injection_classifier import InjectionClassifier


class CountingHFClassifier(InjectionClassifier):
    """Simulates the HF backend: counts unique HTTP calls, fakes latency."""

    def __init__(self, latency: float = 0.0):
        super().__init__(enabled=False, backend="hf")
        self.enabled = True
        self._available = True
        self.backend = "hf"
        self._latency = latency
        self._scored_texts = []
        self._call_lock = threading.Lock()

    # Override the single-text scorer used by the parallel pool.
    def _score_hf(self, text: str) -> float:
        with self._counter_lock:
            self.api_calls += 1
        if self._latency:
            time.sleep(self._latency)
        with self._call_lock:
            self._scored_texts.append(text)
        # Deterministic fake score: 0.9 if "attack" in text else 0.1
        return 0.9 if "attack" in text.lower() else 0.1


def test_variant_dedup_reduces_api_calls():
    """Shared variants across prompts are scored once (global dedup)."""
    clf = CountingHFClassifier()
    prompts = ["hello world", "hello world", "an attack prompt"]
    scores = clf.score_variants_batch(prompts)

    assert scores[0] == pytest.approx(0.1)
    assert scores[1] == pytest.approx(0.1)
    assert scores[2] == pytest.approx(0.9)
    # "hello world" (raw == normalized) appears in 2 rows but scored once.
    # Unique variants here: {"hello world", "an attack prompt"} -> <= 2 calls.
    assert clf.api_calls <= 2


def test_parallel_faster_than_sequential(monkeypatch):
    """Thread pool overlaps I/O latency (8 calls × 0.1s should be well under 0.8s)."""
    monkeypatch.setenv("HF_INFERENCE_WORKERS", "8")
    clf = CountingHFClassifier(latency=0.1)
    texts = [f"unique attack {i}" for i in range(8)]
    t0 = time.time()
    scores = clf._score_unique(texts)
    elapsed = time.time() - t0

    assert len(scores) == 8
    assert all(s == pytest.approx(0.9) for s in scores)
    assert elapsed < 0.5  # parallel; sequential would be ~0.8s


def test_results_order_preserved():
    """Parallel scoring maps results back to original order."""
    clf = CountingHFClassifier()
    texts = ["attack one", "benign two", "attack three", "benign four"]
    scores = clf._score_unique(texts)
    assert scores == [
        pytest.approx(0.9),
        pytest.approx(0.1),
        pytest.approx(0.9),
        pytest.approx(0.1),
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
