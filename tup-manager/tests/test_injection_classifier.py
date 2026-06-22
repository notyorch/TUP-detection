"""Tests for injection_classifier.py multi-variant scoring."""
import pytest

from tup_manager.injection_classifier import (
    InjectionClassifier,
    _malicious_score_from_hf_labels,
    resolve_classifier_api_key,
)


@pytest.fixture
def disabled_classifier():
    """Classifier instance that's disabled (no model load)."""
    return InjectionClassifier(enabled=False)


def test_classifier_disabled_score():
    """Test scoring returns 0 when disabled."""
    clf = InjectionClassifier(enabled=False)
    score = clf.score("any text")
    assert score == 0.0


def test_classifier_disabled_batch_score():
    """Test batch scoring returns zeros when disabled."""
    clf = InjectionClassifier(enabled=False)
    scores = clf.score_batch(["text1", "text2"])
    assert scores == [0.0, 0.0]


def test_classifier_disabled_variants():
    """Test variant scoring returns 0 when disabled."""
    clf = InjectionClassifier(enabled=False)
    score = clf.score_text_variants("base64text")
    assert score == 0.0


def test_classifier_score_text_variants_empty():
    """Test variant scoring with empty text."""
    clf = InjectionClassifier(enabled=False)
    score = clf.score_text_variants("")
    assert score == 0.0

def test_classifier_enabled_hf_missing_key():
    """HF backend without API key should fail gracefully."""
    import os
    saved = {k: os.environ.pop(k, None) for k in (
        "SENTINEL_API_KEY", "INJECTION_CLASSIFIER_API_KEY", "HF_INFERENCE_API_KEY",
        "HF_TOKEN", "HUGGINGFACE_TOKEN",
    )}
    try:
        clf = InjectionClassifier(enabled=True, backend="hf")
        assert not clf.available
        assert clf.init_error
        assert "SENTINEL_API_KEY" in clf.init_error or "HF_TOKEN" in clf.init_error
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_malicious_score_from_jailbreak_label():
    """Sentinel v2 returns label 'jailbreak' for attacks."""
    class _Item:
        def __init__(self, label, score):
            self.label = label
            self.score = score

    score = _malicious_score_from_hf_labels([
        _Item("jailbreak", 0.99),
        _Item("benign", 0.01),
    ])
    assert score == pytest.approx(0.99)


def test_resolve_classifier_api_key_prefers_sentinel():
    import os
    saved = {k: os.environ.get(k) for k in ("SENTINEL_API_KEY", "HF_TOKEN")}
    os.environ["SENTINEL_API_KEY"] = "hf_sentinel_test"
    os.environ["HF_TOKEN"] = "hf_other"
    try:
        assert resolve_classifier_api_key() == "hf_sentinel_test"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_classifier_enabled_init_fails_gracefully():
    """Test that classifier gracefully handles missing models."""
    clf = InjectionClassifier(enabled=True)
    if not clf.available:
        assert clf.score("test") == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
