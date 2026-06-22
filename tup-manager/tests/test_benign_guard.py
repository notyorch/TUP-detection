"""Tests for benign_guard.py modes."""
import os
import pytest

from tup_manager.benign_guard import should_suppress_classifier_hit


def test_suppress_documents_borderline(monkeypatch):
    """Test suppression in documents category (borderline score)."""
    monkeypatch.setenv("BENIGN_GUARD_ENABLED", "true")
    monkeypatch.setenv("INJECTION_HIGH_CONFIDENCE", "0.995")

    # Borderline hit in documents should be suppressed
    result = should_suppress_classifier_hit(
        category="documents",
        classifier_score=0.80,
        threshold=0.71,
    )
    assert result is True


def test_no_suppress_documents_high_confidence(monkeypatch):
    """Test no suppression for high-confidence hits in documents."""
    monkeypatch.setenv("BENIGN_GUARD_ENABLED", "true")
    monkeypatch.setenv("INJECTION_HIGH_CONFIDENCE", "0.995")

    # High-confidence hit should NOT be suppressed
    result = should_suppress_classifier_hit(
        category="documents",
        classifier_score=0.996,
        threshold=0.71,
    )
    assert result is False


def test_no_suppress_chat_category(monkeypatch):
    """Test that chat category is NOT suppressed (conservative mode)."""
    monkeypatch.setenv("BENIGN_GUARD_ENABLED", "true")
    monkeypatch.setenv("INJECTION_HIGH_CONFIDENCE", "0.995")

    # Chat borderline should NOT be suppressed (conservative change)
    result = should_suppress_classifier_hit(
        category="chat",
        classifier_score=0.80,
        threshold=0.71,
    )
    assert result is False


def test_no_suppress_when_disabled(monkeypatch):
    """Test that BENIGN_GUARD_ENABLED=false disables suppression."""
    monkeypatch.setenv("BENIGN_GUARD_ENABLED", "false")
    monkeypatch.setenv("INJECTION_HIGH_CONFIDENCE", "0.995")

    # Even borderline in documents should not be suppressed if disabled
    result = should_suppress_classifier_hit(
        category="documents",
        classifier_score=0.80,
        threshold=0.71,
    )
    assert result is False


def test_no_suppress_hard_negatives_high_confidence(monkeypatch):
    """Test no suppression for high-confidence in hard_negatives."""
    monkeypatch.setenv("BENIGN_GUARD_ENABLED", "true")
    monkeypatch.setenv("INJECTION_HIGH_CONFIDENCE", "0.995")

    # high_negatives high-confidence should NOT be suppressed
    result = should_suppress_classifier_hit(
        category="hard_negatives",
        classifier_score=0.999,
        threshold=0.71,
    )
    assert result is False


def test_suppress_hard_negatives_borderline(monkeypatch):
    """Test suppression in hard_negatives (borderline score)."""
    monkeypatch.setenv("BENIGN_GUARD_ENABLED", "true")
    monkeypatch.setenv("INJECTION_HIGH_CONFIDENCE", "0.995")

    # Borderline in hard_negatives should be suppressed
    result = should_suppress_classifier_hit(
        category="hard_negatives",
        classifier_score=0.75,
        threshold=0.71,
    )
    assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
