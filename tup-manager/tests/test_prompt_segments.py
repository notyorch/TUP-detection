"""Tests for prompt_segments.py"""
import pytest

from tup_manager.prompt_segments import (
    PromptSegments,
    enrich_with_segments,
    extract_context_prompt,
    extract_user_prompt,
)


def test_extract_user_prompt_simple():
    """Test simple text extraction (fallback)."""
    text = "Hello world"
    assert extract_user_prompt(text) == "Hello world"


def test_extract_user_prompt_user_marker():
    """Test <|user|> marker extraction."""
    text = "<|user|>Attack me<|assistant|>Sure"
    assert extract_user_prompt(text) == "Attack me"


def test_extract_user_prompt_bracket_marker():
    """Test [USER]...[/USER] marker extraction."""
    text = "[USER]Inject this[/USER]"
    assert extract_user_prompt(text) == "Inject this"


def test_extract_context_none():
    """Test no context found returns None."""
    text = "Hello"
    assert extract_context_prompt(text) is None


def test_extract_context_marker():
    """Test <|context|> marker extraction."""
    text = "<|context|>System prompt<|/context|><|user|>User input"
    assert extract_context_prompt(text) == "System prompt"


def test_prompt_segments_basic():
    """Test PromptSegments object."""
    text = "Single prompt"
    seg = PromptSegments(text)
    assert seg.full == text
    assert seg.user == text
    assert seg.context is None


def test_prompt_segments_with_user():
    """Test PromptSegments with user marker."""
    text = "<|user|>User input<|assistant|>Model output"
    seg = PromptSegments(text)
    assert seg.full == text
    assert seg.user == "User input"
    assert seg.context is None


def test_prompt_segments_all_segments():
    """Test all_segments() deduplication."""
    text = "<|context|>Ctx<|/context|><|user|>User<|assistant|>Resp"
    seg = PromptSegments(text)
    all_segs = seg.all_segments()
    assert len(all_segs) == 3
    assert "Ctx" in all_segs
    assert "User" in all_segs


def test_enrich_with_segments():
    """Test event enrichment."""
    event = {"prompt": "<|user|>Test<|assistant|>Ok"}
    enrich_with_segments(event)
    assert "prompt_full" in event
    assert "prompt_user" in event
    assert "prompt_segments" in event
    assert event["prompt_user"] == "Test"


def test_enrich_with_segments_empty():
    """Test enrichment with empty prompt."""
    event = {"prompt": ""}
    enrich_with_segments(event)
    assert event["prompt_segments"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
