"""Parse prompt into semantic segments: user input, context, full prompt."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def extract_user_prompt(text: str) -> str:
    """Extract user input from multi-segment prompts (Antijection format).

    Handles:
      - <|user|>...<|assistant|>
      - [USER]...[/USER]
      - Falls back to full text if no markers found.
    """
    if not text:
        return ""

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


def extract_context_prompt(text: str) -> Optional[str]:
    """Extract context/system segment if present.

    Handles:
      - <|context|>...<|/context|>
      - <|system|>...<|/system|>
    Returns None if no context found.
    """
    if not text:
        return None

    for tag_pair in [("<|context|>", "<|/context|>"), ("<|system|>", "<|/system|>")]:
        start, end = tag_pair
        if start in text and end in text:
            parts = text.split(start, 1)
            if len(parts) > 1:
                ctx = parts[1].split(end, 1)[0]
                stripped = ctx.strip()
                if stripped:
                    return stripped

    return None


class PromptSegments:
    """Parsed semantic segments of a prompt."""

    def __init__(self, text: str) -> None:
        self.full: str = text.strip() if text else ""
        self.user: str = extract_user_prompt(self.full)
        self.context: Optional[str] = extract_context_prompt(self.full)

    def all_segments(self) -> List[str]:
        """Return all non-empty segments for multi-variant scoring."""
        out = []
        if self.full:
            out.append(self.full)
        if self.user and self.user != self.full:
            out.append(self.user)
        if self.context and self.context not in out:
            out.append(self.context)
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for event enrichment."""
        return {
            "prompt_full": self.full,
            "prompt_user": self.user,
            "prompt_context": self.context,
        }


def enrich_with_segments(event: Dict[str, Any]) -> Dict[str, Any]:
    """Add semantic prompt segments to event."""
    prompt = event.get("prompt") or ""
    if not prompt:
        event["prompt_segments"] = {}
        return event

    segments = PromptSegments(prompt)
    event.update(segments.to_dict())
    event["prompt_segments"] = segments
    return event
