import base64
import re
import unicodedata
from typing import List, Optional

_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\ufeff\u2060\ufffe]")
_BASE64_CHUNK = re.compile(r"(?:[A-Za-z0-9+/]{40,}={0,2})")


def normalize_prompt(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _ZERO_WIDTH.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def try_base64_decode(text: str) -> Optional[str]:
    if not text:
        return None
    match = _BASE64_CHUNK.search(text.replace("\n", "").replace(" ", ""))
    if not match:
        return None
    chunk = match.group(0)
    pad = (-len(chunk)) % 4
    try:
        raw = base64.b64decode(chunk + ("=" * pad), validate=False)
        decoded = raw.decode("utf-8", errors="ignore").strip()
        if len(decoded) >= 8 and decoded.isprintable():
            return decoded
    except Exception:
        return None
    return None


def prompt_variants(text: str) -> List[str]:
    raw = text or ""
    seen: set[str] = set()
    out: List[str] = []

    def add(value: str) -> None:
        v = value.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)

    add(raw)
    normalized = normalize_prompt(raw)
    add(normalized)
    decoded = try_base64_decode(raw)
    if decoded:
        add(decoded)
        add(normalize_prompt(decoded))
    return out


def enrich_prompt_fields(event: dict) -> dict:
    prompt = event.get("prompt") or ""
    event["prompt_normalized"] = normalize_prompt(prompt)
    event["prompt_variants"] = prompt_variants(prompt)
    return event
