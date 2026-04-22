from __future__ import annotations


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def slugify(text: str, max_len: int = 120) -> str:
    """Convert an arbitrary string to a filesystem- and Milvus-safe identifier."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)[:max_len]
