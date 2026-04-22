from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from commons.text import normalize_whitespace, slugify  # noqa: F401  re-exported


@dataclass
class Segment:
    """One indexable chunk of text extracted from a reference document.
    """
    zone: str           # "abstract" | "body"
    level: str          # "sentence" | "paragraph" | "section"
    text: str
    section_path: str   # e.g. "Introduction/Related Work"
    para_i: int         # paragraph index within its section
    sent_i: int         # sentence index within its paragraph

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        return cls(**data)
