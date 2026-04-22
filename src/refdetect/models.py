from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Reference:
    raw: str
    index: Optional[int] = None
    doi: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "raw": self.raw,
            "doi": self.doi,
        }
        if self.index is not None:
            d["index"] = self.index
        return d
