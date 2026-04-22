#!/usr/bin/env python3
from __future__ import annotations

import regex as re

from .models import Reference
from .normalize import _nfc

_NUM_PREFIX = re.compile(r"^\s*\[(?P<i>\d{1,3})\]\s+")
_DOI_RX = re.compile(
    r"(?:(?:https?://)?(?:dx\.)?doi\.org/)?"
    r"(?P<doi>10\.\d{4,9}\s*/\s*[-._;()/:A-Z0-9]+(?:\s*[-._;()/:A-Z0-9]+)*)",
    re.I,
)


def _normalize_doi(value: str) -> str:
    doi = re.sub(r"\s+", "", value)
    return doi.rstrip(".,;:)]}")


class IEEEParser:
    def parse(self, raw: str) -> Reference:
        src = _nfc(raw).strip()

        idx = None
        body = src
        m = _NUM_PREFIX.match(src)
        if m:
            try:
                idx = int(m.group("i"))
            except Exception:
                idx = None
            body = src[m.end():].strip()

        ref = Reference(raw=src, index=idx)

        dm = _DOI_RX.search(body)
        if dm:
            ref.doi = _normalize_doi(dm.group("doi"))

        return ref
