"""DOI extraction and normalization utilities."""
from __future__ import annotations

import re
from typing import List, Optional

DOI_REGEX = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.I)

_DOI_URL_PREFIX_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.I)
_DOI_INLINE_PREFIX_RE = re.compile(r"^doi:\s*", re.I)
_TRAILING_PUNCT_RE = re.compile(r"[)\.,;}\]>\"']+$")


def normalize_doi(doi: str) -> str:
    doi = (doi or "").strip()
    doi = _DOI_URL_PREFIX_RE.sub("", doi)
    return doi.strip().strip(".").lower()


def _clean_doi_candidate(raw: str) -> str:
    candidate = (raw or "").strip()
    if not candidate:
        return ""
    candidate = _TRAILING_PUNCT_RE.sub("", candidate)
    candidate = _DOI_INLINE_PREFIX_RE.sub("", candidate)
    candidate = re.split(r"\s", candidate, maxsplit=1)[0]
    return normalize_doi(candidate)


def extract_doi_candidates(text: str) -> List[str]:
    if not text:
        return []

    candidates: List[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        cleaned = _clean_doi_candidate(raw)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            candidates.append(cleaned)

    for match in re.finditer(r"doi\.org/(10\.\d{4,9}/[^\s\"<>]+)", text, flags=re.I):
        _add(match.group(1))

    for match in DOI_REGEX.finditer(text):
        _add(match.group(0))

    return candidates


def pick_best_doi(
    reference_doi: str,
    reference_raw: str,
) -> Optional[str]:
    for field in (reference_doi or "", reference_raw or ""):
        candidates = extract_doi_candidates(field)
        if candidates:
            return candidates[0]
    return None
