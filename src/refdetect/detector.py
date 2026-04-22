#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import List, Optional, Tuple

import fitz
import regex as re

from .ieee import IEEEParser
from .models import Reference
from .normalize import _flatten_reference_whitespace, _nfc, _normalize_text

# Match IEEE reference-entry markers ONLY at line start:
#   [1] Author...
ENTRY_NUM_START = re.compile(r"(?m)^\s*\[(?P<idx>\d{1,4})\]\s+")


def _document_text(doc) -> str:
    joined = "\n\n".join(page.get_text("text", sort=True) for page in doc)
    return _normalize_text(joined)


def _find_bibliography_span_by_numbered_suffix(
    full_text: str,
    min_entries: int = 6,
    max_gap: int = 3,
    max_noise_breaks: int = 2,
) -> Optional[Tuple[int, int]]:
    """
    We look for a *suffix* of the document that contains many line-start
    reference markers like:
        [23] ...
        [22] ...
        ...
        [1] ...

    We scan matches from the end and build the longest "mostly decreasing" run.

    Parameters:
      - min_entries: require at least this many entries in the suffix to accept it
      - max_gap: allow missing numbers e.g. [10] then [8] up to this gap
      - max_noise_breaks: allow a couple of "bad steps" before breaking the run
    """
    matches = list(ENTRY_NUM_START.finditer(full_text))
    if len(matches) < min_entries:
        return None

    run: List[re.Match] = []
    noise_breaks = 0

    last = matches[-1]
    try:
        prev_idx = int(last.group("idx"))
    except Exception:
        return None

    run.append(last)

    for m in reversed(matches[:-1]):
        try:
            idx = int(m.group("idx"))
        except Exception:
            continue

        if idx <= prev_idx and (prev_idx - idx) <= max_gap:
            run.append(m)
            prev_idx = idx
            continue

        noise_breaks += 1
        if noise_breaks > max_noise_breaks:
            break

    if len(run) < min_entries:
        return None

    run = list(reversed(run))
    start_pos = run[0].start()
    end_pos = len(full_text)

    min_idx = None
    min_pos = start_pos
    for m in run:
        try:
            idx = int(m.group("idx"))
        except Exception:
            continue
        if min_idx is None or idx < min_idx:
            min_idx = idx
            min_pos = m.start()
    start_pos = min_pos

    return (start_pos, end_pos)


def _split_entries(bib_text: str) -> List[str]:
    hits = list(ENTRY_NUM_START.finditer(bib_text))
    if not hits:
        return []

    out: List[str] = []
    for i, m in enumerate(hits):
        entry_start = m.start()
        entry_end = hits[i + 1].start() if i + 1 < len(hits) else len(bib_text)
        chunk = bib_text[entry_start:entry_end].strip()
        if chunk:
            out.append(chunk)
    return out


def extract_references(pdf_path: str) -> List[Reference]:
    with fitz.open(pdf_path) as doc:
        full_text = _document_text(doc)

    span = _find_bibliography_span_by_numbered_suffix(full_text)
    if not span:
        print("No bibliography detected.")
        return []

    b0, b1 = span
    bib = full_text[b0:b1].strip()

    entries = _split_entries(bib)
    if not entries:
        print("Bibliography suffix found, but no parsable [n] entries.")
        return []

    ieee = IEEEParser()
    parsed: List[Reference] = []

    for entry in entries:
        normalized_entry = _flatten_reference_whitespace(_nfc(entry.strip()))
        if not normalized_entry:
            continue
        if not ENTRY_NUM_START.match(normalized_entry):
            continue
        parsed.append(ieee.parse(normalized_entry))

    seen = set()
    out: List[Reference] = []
    for ref in parsed:
        key = re.sub(r"\s+", " ", ref.raw).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)

    return out


def write_json(refs: List[Reference], out_path: str) -> None:
    data = [r.to_dict() for r in refs]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

