#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, List

import regex as re


IEEE_CITATION_PATTERN = re.compile(
    r"""
    \[
      \s*
      \d+(?:\s*[-\u2013]\s*\d+)?
      (?:\s*,\s*\d+(?:\s*[-\u2013]\s*\d+)?)*
      \s*
    \]
    """,
    re.VERBOSE,
)


def expand_citation_numbers(citation_text: str) -> List[str]:
    inside_brackets = citation_text.strip()[1:-1]
    citation_numbers: List[str] = []

    for part in re.split(r"\s*,\s*", inside_brackets):
        if not part:
            continue

        range_parts = re.split(r"\s*[-\u2013]\s*", part)
        if len(range_parts) == 2 and range_parts[0].isdigit() and range_parts[1].isdigit():
            start_number = int(range_parts[0])
            end_number = int(range_parts[1])
            step = 1 if start_number <= end_number else -1
            for number in range(start_number, end_number + step, step):
                citation_numbers.append(f"[{number}]")
            continue

        if part.isdigit():
            citation_numbers.append(f"[{int(part)}]")

    unique_numbers: List[str] = []
    seen_numbers = set()
    for citation_number in citation_numbers:
        if citation_number in seen_numbers:
            continue
        seen_numbers.add(citation_number)
        unique_numbers.append(citation_number)

    return unique_numbers


class IEEEParser:
    name = "ieee"

    def find_citations_in_text(self, text: str) -> List[Dict[str, object]]:
        citations: List[Dict[str, object]] = []

        for match in IEEE_CITATION_PATTERN.finditer(text):
            raw_text = match.group(0)
            normalized_text = re.sub(r"\s+", " ", raw_text).strip()
            citation_numbers = expand_citation_numbers(normalized_text)
            citations.append(
                {
                    "raw_text": raw_text,
                    "citation_numbers": citation_numbers,
                }
            )

        citations.sort(key=lambda citation: text.find(citation["raw_text"]))
        return citations
