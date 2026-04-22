#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import regex as re

from .ieee import IEEEParser
from .spans import (
    compute_sentence_ranges,
    find_quote_before_citation,
    find_quote_ranges,
    find_sentence_at_position,
)
from .text_utils import build_document_text, clean_passage_text, normalize_text

# matches "1" or "[1]" or " [  1 ] "
CITATION_NUMBER_PATTERN = re.compile(r"^\s*\[?\s*(\d{1,4})\s*\]?\s*$")


def parse_citation_number(citation_text: str) -> str | None:
    match = CITATION_NUMBER_PATTERN.match(str(citation_text))
    return match.group(1) if match else None


def remove_last_passage_per_citation(passages_by_citation: Dict[str, Dict[str, None]]) -> None:
    """Last sentence is always the reference, therefore it should be popped."""
    for passages in passages_by_citation.values():
        if passages:
            passages.popitem()


def process_pdf(
    pdf_path: str,
    out_json_path: Optional[str] = None,
    json_indent: Optional[int] = 2,
) -> List[Dict[str, object]]:
    parser = IEEEParser()

    with fitz.open(pdf_path) as pdf_document:
        document_text = build_document_text(pdf_document)
        quote_ranges = find_quote_ranges(document_text)
        sentence_ranges = compute_sentence_ranges(document_text)
        passages_by_citation: Dict[str, Dict[str, None]] = {}

        citation_matches = parser.find_citations_in_text(document_text)
        for citation_match in citation_matches:
            raw_citation_text = normalize_text(citation_match["raw_text"])
            citation_numbers = citation_match["citation_numbers"]
            if not raw_citation_text:
                continue

            for citation_occurrence in re.finditer(re.escape(raw_citation_text), document_text):
                citation_start = citation_occurrence.start()
                quote_range = find_quote_before_citation(citation_start, quote_ranges, document_text)

                if quote_range:
                    passage_text = quote_range[2]
                else:
                    sentence_range = find_sentence_at_position(citation_start, sentence_ranges)
                    if not sentence_range:
                        continue
                    passage_text = sentence_range[2]

                passage_text = clean_passage_text(passage_text)
                if not passage_text:
                    continue

                for citation_text in citation_numbers:
                    citation_number = parse_citation_number(citation_text)
                    if not citation_number:
                        continue
                    citation_passages = passages_by_citation.setdefault(citation_number, {})
                    citation_passages.setdefault(passage_text, None)

        remove_last_passage_per_citation(passages_by_citation)

        results: List[Dict[str, object]] = []
        for citation_number, citation_passages in sorted(
            passages_by_citation.items(), key=lambda item: int(item[0])
        ):
            passages = list(citation_passages.keys())
            if not passages:
                continue
            results.append({"citation": f"[{citation_number}]", "sentences": passages})

    if out_json_path:
        os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, ensure_ascii=False, indent=json_indent)

    return results
