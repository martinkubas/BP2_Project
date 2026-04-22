#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Tuple

import regex as re


SENTENCE_BOUNDARY_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=(?:["\'\u201c\u201e\u2018\u00ab\u2039(\[{])?[A-Z\u00C0-\u017D0-9])'
)
QUOTE_PATTERNS = [
    re.compile(r"\u201c[^\u201d]{1,5000}\u201d"),
    re.compile(r"\u201e[^\u201c]{1,5000}\u201c"),
    re.compile(r"\u00ab[^\u00bb]{1,5000}\u00bb"),
    re.compile(r"\u2039[^\u203a]{1,5000}\u203a"),
    re.compile(r"\u2018[^\u2019]{1,5000}\u2019"),
    re.compile(r'"[^"\n]{1,10000}"'),
    re.compile(r"(?<!\w)'[^'\n]{1,600}'(?!\w)"),
]
TEXT_ALLOWED_BETWEEN_QUOTE_AND_CITATION = re.compile(r"^[\s\)\]\}\.,;:!?\u2014\u2013-]*$")
MAX_CHARACTERS_BETWEEN_QUOTE_AND_CITATION = 3


def split_sentences(text: str) -> List[str]:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()

    dot_placeholder = "\u2E2F"
    exclamation_placeholder = "\u2E30"
    question_placeholder = "\u2E31"
    hide_punctuation = str.maketrans(
        {
            ".": dot_placeholder,
            "!": exclamation_placeholder,
            "?": question_placeholder,
        }
    )
    restore_punctuation = str.maketrans(
        {
            dot_placeholder: ".",
            exclamation_placeholder: "!",
            question_placeholder: "?",
        }
    )

    quote_matches = []
    for quote_pattern in QUOTE_PATTERNS:
        quote_matches.extend(list(quote_pattern.finditer(text)))
    quote_matches.sort(key=lambda match: match.start(), reverse=True)

    for quote_match in quote_matches:
        quoted_text = text[quote_match.start() : quote_match.end()]
        text = (
            text[: quote_match.start()]
            + quoted_text.translate(hide_punctuation)
            + text[quote_match.end() :]
        )

    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_PATTERN.split(text) if sentence.strip()]
    return [sentence.translate(restore_punctuation) for sentence in sentences]


def compute_sentence_ranges(text: str) -> List[Tuple[int, int, str]]:
    sentences = split_sentences(text)
    sentence_ranges: List[Tuple[int, int, str]] = []
    search_start = 0

    for sentence in sentences:
        sentence_start = text.find(sentence, search_start)
        if sentence_start == -1:
            sentence_start = text.find(sentence)
            if sentence_start == -1:
                continue

        sentence_end = sentence_start + len(sentence)
        sentence_ranges.append((sentence_start, sentence_end, sentence))
        search_start = sentence_end

    return sentence_ranges


def find_quote_ranges(text: str):
    quote_ranges = []
    for quote_pattern in QUOTE_PATTERNS:
        for match in quote_pattern.finditer(text):
            quote_ranges.append((match.start(), match.end(), match.group(0)))
    quote_ranges.sort(key=lambda item: item[0])
    return quote_ranges


def find_quote_before_citation(citation_start: int, quote_ranges, text: str):
    candidate_quotes = [
        (quote_start, quote_end, quoted_text)
        for quote_start, quote_end, quoted_text in quote_ranges
        if quote_end <= citation_start
        and (citation_start - quote_end) <= MAX_CHARACTERS_BETWEEN_QUOTE_AND_CITATION
    ]
    if not candidate_quotes:
        return None

    quote_start, quote_end, quoted_text = max(candidate_quotes, key=lambda item: item[1])
    text_between = text[quote_end:citation_start]
    if TEXT_ALLOWED_BETWEEN_QUOTE_AND_CITATION.match(text_between):
        return (quote_start, quote_end, quoted_text)
    return None


def find_sentence_at_position(position: int, sentence_ranges):
    for sentence_start, sentence_end, sentence in sentence_ranges:
        if sentence_start <= position < sentence_end:
            return (sentence_start, sentence_end, sentence)
    return None
