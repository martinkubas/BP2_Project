#!/usr/bin/env python3
from __future__ import annotations

import unicodedata
import regex as re


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _collapse_spaces_keep_newlines(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(?m)^\s*\d+\s*$", "", s)
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _normalize_text(s: str) -> str:
    s = _nfc(s)
    s = s.replace("\u00a0", " ")
    s = s.replace("\u00b7", " ")
    return _collapse_spaces_keep_newlines(s)


def _flatten_reference_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"[ \t\f\v]+", " ", s).strip()
    return s
