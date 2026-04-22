#!/usr/bin/env python3
from __future__ import annotations

import fitz  # PyMuPDF
import regex as re

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_document_text(pdf_document: fitz.Document) -> str:
    page_text = "\n\n".join(page.get_text() for page in pdf_document)
    return normalize_text(page_text)


def clean_passage_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text.strip()
