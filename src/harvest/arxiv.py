"""
ArXiv reference detection, metadata retrieval, and PDF download.

ArXiv references are identified early before any CrossRef lookup by
pattern-matching the raw citation string and reference URL.

This avoids routing arXiv references through CrossRef, which does not reliably
index arXiv preprints and often returns wrong or unrelated results.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from .http_downloader import download_pdf_http

_ARXIV_DOI_PREFIX_RE = re.compile(
    r"10\.48550/arxiv\.(\d{4}\.\d{4,5})(v\d+)?",
    re.I,
)
_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?",
    re.I,
)
_ARXIV_INLINE_RE = re.compile(
    r"\barxiv[:\s]+(\d{4}\.\d{4,5})(v\d+)?\b",
    re.I,
)

_ARXIV_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Pattern to strip the URL prefix and version suffix from the <id> element value
_ARXIV_ID_FROM_ATOM_RE = re.compile(
    r"arxiv\.org/abs/(\d{4}\.\d{4,5})",
    re.I,
)

@dataclass(frozen=True)
class ArXivResult:
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    pdf_url: str

    def __post_init__(self) -> None:
        pass


def is_arxiv_reference(
    reference_raw: str,
    reference_doi: str,
) -> bool:
    for text in (reference_doi or "", reference_raw or ""):
        if (
            _ARXIV_DOI_PREFIX_RE.search(text)
            or _ARXIV_URL_RE.search(text)
            or _ARXIV_INLINE_RE.search(text)
        ):
            return True
    return False


def extract_arxiv_id(
    reference_raw: str,
    reference_doi: str,
) -> Optional[str]:
    if reference_doi:
        match = _ARXIV_DOI_PREFIX_RE.search(reference_doi)
        if match:
            return match.group(1)

    raw = reference_raw or ""
    match = _ARXIV_URL_RE.search(raw)
    if match:
        return match.group(1)
    match = _ARXIV_INLINE_RE.search(raw)
    if match:
        return match.group(1)

    return None



def fetch_arxiv_metadata(
    session: requests.Session,
    arxiv_id: str,
    connect_timeout: int,
    read_timeout: int,
) -> Optional[ArXivResult]:
    print(f"    [arXiv] querying API for id={arxiv_id}", flush=True)
    try:
        response = session.get(
            _ARXIV_API_URL,
            params={"id_list": arxiv_id},
            timeout=(connect_timeout, read_timeout),
        )
        print(f"    [arXiv] status={response.status_code}", flush=True)
        if response.status_code >= 400:
            return None

        root = ET.fromstring(response.content)
        ns = {"atom": _ARXIV_ATOM_NS}

        entries = root.findall("atom:entry", ns)
        if not entries:
            print(f"    [arXiv] no entry found for id={arxiv_id}", flush=True)
            return None

        entry = entries[0]

        id_element = entry.find("atom:id", ns)
        raw_id_text = (id_element.text or "").strip() if id_element is not None else ""
        id_match = _ARXIV_ID_FROM_ATOM_RE.search(raw_id_text)
        canonical_id = id_match.group(1) if id_match else arxiv_id

        title_element = entry.find("atom:title", ns)
        title = " ".join((title_element.text or "").split()) if title_element is not None else ""

        summary_element = entry.find("atom:summary", ns)
        abstract = " ".join((summary_element.text or "").split()) if summary_element is not None else ""

        authors = [
            " ".join((name_el.text or "").split())
            for author_el in entry.findall("atom:author", ns)
            for name_el in [author_el.find("atom:name", ns)]
            if name_el is not None and name_el.text
        ]

        pdf_url = f"https://arxiv.org/pdf/{canonical_id}.pdf"
        print(f"    [arXiv] found: {title[:80]!r} authors={len(authors)}", flush=True)

        return ArXivResult(
            arxiv_id=canonical_id,
            title=title,
            authors=authors,
            abstract=abstract,
            pdf_url=pdf_url,
        )

    except Exception as exc:
        print(f"    [arXiv] metadata fetch failed: {exc!r}", flush=True)
        return None


def resolve_arxiv_reference(
    session: requests.Session,
    reference_raw: str,
    reference_doi: str,
    connect_timeout: int,
    read_timeout: int,
) -> tuple[Optional[str], Optional[str], Optional[ArXivResult]]:
    """Extract arXiv ID, fetch metadata, and build the canonical DOI."""
    arxiv_id = extract_arxiv_id(reference_raw, reference_doi)
    if not arxiv_id:
        return None, None, None
    arxiv_result = fetch_arxiv_metadata(session, arxiv_id, connect_timeout, read_timeout)
    return f"10.48550/arxiv.{arxiv_id}", arxiv_id, arxiv_result


def download_arxiv_pdf(
    session: requests.Session,
    arxiv_id: str,
    out_path: Path,
    connect_timeout: int,
    read_timeout: int,
) -> Tuple[bool, Optional[str], str]:
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return download_pdf_http(session, pdf_url, out_path, connect_timeout, read_timeout)
