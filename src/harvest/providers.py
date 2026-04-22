"""Publisher-specific PDF downloaders and abstract fallbacks."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

import requests

from .crossref import strip_jats
from .doi import normalize_doi
from .http_downloader import download_pdf_http


_DOI_PREFIX_PROVIDERS: Dict[str, str] = {
    "10.48550/arxiv.": "arxiv",
    "10.1016/":        "elsevier",
    "10.1007/":        "springer",
    "10.1109/":        "ieee",
}

_PUBLISHER_NAME_PROVIDERS: Tuple[Tuple[str, str], ...] = (
    ("elsevier",        "elsevier"),
    ("springer nature", "springer"),
    ("springer",        "springer"),
    ("ieee",            "ieee"),
)


def detect_provider(
    doi: str,
    crossref_meta: Optional[Dict[str, Any]],
) -> Optional[str]:
    doi_normalized = normalize_doi(doi)
    for prefix, provider in _DOI_PREFIX_PROVIDERS.items():
        if doi_normalized.startswith(prefix):
            return provider

    publisher = ((crossref_meta or {}).get("publisher") or "").lower()
    for name_fragment, provider in _PUBLISHER_NAME_PROVIDERS:
        if name_fragment in publisher:
            return provider

    return None


# ---------------------------------------------------------------------------
# Elsevier
# ---------------------------------------------------------------------------

def download_elsevier_pdf(
    session: requests.Session,
    doi: str,
    out_path: Path,
    connect_timeout: int,
    read_timeout: int,
) -> Tuple[bool, Optional[str], str]:
    api_key = os.getenv("ELSEVIER_API_KEY", "").strip()
    if not api_key:
        return False, None, "missing ELSEVIER_API_KEY"

    insttoken = os.getenv("ELSEVIER_INSTTOKEN", "").strip()
    url = f"https://api.elsevier.com/content/article/doi/{quote(normalize_doi(doi), safe='')}"
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/pdf"}
    if insttoken:
        headers["X-ELS-Insttoken"] = insttoken

    print(f"      [Elsevier] GET {url} (PDF)", flush=True)
    try:
        with session.get(
            url, headers=headers, stream=True, allow_redirects=True,
            timeout=(connect_timeout, read_timeout),
        ) as response:
            final_url = response.url
            content_type = (response.headers.get("Content-Type") or "").lower()
            print(
                f"      [Elsevier] status={response.status_code} "
                f"ct={content_type} final={final_url}",
                flush=True,
            )
            if response.status_code >= 400:
                return False, final_url, f"HTTP {response.status_code}"
            if "application/pdf" not in content_type:
                return False, final_url, "not a PDF response"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as pdf_file:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        pdf_file.write(chunk)
            return True, final_url, ""
    except Exception as exc:
        return False, url, repr(exc)


def elsevier_abstract(
    session: requests.Session,
    doi: str,
    connect_timeout: int,
    read_timeout: int,
) -> str:
    api_key = os.getenv("ELSEVIER_API_KEY", "").strip()
    if not api_key:
        return ""

    url = f"https://api.elsevier.com/content/article/doi/{quote(normalize_doi(doi), safe='')}"
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}

    print(f"      [Elsevier-abstract] GET {url} (JSON)", flush=True)
    try:
        response = session.get(url, headers=headers, timeout=(connect_timeout, read_timeout))
        print(f"      [Elsevier-abstract] status={response.status_code}", flush=True)
        if response.status_code >= 400:
            return ""
        data = response.json() or {}
        full_text = data.get("full-text-retrieval-response") or {}
        coredata = full_text.get("coredata") or {}
        description = coredata.get("dc:description") or ""
        return strip_jats(str(description))
    except Exception as exc:
        print(f"      [Elsevier-abstract] failed: {exc!r}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# Springer
# ---------------------------------------------------------------------------

def _springer_oa_pdf_url(
    session: requests.Session,
    doi: str,
    connect_timeout: int,
    read_timeout: int,
) -> Optional[str]:
    api_key = os.getenv("SPRINGER_OA_API_KEY", "").strip()
    if not api_key:
        return None

    doi_normalized = normalize_doi(doi)
    params = {"q": f"doi:{doi_normalized}", "api_key": api_key}

    print(f"      [Springer-OA] query doi:{doi_normalized}", flush=True)
    try:
        response = session.get(
            "https://api.springernature.com/openaccess/json",
            params=params,
            timeout=(connect_timeout, read_timeout),
        )
        print(f"      [Springer-OA] status={response.status_code}", flush=True)
        if response.status_code >= 400:
            return None

        records = (response.json() or {}).get("records") or []
        if not records or not isinstance(records[0], dict):
            return None

        url_entries = records[0].get("url") or []
        for entry in url_entries:
            if not isinstance(entry, dict):
                continue
            value = (entry.get("value") or "").strip()
            fmt = (entry.get("format") or entry.get("type") or "").lower()
            if value and (fmt == "pdf" or value.lower().endswith(".pdf") or "/pdf" in value.lower()):
                return value

        return None

    except Exception as exc:
        print(f"      [Springer-OA] failed: {exc!r}", flush=True)
        return None


def download_springer_oa_pdf(
    session: requests.Session,
    doi: str,
    out_path: Path,
    connect_timeout: int,
    read_timeout: int,
) -> Tuple[bool, Optional[str], str]:
    pdf_url = _springer_oa_pdf_url(session, doi, connect_timeout, read_timeout)
    if not pdf_url:
        return False, None, "no OA PDF URL found (paper may not be open access)"
    print(f"      [Springer-OA] PDF link → {pdf_url}", flush=True)
    return download_pdf_http(session, pdf_url, out_path, connect_timeout, read_timeout)


def springer_meta_abstract(
    session: requests.Session,
    doi: str,
    connect_timeout: int,
    read_timeout: int,
) -> str:
    api_key = os.getenv("SPRINGER_META_API_KEY", "").strip()
    if not api_key:
        return ""

    doi_normalized = normalize_doi(doi)
    params = {"q": f"doi:{doi_normalized}", "api_key": api_key}

    print(f"      [Springer-meta] query doi:{doi_normalized}", flush=True)
    try:
        response = session.get(
            "https://api.springernature.com/meta/v2/json",
            params=params,
            timeout=(connect_timeout, read_timeout),
        )
        print(f"      [Springer-meta] status={response.status_code}", flush=True)
        if response.status_code >= 400:
            return ""
        records = (response.json() or {}).get("records") or []
        if not records or not isinstance(records[0], dict):
            return ""
        abstract = records[0].get("abstract") or records[0].get("summary") or ""
        return strip_jats(str(abstract))
    except Exception as exc:
        print(f"      [Springer-meta] failed: {exc!r}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# IEEE
# ---------------------------------------------------------------------------

def ieee_abstract(
    session: requests.Session,
    doi: str,
    connect_timeout: int,
    read_timeout: int,
) -> str:
    api_key = os.getenv("IEEE_API_KEY", "").strip()
    if not api_key:
        return ""

    params = {
        "apikey": api_key,
        "format": "json",
        "max_records": 1,
        "start_record": 1,
        "doi": normalize_doi(doi),
    }

    print("      [IEEE-abstract] GET ieeexploreapi (DOI search)", flush=True)
    try:
        response = session.get(
            "https://ieeexploreapi.ieee.org/api/v1/search/articles",
            params=params,
            timeout=(connect_timeout, read_timeout),
        )
        print(f"      [IEEE-abstract] status={response.status_code}", flush=True)
        if response.status_code >= 400:
            return ""
        articles = (response.json() or {}).get("articles") or []
        if articles and isinstance(articles[0], dict):
            abstract = articles[0].get("abstract") or articles[0].get("abstractText") or ""
            return strip_jats(str(abstract))
        return ""
    except Exception as exc:
        print(f"      [IEEE-abstract] failed: {exc!r}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# Abstract file saving
# ---------------------------------------------------------------------------

def save_abstract(
    abstract_dir: Path,
    doi: str,
    text: str,
) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None

    abstract_dir.mkdir(parents=True, exist_ok=True)
    filename = _doi_to_safe_filename(doi) + ".txt"
    file_path = abstract_dir / filename

    with open(file_path, "w", encoding="utf-8") as abstract_file:
        abstract_file.write(text)
        abstract_file.write("\n")

    return str(file_path)


def _doi_to_safe_filename(doi: str, max_length: int = 180) -> str:
    name = (doi or "").strip()
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if len(name) > max_length:
        name = name[:max_length].rstrip("_")
    return name or "file"
