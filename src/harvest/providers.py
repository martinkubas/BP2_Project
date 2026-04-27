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
from .utils import (
    extract_pdf_url_from_response,
    extract_token_from_response,
    log_provider_transmission,
    write_response_pdf,
)


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
    transmission_log_path: Path,
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
            log_provider_transmission(transmission_log_path, "elsevier", doi, response.status_code)
            print(
                f"      [Elsevier] status={response.status_code} "
                f"ct={content_type} final={final_url}",
                flush=True,
            )
            if response.status_code >= 400:
                return False, final_url, f"HTTP {response.status_code}"
            if "application/pdf" not in content_type:
                return False, final_url, "not a PDF response"

            write_response_pdf(response, out_path)
            return True, final_url, ""
    except Exception as exc:
        return False, url, repr(exc)


def elsevier_abstract(
    session: requests.Session,
    doi: str,
    transmission_log_path: Path,
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
        log_provider_transmission(transmission_log_path, "elsevier", doi, response.status_code)
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

def _springer_request_spec(doi: str) -> Optional[tuple[str, Dict[str, str], Dict[str, str]]]:
    api_key = os.getenv("SPRINGER_FULLTEXT_API_KEY", "").strip()
    if not api_key:
        return None

    url_template = os.getenv("SPRINGER_FULLTEXT_URL_TEMPLATE", "").strip()
    if not url_template:
        print("      [Springer] missing SPRINGER_FULLTEXT_URL_TEMPLATE", flush=True)
        return None

    doi_normalized = normalize_doi(doi)
    encoded_doi = quote(doi_normalized, safe="")
    url = url_template.format(doi=encoded_doi, doi_raw=doi_normalized)

    headers = {
        "Accept": "application/pdf,application/json,application/xml,text/xml,text/plain;q=0.9,*/*;q=0.8",
    }
    params: Dict[str, str] = {}

    key_param = os.getenv("SPRINGER_FULLTEXT_API_KEY_PARAM", "api_key").strip()
    if key_param:
        params[key_param] = api_key

    key_header = os.getenv("SPRINGER_FULLTEXT_API_KEY_HEADER", "").strip()
    if key_header:
        headers[key_header] = api_key

    return url, headers, params


def download_springer_pdf(
    session: requests.Session,
    doi: str,
    out_path: Path,
    transmission_log_path: Path,
    connect_timeout: int,
    read_timeout: int,
) -> Tuple[bool, Optional[str], str]:
    request_spec = _springer_request_spec(doi)
    if not request_spec:
        return False, None, "missing SPRINGER_FULLTEXT_API_KEY or SPRINGER_FULLTEXT_URL_TEMPLATE"

    url, headers, params = request_spec
    print(f"      [Springer] GET {url}", flush=True)
    try:
        with session.get(
            url,
            params=params,
            headers=headers,
            stream=True,
            allow_redirects=True,
            timeout=(connect_timeout, read_timeout),
        ) as response:
            final_url = response.url
            content_type = (response.headers.get("Content-Type") or "").lower()
            log_provider_transmission(transmission_log_path, "springer", doi, response.status_code)
            print(
                f"      [Springer] status={response.status_code} "
                f"ct={content_type} final={final_url}",
                flush=True,
            )
            if response.status_code >= 400:
                return False, final_url, f"HTTP {response.status_code}"

            if "application/pdf" in content_type or final_url.lower().endswith(".pdf"):
                write_response_pdf(response, out_path)
                return True, final_url, ""

            pdf_url = extract_pdf_url_from_response(response)
            if pdf_url:
                print(f"      [Springer] payload exposed PDF URL -> {pdf_url}", flush=True)
                return download_pdf_http(session, pdf_url, out_path, connect_timeout, read_timeout)

            return False, final_url, "response did not contain a PDF or PDF URL"
    except Exception as exc:
        print(f"      [Springer] failed: {exc!r}", flush=True)
        return False, url, repr(exc)


# ---------------------------------------------------------------------------
# IEEE
# ---------------------------------------------------------------------------

def _ieee_article_metadata(
    session: requests.Session,
    doi: str,
    transmission_log_path: Path,
    connect_timeout: int,
    read_timeout: int,
) -> Dict[str, Any]:
    api_key = os.getenv("IEEE_API_KEY", "").strip()
    if not api_key:
        return {}

    params = {
        "apikey": api_key,
        "format": "json",
        "max_records": 1,
        "start_record": 1,
        "doi": normalize_doi(doi),
    }

    print("      [IEEE] metadata lookup by DOI", flush=True)
    try:
        response = session.get(
            "https://ieeexploreapi.ieee.org/api/v1/search/articles",
            params=params,
            timeout=(connect_timeout, read_timeout),
        )
        log_provider_transmission(transmission_log_path, "ieee", doi, response.status_code)
        print(f"      [IEEE] metadata status={response.status_code}", flush=True)
        if response.status_code >= 400:
            return {}
        articles = (response.json() or {}).get("articles") or []
        if articles and isinstance(articles[0], dict):
            return articles[0]
        return {}
    except Exception as exc:
        print(f"      [IEEE] metadata failed: {exc!r}", flush=True)
        return {}


def _ieee_fulltext_session_token(
    session: requests.Session,
    api_key: str,
    doi: str,
    transmission_log_path: Path,
    connect_timeout: int,
    read_timeout: int,
) -> str:
    for env_name in ("IEEE_FULL_TEXT_SESSION_TOKEN", "IEEE_SESSION_TOKEN"):
        token = os.getenv(env_name, "").strip()
        if token:
            return token

    auth_code = os.getenv("IEEE_FULL_TEXT_AUTH_CODE", "").strip()
    auth_token = os.getenv("IEEE_FULL_TEXT_AUTH_TOKEN", "").strip()
    if auth_token:
        auth_code = auth_token
    if not auth_code:
        return ""

    auth_endpoint = os.getenv(
        "IEEE_FULL_TEXT_AUTH_ENDPOINT",
        "http://ieeexploreapi.ieee.org/api/v1/auth/token",
    ).strip()
    auth_code_param = os.getenv("IEEE_FULL_TEXT_AUTH_CODE_PARAM", "cltoken").strip() or "cltoken"
    token_fields = tuple(
        field.strip()
        for field in os.getenv(
            "IEEE_FULL_TEXT_SESSION_TOKEN_FIELDS",
            "cltoken,token,authToken,sessionToken",
        ).split(",")
        if field.strip()
    )

    print(f"      [IEEE] requesting session token from {auth_endpoint}", flush=True)
    try:
        response = session.get(
            auth_endpoint,
            params={"apikey": api_key, auth_code_param: auth_code},
            timeout=(connect_timeout, read_timeout),
        )
        log_provider_transmission(transmission_log_path, "ieee", doi, response.status_code)
        print(f"      [IEEE] auth status={response.status_code}", flush=True)
        if response.status_code >= 400:
            return ""
        return extract_token_from_response(response, token_fields)
    except Exception as exc:
        print(f"      [IEEE] auth token request failed: {exc!r}", flush=True)
        return ""


def download_ieee_pdf(
    session: requests.Session,
    doi: str,
    out_path: Path,
    transmission_log_path: Path,
    connect_timeout: int,
    read_timeout: int,
) -> Tuple[bool, Optional[str], str]:
    api_key = os.getenv("IEEE_API_KEY", "").strip()
    if not api_key:
        return False, None, "missing IEEE_API_KEY"

    article = _ieee_article_metadata(
        session, doi, transmission_log_path, connect_timeout, read_timeout,
    )
    if not article:
        return False, None, "IEEE metadata lookup returned no article"

    article_number = str(
        article.get("article_number")
        or article.get("articleNumber")
        or article.get("articleId")
        or ""
    ).strip()
    pdf_url = str(article.get("pdf_url") or article.get("pdfUrl") or "").strip()
    access_type = str(article.get("access_type") or article.get("accessType") or "").strip()

    if pdf_url:
        print(f"      [IEEE] trying metadata PDF URL ({access_type or 'unknown access'})", flush=True)
        success, final_url, error = download_pdf_http(
            session, pdf_url, out_path, connect_timeout, read_timeout,
        )
        if success:
            return True, final_url, ""
        print(f"      [IEEE] metadata PDF URL failed: {error}", flush=True)

    if not article_number:
        return False, None, "IEEE metadata missing article_number and pdf_url"

    fulltext_url = f"https://ieeexploreapi.ieee.org/api/v1/search/document/{article_number}/fulltext"
    params = {"apikey": api_key, "format": os.getenv("IEEE_FULL_TEXT_FORMAT", "xml").strip() or "xml"}

    session_token = _ieee_fulltext_session_token(
        session, api_key, doi, transmission_log_path, connect_timeout, read_timeout,
    )
    if session_token:
        params["cltoken"] = session_token

    print(f"      [IEEE] GET {fulltext_url}", flush=True)
    try:
        with session.get(
            fulltext_url,
            params=params,
            headers={"Accept": "application/pdf,application/json,application/xml,text/xml,text/plain;q=0.9,*/*;q=0.8"},
            stream=True,
            allow_redirects=True,
            timeout=(connect_timeout, read_timeout),
        ) as response:
            final_url = response.url
            content_type = (response.headers.get("Content-Type") or "").lower()
            log_provider_transmission(transmission_log_path, "ieee", doi, response.status_code)
            print(
                f"      [IEEE] fulltext status={response.status_code} "
                f"ct={content_type} final={final_url}",
                flush=True,
            )
            if response.status_code >= 400:
                if not session_token:
                    return False, final_url, (
                        f"HTTP {response.status_code} (chargeable full text may require "
                        "IEEE_FULL_TEXT_SESSION_TOKEN or IEEE_FULL_TEXT_AUTH_CODE)"
                    )
                return False, final_url, f"HTTP {response.status_code}"

            if "application/pdf" in content_type or final_url.lower().endswith(".pdf"):
                write_response_pdf(response, out_path)
                return True, final_url, ""

            nested_pdf_url = extract_pdf_url_from_response(response)
            if nested_pdf_url:
                print(f"      [IEEE] payload exposed PDF URL -> {nested_pdf_url}", flush=True)
                return download_pdf_http(
                    session, nested_pdf_url, out_path, connect_timeout, read_timeout,
                )

            if not session_token and access_type and "open" not in access_type.lower():
                return False, final_url, (
                    "no PDF returned; provide IEEE_FULL_TEXT_SESSION_TOKEN or "
                    "IEEE_FULL_TEXT_AUTH_CODE for chargeable full text"
                )

            return False, final_url, "response did not contain a PDF or PDF URL"
    except Exception as exc:
        print(f"      [IEEE] fulltext failed: {exc!r}", flush=True)
        return False, fulltext_url, repr(exc)


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
