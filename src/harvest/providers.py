"""Publisher-specific PDF downloaders and abstract fallbacks."""
from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode, urlparse, urlunparse, parse_qs

import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

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

_SENSITIVE_PARAMS = {"api_key", "apikey", "token", "key"}


def _strip_sensitive_params(url: str) -> str:
    """Remove API key / token query parameters from a URL before storing it."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query, keep_blank_values=True)
    cleaned = {k: v for k, v in qs.items() if k.lower() not in _SENSITIVE_PARAMS}
    return urlunparse(parsed._replace(query=urlencode(cleaned, doseq=True)))


def _springer_request_spec(doi: str) -> Optional[tuple[str, Dict[str, str], Dict[str, str]]]:
    api_key = (
        os.getenv("SPRINGER_TDM_API_KEY", "").strip()
    )
    if not api_key:
        return None

    metric = os.getenv("SPRINGER_TDM_METRIC", "").strip()
    if metric:
        api_key = f"{api_key}/{metric}"

    url_template = os.getenv("SPRINGER_FULLTEXT_URL_TEMPLATE", "").strip()
    if not url_template:
        url_template = "https://spdi.public.springernature.app/xmldata/jats?q=doi:{doi_raw}"

    doi_normalized = normalize_doi(doi)
    encoded_doi = quote(doi_normalized, safe="")
    url = url_template.format(doi=encoded_doi, doi_raw=doi_normalized)

    headers = {
        "Accept": "application/xml,text/xml,application/pdf,application/json,text/plain;q=0.9,*/*;q=0.8",
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
            final_url = _strip_sensitive_params(response.url)
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

            if "xml" in content_type or "text/plain" in content_type:
                body = response.text
                print(f"      [Springer] JATS XML response, extracting fulltext", flush=True)
                paragraphs = _extract_text_from_xml(body)
                if _create_pdf_from_text(paragraphs, out_path, doi):
                    return True, final_url, ""
                return False, final_url, "JATS XML received but contained no usable fulltext"

            return False, final_url, "response did not contain a PDF or PDF URL"
    except Exception as exc:
        print(f"      [Springer] failed: {exc!r}", flush=True)
        return False, url, repr(exc)


# ---------------------------------------------------------------------------
# Springer JATS abstract helper
# ---------------------------------------------------------------------------

def springer_abstract_from_jats(xml_text: str) -> str:
    """Extract abstract text from a Springer TDM JATS XML response."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""
    for el in root.iter():
        tag = re.sub(r"\{[^}]*\}", "", el.tag).lower()
        if tag == "abstract":
            text = "".join(el.itertext()).strip()
            if text:
                return text
    return ""


# ---------------------------------------------------------------------------
# XML → PDF helpers
# ---------------------------------------------------------------------------

def _extract_text_from_xml(xml_text: str) -> List[str]:
    """Extract readable paragraphs from IEEE fulltext XML."""
    paragraphs: List[str] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        raw = re.sub(r"<[^>]+>", " ", xml_text)
        raw = re.sub(r"\s+", " ", raw).strip()
        return [raw] if raw else []

    def text_of(el: ET.Element) -> str:
        return "".join(el.itertext()).strip()

    def tag(el: ET.Element) -> str:
        return re.sub(r"\{[^}]*\}", "", el.tag).lower()

    for el in root.iter():
        if tag(el) in ("article-title", "title", "documenttitle", "articletitle"):
            t = text_of(el)
            if t and len(t) < 500:
                paragraphs.append(t)
                break

    seen: set = set()
    para_tags = {
        "p", "para", "paragraph", "abstract", "sec", "section",
        "body", "fulltext", "articlecontent", "textbody",
    }
    for el in root.iter():
        if tag(el) in para_tags:
            t = text_of(el)
            if t and t not in seen and len(t) > 20:
                paragraphs.append(t)
                seen.add(t)

    if not paragraphs:
        for el in root.iter():
            t = (el.text or "").strip()
            if len(t) > 30 and t not in seen:
                paragraphs.append(t)
                seen.add(t)

    return paragraphs


def _create_pdf_from_text(paragraphs: List[str], out_path: Path, doi: str) -> bool:
    """Render extracted text paragraphs into a PDF using reportlab."""
    if not paragraphs:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"DOI: {doi}", styles["Heading2"]))
    story.append(Spacer(1, 0.3 * cm))

    for para in paragraphs:
        cleaned = para.strip()
        if not cleaned:
            continue
        # Escape reportlab special chars
        cleaned = cleaned.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(cleaned, styles["Normal"]))
        story.append(Spacer(1, 0.15 * cm))

    try:
        doc = SimpleDocTemplate(str(out_path), pagesize=A4)
        doc.build(story)
        return True
    except Exception as exc:
        print(f"      [IEEE] PDF creation failed: {exc!r}", flush=True)
        return False


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


_IEEE_ARTICLE_NUM_RE = re.compile(r"/document/(\d+)", re.I)


def _ieee_article_number_from_doi(
    session: requests.Session,
    doi: str,
    connect_timeout: int,
    read_timeout: int,
) -> Optional[str]:
    url = f"https://doi.org/{normalize_doi(doi)}"
    print(f"      [IEEE] resolving DOI redirect: {url}", flush=True)
    try:
        resp = session.head(
            url,
            allow_redirects=True,
            timeout=(connect_timeout, read_timeout),
        )
        final = resp.url
        m = _IEEE_ARTICLE_NUM_RE.search(final)
        if m:
            print(f"      [IEEE] article number from redirect: {m.group(1)}", flush=True)
            return m.group(1)
        resp = session.get(
            url,
            allow_redirects=True,
            timeout=(connect_timeout, read_timeout),
        )
        m = _IEEE_ARTICLE_NUM_RE.search(resp.url)
        if m:
            print(f"      [IEEE] article number from redirect: {m.group(1)}", flush=True)
            return m.group(1)
    except Exception as exc:
        print(f"      [IEEE] DOI redirect failed: {exc!r}", flush=True)
    return None


_ieee_token_cache: tuple[str, float] = ("", 0.0)
_IEEE_TOKEN_TTL = 13 * 60  # refresh 2 min before the 15-min IEEE expiry


def _ieee_fulltext_session_token(
    session: requests.Session,
    api_key: str,
    doi: str,
    transmission_log_path: Path,
    connect_timeout: int,
    read_timeout: int,
) -> str:
    global _ieee_token_cache

    # 1. Cached token still valid?
    cached_token, expires_at = _ieee_token_cache
    if cached_token and time.monotonic() < expires_at:
        return cached_token

    # 2. Permanent auth code from env
    auth_code = (
        os.getenv("IEEE_AUTH_TOKEN", "")
    ).strip()
    if not auth_code:
        return ""

    auth_endpoint = "http://ieeexploreapi.ieee.org/api/v1/auth/token"

    literal_url = f"{auth_endpoint}&cltoken={auth_code}&apikey={api_key}"
    print(f"      [IEEE] requesting session token", flush=True)
    try:
        response = session.get(
            literal_url,
            timeout=(connect_timeout, read_timeout),
            allow_redirects=True,
        )
        log_provider_transmission(transmission_log_path, "ieee", doi, response.status_code)
        print(f"      [IEEE] auth status={response.status_code}", flush=True)
        if response.status_code >= 400:
            ct = (response.headers.get("Content-Type") or "").lower()
            snippet = response.text[:200].replace("\n", " ")
            print(f"      [IEEE] auth error body: {snippet}", flush=True)
            return ""
        token_fields = ("token", "cltoken", "authToken", "sessionToken")
        token = extract_token_from_response(response, token_fields)
        if token:
            _ieee_token_cache = (token, time.monotonic() + _IEEE_TOKEN_TTL)
        return token
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
    article_number: Optional[str] = None,
) -> Tuple[bool, Optional[str], str]:
    api_key = os.getenv("IEEE_API_KEY", "").strip()
    if not api_key:
        return False, None, "missing IEEE_API_KEY"

    access_type = ""
    if not article_number:
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
        article_number = _ieee_article_number_from_doi(
            session, doi, connect_timeout, read_timeout,
        )

    if not article_number:
        return False, None, "IEEE could not resolve article number"

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
            if response.status_code == 204:
                return False, final_url, "HTTP 204 — no fulltext available for this article (not digitized or outside access tier)"

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

            # IEEE fulltext API returns XML — convert it to a PDF
            if session_token and ("xml" in content_type or "json" in content_type):
                try:
                    body = response.text
                except Exception:
                    body = ""
                if body:
                    print(f"      [IEEE] converting XML fulltext to PDF", flush=True)
                    paragraphs = _extract_text_from_xml(body)
                    if _create_pdf_from_text(paragraphs, out_path, doi):
                        return True, final_url, ""
                    return False, final_url, "XML fulltext received but text extraction yielded no content"

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
