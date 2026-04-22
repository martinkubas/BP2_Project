"""Generic HTTP PDF downloader."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests

_HREF_RE = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.I)
_PDF_URL_PATTERN = re.compile(r"\.pdf(\?|$)|/pdf\b|format=pdf|type=pdf", re.I)

_MAX_HTML_BYTES = 2 * 1024 * 1024    # 2 MB cap on HTML reads before giving up
_MAX_PDF_BYTES = 150 * 1024 * 1024   # 150 MB cap on PDF writes
_MAX_HTML_PDF_HOPS = 2               # max depth when following HTML → PDF links


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121 Safari/537.36"
        ),
        "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,sk;q=0.8",
        "Connection": "keep-alive",
    })
    return session


def _extract_pdf_links_from_html(html: str, base_url: str) -> List[str]:
    pdf_urls: List[str] = []
    seen: set[str] = set()

    for href in _HREF_RE.findall(html or ""):
        href = (href or "").strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        absolute_url = urljoin(base_url, href)
        if _PDF_URL_PATTERN.search(absolute_url) and absolute_url not in seen:
            seen.add(absolute_url)
            pdf_urls.append(absolute_url)

    return pdf_urls[:20]


def download_pdf_http(
    session: requests.Session,
    url: str,
    out_path: Path,
    connect_timeout: int,
    read_timeout: int,
    max_bytes: int = _MAX_PDF_BYTES,
    _hop_depth: int = 0,
) -> Tuple[bool, Optional[str], str]:
    """Download a PDF from url, following redirects and HTML landing pages.

    On success, writes the PDF bytes to out_path and returns (True, final_url, "").
    On failure, returns (False, last_url_seen, error_description).

    Handles three response cases:
      1. PDF response: stream to file.
      2. HTML response: extract PDF-looking links, recurse on each.
      3. Anything else: return failure.
    """
    if _hop_depth > _MAX_HTML_PDF_HOPS:
        return False, None, "too many html→pdf hops"

    print(f"      [HTTP] GET {url}", flush=True)
    try:
        with session.get(
            url, stream=True, allow_redirects=True,
            timeout=(connect_timeout, read_timeout),
        ) as response:
            final_url = response.url
            content_type = (response.headers.get("Content-Type") or "").lower()
            print(
                f"      [HTTP] status={response.status_code} "
                f"ct={content_type} final={final_url}",
                flush=True,
            )

            if response.status_code >= 400:
                return False, final_url, f"HTTP {response.status_code}"

            is_pdf_response = (
                "application/pdf" in content_type
                or final_url.lower().endswith(".pdf")
            )

            if is_pdf_response:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                bytes_written = 0
                with open(out_path, "wb") as pdf_file:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        pdf_file.write(chunk)
                        bytes_written += len(chunk)
                        if bytes_written > max_bytes:
                            return False, final_url, "file too large (cap exceeded)"
                return True, final_url, ""

            # Not a PDF — read a limited amount as HTML and look for PDF links
            html_bytes = b""
            bytes_read = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                html_bytes += chunk
                bytes_read += len(chunk)
                if bytes_read >= _MAX_HTML_BYTES:
                    break

            try:
                html_text = html_bytes.decode(response.encoding or "utf-8", errors="replace")
            except Exception:
                html_text = html_bytes.decode("utf-8", errors="replace")

            is_html = (
                "text/html" in content_type
                or "<html" in html_text.lower()
                or "doctype html" in html_text.lower()
            )
            if is_html:
                pdf_links = _extract_pdf_links_from_html(html_text, final_url)
                print(
                    f"      [HTTP] HTML landing page; found {len(pdf_links)} PDF-like links",
                    flush=True,
                )
                for pdf_link in pdf_links:
                    success, link_final_url, error = download_pdf_http(
                        session, pdf_link, out_path,
                        connect_timeout, read_timeout,
                        max_bytes=max_bytes,
                        _hop_depth=_hop_depth + 1,
                    )
                    if success:
                        return True, link_final_url, ""
                return False, final_url, "html landing page — no working PDF link found"

            return False, final_url, "not a PDF response"

    except requests.exceptions.ConnectTimeout:
        return False, url, "connect timeout"
    except requests.exceptions.ReadTimeout:
        return False, url, "read timeout"
    except Exception as exc:
        return False, url, repr(exc)
