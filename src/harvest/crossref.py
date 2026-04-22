from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from .doi import normalize_doi

_JATS_TAG_RE = re.compile(r"</?[^>]+>")

def strip_jats(text: str) -> str:
    """Remove JATS/XML tags from an abstract string and collapse whitespace."""
    text = (text or "").strip()
    if not text:
        return ""
    text = _JATS_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()



_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _significant_title_tokens(title: str) -> List[str]:
    """Extract lowercase tokens of >= 5 chars from a title."""
    return [token.lower() for token in re.findall(r"[A-Za-z]{5,}", title)]


def _score_candidate(crossref_item: Dict[str, Any], raw_citation: str) -> float:
    """Compute a match score for one CrossRef candidate against the raw citation.

    Score components:

      1. Title token overlap ratio (0.0–1.0):
         Fraction of _significant_title_tokens(title) found in raw_citation.lower().
         Weight: 1.0. Returns 0.0 if the title has no significant tokens.

      2. Year compatibility (−0.3 / 0.0 / +0.3):
         +0.3 if CrossRef year matches a 4-digit year in the citation (within ±1).
         −0.3 if years are present on both sides but disagree by more than 1 year.
         0.0 if either side has no extractable year.

      3. First-author family name bonus (+0.2):
         +0.2 if CrossRef first author family name appears in the citation (case-insensitive).
         0.0 if CrossRef has no author data or name is too short to be reliable.
    """
    title_list = crossref_item.get("title") or []
    title = str(title_list[0]).strip() if title_list else ""
    if not title:
        return 0.0

    citation_lower = raw_citation.lower()

    # 1. Title token overlap
    title_tokens = _significant_title_tokens(title)
    if title_tokens:
        matched = sum(1 for token in title_tokens if token in citation_lower)
        overlap_ratio = matched / len(title_tokens)
    else:
        overlap_ratio = 0.0
    score = overlap_ratio

    # 2. Year compatibility
    issued = (crossref_item.get("issued") or {}).get("date-parts") or []
    crossref_year: Optional[int] = None
    if issued and isinstance(issued[0], list) and issued[0]:
        try:
            crossref_year = int(issued[0][0])
        except (TypeError, ValueError):
            pass

    citation_years = [int(m) for m in _YEAR_RE.findall(raw_citation)]
    if crossref_year and citation_years:
        if any(abs(y - crossref_year) <= 1 for y in citation_years):
            score += 0.3
        else:
            score -= 0.3

    # 3. First-author family name bonus
    authors = crossref_item.get("author") or []
    if authors and isinstance(authors[0], dict):
        family_name = (authors[0].get("family") or "").lower().strip()
        if len(family_name) >= 3 and family_name in citation_lower:
            score += 0.2

    return score


def crossref_biblio_lookup(
    session: requests.Session,
    mailto: str,
    raw_citation: str,
    connect_timeout: int,
    read_timeout: int,
    min_score: float,
) -> Optional[str]:
    """Search CrossRef by bibliographic string and return the best matching DOI.

    Sends a query.bibliographic request for the top 5 results, scores each
    candidate against the raw citation string, and accepts the highest-scoring
    one if its score meets the minimum threshold.
    """
    raw_citation = (raw_citation or "").strip()
    if not raw_citation:
        return None

    params = {
        "query.bibliographic": raw_citation,
        "rows": 5,
        "select": "DOI,title,author,issued,URL,score",
        "mailto": mailto,
    }

    print("    [CrossRef] bibliographic search ...", flush=True)
    try:
        response = session.get(
            "https://api.crossref.org/works",
            params=params,
            timeout=(connect_timeout, read_timeout),
        )
        print(f"    [CrossRef] status={response.status_code}", flush=True)
        if response.status_code >= 400:
            return None

        candidates = (response.json().get("message", {}) or {}).get("items", []) or []
        if not candidates:
            return None

        scored = [
            (item, _score_candidate(item, raw_citation))
            for item in candidates
        ]
        best_item, best_score = max(scored, key=lambda pair: pair[1])

        title_list = best_item.get("title") or []
        best_title = str(title_list[0]).strip() if title_list else "(no title)"

        if best_score < min_score:
            print(
                f"    [CrossRef] rejected best candidate (score={best_score:.2f} < "
                f"{min_score}): {best_title[:90]!r}",
                flush=True,
            )
            return None

        doi = normalize_doi(best_item.get("DOI") or "")
        if not doi:
            return None

        print(
            f"    [CrossRef] accepted DOI={doi} score={best_score:.2f} "
            f"title={best_title[:90]!r}",
            flush=True,
        )
        return doi

    except Exception as exc:
        print(f"    [CrossRef] bibliographic search failed: {exc!r}", flush=True)
        return None


def crossref_work(
    session: requests.Session,
    mailto: str,
    doi: str,
    connect_timeout: int,
    read_timeout: int,
) -> Optional[Dict[str, Any]]:
    """Fetch the full CrossRef work metadata for a known DOI."""
    doi_normalized = normalize_doi(doi)
    url = f"https://api.crossref.org/works/{quote(doi_normalized, safe='')}"

    print("    [CrossRef] fetch work metadata ...", flush=True)
    try:
        response = session.get(
            url, params={"mailto": mailto},
            timeout=(connect_timeout, read_timeout),
        )
        print(f"    [CrossRef] status={response.status_code}", flush=True)
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            return None
        body = response.json()
        if isinstance(body, dict):
            return body.get("message") or None
        return None
    except Exception as exc:
        print(f"    [CrossRef] work fetch failed: {exc!r}", flush=True)
        return None


def crossref_pdf_links(work_meta: Dict[str, Any]) -> List[str]:
    pdf_urls: List[str] = []
    seen: set[str] = set()

    for link_entry in (work_meta.get("link") or []):
        if not isinstance(link_entry, dict):
            continue
        content_type = (link_entry.get("content-type") or "").lower()
        url = link_entry.get("URL") or ""
        if content_type == "application/pdf" and url and url not in seen:
            seen.add(url)
            pdf_urls.append(url)

    return pdf_urls


def crossref_abstract_text(work_meta: Dict[str, Any]) -> str:
    return strip_jats((work_meta or {}).get("abstract") or "")


def crossref_first_title(work_meta: Dict[str, Any]) -> str:
    titles = (work_meta or {}).get("title") or []
    if isinstance(titles, list) and titles:
        return str(titles[0]).strip()
    if isinstance(titles, str):
        return titles.strip()
    return ""
