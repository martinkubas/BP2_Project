"""
Per-reference download orchestration.

  1. arXiv-first detection.
  2. CrossRef DOI resolution.
  3. Milvus skip-check.
  4. On-disk existence check.
  5. Provider-specific download.
  6. HTTP fallback.
  7. Abstract fallback.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from .arxiv import (
    ArXivResult,
    download_arxiv_pdf,
    extract_arxiv_id,
    is_arxiv_reference,
    resolve_arxiv_reference,
)
from .crossref import (
    crossref_abstract_text,
    crossref_biblio_lookup,
    crossref_first_title,
    crossref_pdf_links,
    crossref_work,
)
from .doi import normalize_doi, pick_best_doi
from .http_downloader import download_pdf_http
from .providers import (
    detect_provider,
    download_elsevier_pdf,
    download_springer_oa_pdf,
    elsevier_abstract,
    ieee_abstract,
    save_abstract,
    springer_meta_abstract,
)
from .utils import (
    DownloadConfig,
    bump_both,
    log,
    safe_filename,
    set_download_info,
)

_PROVIDER_STAT_KEY: Dict[str, str] = {
    "arxiv":    "arxiv",
    "elsevier": "elsevier",
    "springer": "springer_oa",
}


def resolve_doi_for_reference(
    session: requests.Session,
    config: DownloadConfig,
    reference_doi: str,
    reference_raw: str,
) -> tuple[Optional[str], Optional[Dict[str, Any]], str]:
    doi = pick_best_doi(reference_doi, reference_raw)

    if doi:
        doi = normalize_doi(doi)
        log(f"    [DOI] extracted candidate={doi}")
        meta = crossref_work(
            session, config.crossref_mailto, doi,
            config.connect_timeout, config.read_timeout,
        )
        if meta:
            return doi, meta, "extracted"
        log("    [DOI] not validated by CrossRef — trying bibliographic search")

    if reference_raw:
        doi = crossref_biblio_lookup(
            session, config.crossref_mailto, reference_raw,
            config.connect_timeout, config.read_timeout,
            config.crossref_min_score,
        )
        if doi:
            doi = normalize_doi(doi)
            meta = crossref_work(
                session, config.crossref_mailto, doi,
                config.connect_timeout, config.read_timeout,
            )
            if meta:
                return doi, meta, "crossref_query"
            log("    [DOI] bibliographic search DOI did not validate via CrossRef")

    return None, None, ""


def attempt_provider_download(
    session: requests.Session,
    config: DownloadConfig,
    doi: str,
    provider: Optional[str],
    arxiv_id: Optional[str],
    reference_raw: str,
    out_pdf: Path,
) -> tuple[bool, Optional[str], str]:
    if provider == "arxiv":
        effective_id = arxiv_id or extract_arxiv_id(reference_raw, doi)
        if effective_id:
            log(f"    [Provider] arXiv id={effective_id}")
            return download_arxiv_pdf(
                session, effective_id, out_pdf,
                config.connect_timeout, config.read_timeout,
            )
        return False, None, "could not extract arXiv ID"

    if provider == "elsevier":
        log("    [Provider] Elsevier")
        return download_elsevier_pdf(
            session, doi, out_pdf, config.connect_timeout, config.read_timeout,
        )

    if provider == "springer":
        log("    [Provider] Springer OA")
        return download_springer_oa_pdf(
            session, doi, out_pdf, config.connect_timeout, config.read_timeout,
        )

    if provider == "ieee":
        log("    [Provider] IEEE (no PDF available via this pipeline)")
        return False, None, "ieee pdf not available via this pipeline (metadata only)"

    return False, None, "no provider"


def attempt_http_fallback(
    session: requests.Session,
    config: DownloadConfig,
    doi: str,
    crossref_meta: Dict[str, Any],
    out_pdf: Path,
) -> tuple[bool, Optional[str], str]:
    """Try CrossRef PDF links and doi.org."""
    candidate_urls: List[str] = []
    seen: set[str] = set()
    for url in [*crossref_pdf_links(crossref_meta), f"https://doi.org/{doi}"]:
        if url and url not in seen:
            seen.add(url)
            candidate_urls.append(url)

    log(f"    [HTTP fallback] {len(candidate_urls)} candidate URL(s)")

    last_final_url: Optional[str] = None
    last_error = "no candidates"
    for url in candidate_urls:
        if config.sleep_between_calls > 0:
            time.sleep(config.sleep_between_calls)
        success, last_final_url, last_error = download_pdf_http(
            session, url, out_pdf, config.connect_timeout, config.read_timeout,
        )
        if success:
            return True, last_final_url, ""

    return False, last_final_url, last_error


def fetch_abstract_fallback(
    session: requests.Session,
    config: DownloadConfig,
    doi: str,
    provider: Optional[str],
    crossref_meta: Optional[Dict[str, Any]],
) -> str:
    """Fetch an abstract when PDF download fails."""
    if crossref_meta:
        abstract = crossref_abstract_text(crossref_meta)
        log(f"    [Abstract] CrossRef abstract chars={len(abstract)}")
        if abstract:
            return abstract

    if provider == "elsevier":
        return elsevier_abstract(session, doi, config.connect_timeout, config.read_timeout)
    if provider == "ieee":
        return ieee_abstract(session, doi, config.connect_timeout, config.read_timeout)
    if provider == "springer":
        return springer_meta_abstract(session, doi, config.connect_timeout, config.read_timeout)

    return ""


def process_reference(
    session: requests.Session,
    config: DownloadConfig,
    link: Dict[str, Any],
    ref_index: int,
    ref_count: int,
    stats: Dict[str, Any],
    work_stats: Dict[str, Any],
    milvus_is_indexed: Callable[[str], bool],
) -> None:
    """Process one reference: resolve DOI, download PDF or abstract, update dicts."""
    reference = link.get("reference")
    if not isinstance(reference, dict):
        reference = {}
        link["reference"] = reference

    reference_raw = reference.get("raw") or ""
    reference_doi = reference.get("doi") or ""

    if config.overwrite:
        reference.pop("resolved_doi", None)
        reference.pop("download", None)

    log(f"  [Ref {ref_index}/{ref_count}] index={link.get('index')} occ={link.get('occurrences')}")
    if reference_raw:
        log(f"    raw: {reference_raw[:140]}{'...' if len(reference_raw) > 140 else ''}")
    if reference_doi:
        log(f"    doi: {str(reference_doi)[:120]}")

    # ---- 1: Resolve DOI ------------------------------------------------
    arxiv_id: Optional[str] = None
    arxiv_result: Optional[ArXivResult] = None
    crossref_meta: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None
    doi: Optional[str] = None
    doi_source = ""

    if is_arxiv_reference(reference_raw, reference_doi):
        doi, arxiv_id, arxiv_result = resolve_arxiv_reference(
            session, reference_raw, reference_doi,
            config.connect_timeout, config.read_timeout,
        )
        if doi:
            log(f"    [arXiv] id={arxiv_id} — skipping CrossRef")
            provider = "arxiv"
            doi_source = "arxiv"
        else:
            log("    [arXiv] detected but could not extract ID — falling back to CrossRef")

    if doi is None:
        doi, crossref_meta, doi_source = resolve_doi_for_reference(
            session, config, reference_doi, reference_raw,
        )

    if not doi:
        log("    [Skip] no valid DOI found")
        reference["resolved_doi"] = None
        set_download_info(reference, status="no_valid_doi")
        bump_both(stats, work_stats, "no_doi")
        return

    doi = normalize_doi(crossref_meta.get("DOI") or doi if crossref_meta else doi)
    reference["resolved_doi"] = doi
    reference["crossref_title"] = (
        crossref_first_title(crossref_meta) if crossref_meta
        else (arxiv_result.title if arxiv_result else "")
    )
    bump_both(stats, work_stats, "doi_found")
    log(f"    DOI={doi} (source={doi_source})")

    # ---- 2: Milvus skip-check ------------------------------------------
    try:
        if milvus_is_indexed(doi):
            log("    [Milvus] already indexed — skipping download")
            set_download_info(
                reference,
                status="already_indexed",
                pdf_path=None,
                abstract_path=None,
                final_url=str((crossref_meta or {}).get("URL") or f"https://doi.org/{doi}"),
                method="milvus_cache",
                provider=provider or detect_provider(doi, crossref_meta) or "unknown",
                error=None,
            )
            bump_both(stats, work_stats, "already_indexed")
            return
    except Exception as milvus_error:
        log(f"    [Milvus] check failed ({milvus_error!r}), proceeding with download")

    if config.sleep_between_calls > 0:
        time.sleep(config.sleep_between_calls)

    if provider is None:
        provider = detect_provider(doi, crossref_meta)

    out_pdf = config.pdf_dir / (safe_filename(doi) + ".pdf")

    # ---- 3: On-disk existence check ------------------------------------
    if out_pdf.exists() and not config.overwrite:
        log(f"    [Skip] exists on disk: {out_pdf.name}")
        set_download_info(
            reference,
            status="downloaded",
            pdf_path=os.path.relpath(str(out_pdf), str(config.output_dir)),
            method="exists",
            provider=None,
        )
        bump_both(stats, work_stats, "pdf_ok")
        return

    # ---- 4: Provider-specific download ---------------------------------
    pdf_ok, final_url, download_error = attempt_provider_download(
        session, config, doi, provider, arxiv_id,
        reference_raw, out_pdf,
    )

    if pdf_ok:
        stat_key = _PROVIDER_STAT_KEY.get(provider or "")
        if stat_key:
            stats["by_provider_pdf_ok"][stat_key] += 1
        set_download_info(
            reference,
            status="downloaded",
            pdf_path=os.path.relpath(str(out_pdf), str(config.output_dir)),
            final_url=final_url,
            method=f"provider:{provider}",
            provider=provider,
            error=None,
        )
        bump_both(stats, work_stats, "pdf_ok")
        return

    if provider:
        log(f"    [Provider] failed: {download_error}")

    # ---- 5: HTTP fallback ----------------------------------------------
    pdf_ok, final_url, download_error = attempt_http_fallback(
        session, config, doi, crossref_meta or {}, out_pdf,
    )

    if pdf_ok:
        stats["by_provider_pdf_ok"]["http"] += 1
        set_download_info(
            reference,
            status="downloaded",
            pdf_path=os.path.relpath(str(out_pdf), str(config.output_dir)),
            final_url=final_url,
            method="http",
            provider="http",
            error=None,
        )
        bump_both(stats, work_stats, "pdf_ok")
        log("    [OK] downloaded via HTTP fallback")
        return

    # ---- 6: Abstract fallback ------------------------------------------
    bump_both(stats, work_stats, "pdf_fail")
    log(f"    [FAIL] PDF download failed: {download_error}")
    log("    [Abstract] trying ...")

    abstract_text = fetch_abstract_fallback(
        session, config, doi, provider, crossref_meta,
    )
    abstract_path = save_abstract(config.abstract_dir, doi, abstract_text)

    if abstract_path:
        set_download_info(
            reference,
            status="abstract_saved",
            abstract_path=os.path.relpath(abstract_path, str(config.output_dir)),
            pdf_path=None,
            final_url=final_url,
            method="abstract",
            provider=provider or "unknown",
            error=None,
        )
        bump_both(stats, work_stats, "abstract_ok")
        log(f"    [OK] abstract saved → {abstract_path}")
    else:
        set_download_info(
            reference,
            status="download_failed",
            pdf_path=None,
            abstract_path=None,
            final_url=final_url,
            method="failed",
            provider=provider or "http",
            error=download_error,
        )
        bump_both(stats, work_stats, "abstract_fail")
        log("    [FAIL] no abstract available either")
