from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .http_downloader import make_session
from .reference import process_reference
from .utils import (
    DownloadConfig,
    build_milvus_checker,
    initial_stats,
    log,
)
from commons.io import iter_json_files, write_json_atomic


def run(
    input_dir: Path,
    output_dir: Path,
    enriched_dir: Path | None = None,
    crossref_mailto: str = "",
    connect_timeout: int = 8,
    read_timeout: int = 60,
    sleep_between_calls: float = 0.0,
    overwrite: bool = False,
    max_refs_per_work: int = 0,
    crossref_min_score: float = 0.25,
    milvus_uri: str = "",
    embed_model: str = "",
    pdf_dir: Path | None = None,
    abstract_dir: Path | None = None,
) -> int:
    """
    Returns 0 on success. Individual reference errors are logged but never abort
    the run — all references in all work files are always attempted.
    """
    if not crossref_mailto:
        raise ValueError("crossref_mailto is required")

    if enriched_dir is None:
        enriched_dir = input_dir

    resolved_pdf_dir = pdf_dir if pdf_dir is not None else output_dir / "pdfs"
    resolved_abstract_dir = abstract_dir if abstract_dir is not None else output_dir / "abstracts"
    log_dir = output_dir / "logs"
    for directory in (output_dir, resolved_pdf_dir, resolved_abstract_dir, log_dir, enriched_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = DownloadConfig(
        crossref_mailto=crossref_mailto,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        sleep_between_calls=sleep_between_calls,
        overwrite=overwrite,
        max_refs_per_work=max_refs_per_work,
        crossref_min_score=crossref_min_score,
        pdf_dir=resolved_pdf_dir,
        abstract_dir=resolved_abstract_dir,
        log_dir=log_dir,
        transmission_log_path=log_dir / "provider_api.csv",
        output_dir=output_dir,
        enriched_dir=enriched_dir,
    )

    milvus_is_indexed = build_milvus_checker(milvus_uri, embed_model)
    session = make_session()
    stats = initial_stats()
    per_work: Dict[str, Any] = {}
    stats_path = output_dir / "stats.json"

    json_files = list(iter_json_files(input_dir))
    log(f"[Start] {len(json_files)} JSON file(s) in {input_dir}")
    log(f"[Out]   PDFs:      {config.pdf_dir}{' (shared)' if pdf_dir is not None else ''}")
    log(f"[Out]   Abstracts: {config.abstract_dir}{' (shared)' if abstract_dir is not None else ''}")
    log(f"[Out]   Logs:      {config.transmission_log_path}")
    log(f"[Out]   Enriched:  {enriched_dir}")

    for work_index, work_path in enumerate(json_files, start=1):
        log("")
        log(f"[Work {work_index}/{len(json_files)}] {work_path.name}")

        try:
            work = json.loads(work_path.read_text(encoding="utf-8"))
        except Exception as err:
            log(f"  [Error] cannot read JSON: {err!r}")
            continue

        links: List[Dict[str, Any]] = work.get("links") or []
        if not isinstance(links, list):
            links = []
        if config.max_refs_per_work > 0:
            links = links[:config.max_refs_per_work]

        stats["works"] += 1
        stats["references"] += len(links)

        work_stats: Dict[str, Any] = {
            "references": len(links), "doi_found": 0, "no_doi": 0,
            "pdf_ok": 0, "pdf_fail": 0, "abstract_ok": 0, "abstract_fail": 0,
        }
        log(f"  [Work] {len(links)} reference(s)")

        for ref_index, link in enumerate(links, start=1):
            if not isinstance(link, dict):
                continue
            try:
                process_reference(
                    session, config, link,
                    ref_index, len(links),
                    stats, work_stats,
                    milvus_is_indexed,
                )
            except Exception as err:
                log(f"  [Error] ref {ref_index} raised: {err!r}")

        per_work[work_path.name] = work_stats
        write_json_atomic(enriched_dir / work_path.name, work)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({"totals": stats, "per_work": per_work}, f, ensure_ascii=False, indent=2)
        log(f"  [Work done] {work_stats}")
        log(f"  [Saved]     {enriched_dir / work_path.name}")

    log("")
    log("[All done]")
    log(f"Stats: {stats_path}")
    close_milvus_checker = getattr(milvus_is_indexed, "close", None)
    if callable(close_milvus_checker):
        close_milvus_checker()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 1: Resolve DOIs and harvest reference PDFs / abstracts.",
    )
    ap.add_argument("--input",            required=True, help="Folder with work JSON files")
    ap.add_argument("--output",           required=True, help="Output folder")
    ap.add_argument("--enriched-output",  default="",    help="Where to write enriched JSONs")
    ap.add_argument("--crossref-mailto",  default=os.getenv("CROSSREF_MAILTO", "").strip(), help="CrossRef polite-pool email (or env CROSSREF_MAILTO)")
    ap.add_argument("--connect-timeout",  type=int,   default=8,   help="TCP connect timeout in seconds")
    ap.add_argument("--timeout",          type=int,   default=60,  help="Read timeout in seconds")
    ap.add_argument("--sleep",            type=float, default=0.0, help="Delay between network calls in seconds")
    ap.add_argument("--overwrite",        action="store_true",     help="Overwrite existing PDFs/abstracts and reset download fields")
    ap.add_argument("--max-refs-per-work",type=int,   default=0,   help="Limit references per work file - 0 = unlimited")
    ap.add_argument("--crossref-min-score", type=float, default=0.25, help="Minimum CrossRef candidate score to accept a DOI match")
    ap.add_argument("--milvus-uri",       default="", help="Milvus URI — when set with --embed-model, indexed references are skipped")
    ap.add_argument("--embed-model",      default="", help="Embedding model slug — must match the model used in verification/cli.py")
    args = ap.parse_args()

    if not args.crossref_mailto:
        raise SystemExit("Missing CROSSREF_MAILTO (env var or --crossref-mailto).")

    enriched_dir = (
        Path(args.enriched_output).expanduser().resolve()
        if args.enriched_output else None
    )

    return run(
        input_dir=Path(args.input).expanduser().resolve(),
        output_dir=Path(args.output).expanduser().resolve(),
        enriched_dir=enriched_dir,
        crossref_mailto=args.crossref_mailto,
        connect_timeout=args.connect_timeout,
        read_timeout=args.timeout,
        sleep_between_calls=args.sleep,
        overwrite=args.overwrite,
        max_refs_per_work=args.max_refs_per_work,
        crossref_min_score=args.crossref_min_score,
        milvus_uri=args.milvus_uri,
        embed_model=args.embed_model,
    )
