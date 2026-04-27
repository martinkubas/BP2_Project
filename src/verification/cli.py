from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

from .citation_verifier import verify_link
from .embedder import create_embedder
from .grobid_client import GrobidClient
from .milvus_store import MilvusSegmentStore
from .text_segmenter import SentenceSplitter, TEISegmenter
from commons.io import iter_json_files, load_json, write_json_atomic
from commons.text import slugify


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description="Verify citation contexts against downloaded reference documents via Milvus."
    )
    argument_parser.add_argument(
        "--enriched", required=True,
        help="Folder of enriched work JSONs (output of the harvest stage).",
    )
    argument_parser.add_argument(
        "--output", required=True,
        help="Output root folder (same --output used in earlier pipeline stages).",
    )
    argument_parser.add_argument("--grobid-url",     default="http://localhost:8070")
    argument_parser.add_argument("--milvus-uri",     default="http://localhost:19530",
                                 help="Milvus connection URI, e.g. http://localhost:19530")
    argument_parser.add_argument("--embed-model",    default="sentence-transformers/paraphrase-xlm-r-multilingual-v1")
    argument_parser.add_argument("--device",         default="",
                                 help="Embedding device: 'cuda', 'cpu', or '' for auto-detect.")
    argument_parser.add_argument("--topk",           type=int,   default=5)
    argument_parser.add_argument("--support-thresh", type=float, default=0.62,
                                 help="Minimum cosine similarity to label a sentence 'supported'.")
    argument_parser.add_argument("--related-thresh", type=float, default=0.42,
                                 help="Minimum cosine similarity to label a sentence 'related'.")
    args = argument_parser.parse_args()

    enriched_dir  = Path(args.enriched).expanduser().resolve()
    output_dir    = Path(args.output).expanduser().resolve()
    tei_cache_dir = output_dir / "cache" / "tei"
    verified_dir  = output_dir / "verified_json"
    reports_dir   = output_dir / "verification_reports"
    verified_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    grobid_client = GrobidClient(args.grobid_url)
    sentence_splitter = SentenceSplitter()
    tei_segmenter = TEISegmenter(sentence_splitter, tei_cache_dir)
    embedder = create_embedder(args.embed_model, device=args.device)

    print(
        f"[verify] model={args.embed_model}  "
        f"device={getattr(embedder, 'device', 'api')}  "
        f"dim={embedder.embedding_dim}"
    )

    milvus_store = MilvusSegmentStore(
        milvus_uri=args.milvus_uri,
        model_slug=embedder.model_slug,
        embedding_dim=embedder.embedding_dim,
    )

    run_summary: Dict[str, Any] = {
        "created_at":  utc_now_iso(),
        "embed_model": args.embed_model,
        "totals": {
            "works":          0,
            "links":          0,
            "verified_links": 0,
            "skipped_links":  0,
        },
        "labels": {"supported": 0, "related": 0, "no_support": 0},
    }

    work_json_files = list(iter_json_files(enriched_dir))
    work_count = len(work_json_files)
    print(f"[verify] {work_count} work file(s) in {enriched_dir}", flush=True)

    for work_index, work_json_path in enumerate(work_json_files, start=1):
        print(f"[Work {work_index}/{work_count}] {work_json_path.name}", flush=True)
        work_data = load_json(work_json_path)
        links: List[Dict[str, Any]] = work_data.get("links") or []
        if not isinstance(links, list):
            links = []

        run_summary["totals"]["works"] += 1
        run_summary["totals"]["links"] += len(links)

        for link in links:
            if not isinstance(link, dict):
                continue

            reference_data = link.get("reference") or {}
            download_info  = reference_data.get("download") or {}
            download_status = download_info.get("status")

            pdf_path:      Path | None = None
            abstract_path: Path | None = None
            source_type:   str  | None = None
            # When True, the downloader already confirmed this DOI is in Milvus.
            # We have no local file so indexing cannot be retried.
            vectors_guaranteed_in_milvus: bool = False

            if download_status == "already_indexed":
                vectors_guaranteed_in_milvus = True
                source_type = "pdf"

            elif download_status == "downloaded" and download_info.get("pdf_path"):
                candidate_path = (output_dir / Path(download_info["pdf_path"])).resolve()
                if not candidate_path.exists():
                    run_summary["totals"]["skipped_links"] += 1
                    link["verification"] = {
                        "status":   "pdf_missing",
                        "pdf_path": download_info["pdf_path"],
                    }
                    continue
                pdf_path = candidate_path
                source_type = "pdf"

            elif download_status == "abstract_saved" and download_info.get("abstract_path"):
                candidate_path = (output_dir / Path(download_info["abstract_path"])).resolve()
                if not candidate_path.exists():
                    run_summary["totals"]["skipped_links"] += 1
                    link["verification"] = {
                        "status":        "abstract_missing",
                        "abstract_path": download_info["abstract_path"],
                    }
                    continue
                abstract_path = candidate_path
                source_type = "abstract"

            else:
                run_summary["totals"]["skipped_links"] += 1
                link["verification"] = {
                    "status":          "no_source_text",
                    "download_status": download_status,
                }
                continue

            raw_ref_key = (
                reference_data.get("resolved_doi")
                or reference_data.get("doi")
                or (pdf_path.stem if pdf_path else (abstract_path.stem if abstract_path else ""))
            )
            source_key = slugify(str(raw_ref_key))

            if vectors_guaranteed_in_milvus:
                # Safety check: Milvus could have been wiped since Stage 1 ran.
                if not milvus_store.has_source(source_key):
                    run_summary["totals"]["skipped_links"] += 1
                    link["verification"] = {
                        "status":  "milvus_data_missing",
                        "ref_key": str(raw_ref_key),
                        "hint":    "Milvus may have been reset. Re-run Stage 1 without --milvus-uri to re-download.",
                    }
                    continue
            elif not milvus_store.has_source(source_key):
                try:
                    if pdf_path is not None:
                        tei_xml = tei_segmenter.get_or_process_tei(
                            source_key_slug=source_key,
                            pdf_path=pdf_path,
                            grobid_client=grobid_client,
                        )
                        segments = tei_segmenter.segments_from_tei(tei_xml)
                    else:
                        abstract_text = read_text_file(abstract_path)
                        segments = tei_segmenter.segments_from_abstract_text(abstract_text)

                    if not segments:
                        print(f"[verify] warning: no segments extracted for {source_key}")
                        run_summary["totals"]["skipped_links"] += 1
                        link["verification"] = {"status": "empty_reference_index", "ref_key": str(raw_ref_key)}
                        continue

                    segment_texts = [segment.text for segment in segments]
                    embedding_matrix = embedder.encode(segment_texts)
                    milvus_store.store_segments(source_key, source_type, segments, embedding_matrix)

                except Exception as indexing_error:
                    run_summary["totals"]["skipped_links"] += 1
                    link["verification"] = {
                        "status": "ref_index_failed",
                        "error":  repr(indexing_error),
                    }
                    continue

            verification_result = verify_link(
                link=link,
                source_key=source_key,
                milvus_store=milvus_store,
                embedder=embedder,
                top_k=args.topk,
                support_threshold=args.support_thresh,
                related_threshold=args.related_thresh,
            )

            verification_result["embed_model"]  = args.embed_model
            verification_result["verified_at"]  = utc_now_iso()
            verification_result["ref_key"]      = str(raw_ref_key)
            verification_result["source_type"]  = source_type
            if pdf_path is not None:
                verification_result["pdf_path"]      = download_info["pdf_path"]
            if abstract_path is not None:
                verification_result["abstract_path"] = download_info["abstract_path"]

            link["verification"] = verification_result

            if verification_result.get("status") == "ok":
                run_summary["totals"]["verified_links"] += 1
                for sentence_result in verification_result["results"]:
                    label = sentence_result.get("label")
                    if label in run_summary["labels"]:
                        run_summary["labels"][label] += 1
            else:
                run_summary["totals"]["skipped_links"] += 1

        write_json_atomic(verified_dir / work_json_path.name, work_data)

        # Write a lightweight per-work report for quick inspection.
        per_work_report = {
            "work_file":   work_json_path.name,
            "embed_model": args.embed_model,
            "verified_at": utc_now_iso(),
            "links": [
                {
                    "index":        link.get("index"),
                    "resolved_doi": (link.get("reference") or {}).get("resolved_doi"),
                    "status":       (link.get("verification") or {}).get("status"),
                    "results":      (link.get("verification") or {}).get("results", [])[:50],
                }
                for link in links
                if isinstance(link, dict)
            ],
        }
        report_path = reports_dir / (work_json_path.stem + ".report.json")
        report_path.write_text(
            json.dumps(per_work_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (reports_dir / "summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    milvus_store.close()
    print(f"[verify] done.  verified_json={verified_dir}  reports={reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
