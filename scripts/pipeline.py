#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

DASHES = "-–—"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

# Fallback for running without `pip install -e .`
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from harvest.cli import run as harvest_run


def _read_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items


def _as_list_of_dicts(obj: Any, prefer_keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in prefer_keys:
            if k in obj and isinstance(obj[k], list):
                return [x for x in obj[k] if isinstance(x, dict)]
        return [obj]
    return []


def _normalize_sentences(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        return out
    return []


def _expand_range(a: int, b: int) -> List[int]:
    return list(range(a, b + 1)) if a <= b else list(range(a, b - 1, -1))


def _parse_citation_indices(label: Any) -> List[int]:
    if isinstance(label, int):
        return [label]
    if not isinstance(label, str):
        return []
    s = label.strip()
    tokens = re.findall(rf"\d+\s*(?:[{DASHES}]\s*\d+)?", s)
    out: List[int] = []
    for t in tokens:
        if any(d in t for d in DASHES):
            a, b = [p.strip() for p in re.split(rf"[{DASHES}]", t, 1)]
            if a.isdigit() and b.isdigit():
                out.extend(_expand_range(int(a), int(b)))
        else:
            if t.isdigit():
                out.append(int(t))
    return out


def _dedupe_preserve_order(xs: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def merge_refs_and_cites(
    refs_obj: Any,
    cites_obj: Any,
) -> Dict[str, Any]:
    refs_list = _as_list_of_dicts(refs_obj, prefer_keys=("refs", "references"))
    refs_by_idx: Dict[int, Dict[str, Any]] = {}
    for r in refs_list:
        idx = r.get("index")
        if isinstance(idx, int):
            refs_by_idx[idx] = r

    cites_list = _as_list_of_dicts(cites_obj, prefer_keys=("cites", "citations", "items"))

    sentences_for_ref: Dict[int, List[str]] = {}
    mention_count: Dict[int, int] = {}
    unmatched_citations: List[Dict[str, Any]] = []
    total_citations = 0

    for c in cites_list:
        label = c.get("citation", c.get("label", c.get("ref", c.get("index"))))
        idxs = _parse_citation_indices(label)
        if not idxs and isinstance(c.get("index"), int):
            idxs = [c["index"]]

        sents = _normalize_sentences(c.get("sentences"))
        total_citations += len(sents)

        if not idxs:
            unmatched_citations.append({"label": label, "indices": [], "sentences": sents})
            continue

        for idx in idxs:
            if idx in refs_by_idx:
                sentences_for_ref.setdefault(idx, []).extend(sents)
                mention_count[idx] = mention_count.get(idx, 0) + len(sents)
            else:
                unmatched_citations.append({"label": label, "indices": [idx], "sentences": sents})

    links: List[Dict[str, Any]] = []
    for idx in sorted(refs_by_idx.keys()):
        ref = dict(refs_by_idx[idx])
        if "index" in ref:
            del ref["index"]
        deduped_sents = _dedupe_preserve_order(sentences_for_ref.get(idx, []))
        links.append({
            "index": idx,
            "reference": ref,
            "sentences": deduped_sents,
            "occurrences": mention_count.get(idx, 0),
        })

    cited_count = sum(1 for l in links if l["occurrences"] > 0)
    return {
        "summary": {
            "total_references": len(links),
            "references_cited": cited_count,
            "total_citations": total_citations,
            "unmatched_citation_entries": len(unmatched_citations),
        },
        "links": links,
        "unmatched_citations": unmatched_citations,
    }


def _run_command(args: List[str]) -> None:
    # Run subprocesses with SRC_DIR on PYTHONPATH and UTF-8 as the default encoding.
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + pp if pp else "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"

    proc = subprocess.Popen(
        args,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout so ordering is preserved
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Subprocess failed with code {rc}: {' '.join(args)}")


def run_detect(pdf_path: str, out_json: str, detect_type: str) -> None:
    cmd = [sys.executable, "-m", detect_type, pdf_path, "--out", out_json]
    _run_command(cmd)


def build_work_json_from_pdf(pdf_path: Path, out_json_path: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        refs_path = os.path.join(td, "refs.json")
        cites_path = os.path.join(td, "cites.json")

        run_detect(str(pdf_path), refs_path, "refdetect.cli")
        run_detect(str(pdf_path), cites_path, "citedetect.cli")

        refs_obj = _read_json_any(refs_path)
        cites_obj = _read_json_any(cites_path)

    merged = merge_refs_and_cites(refs_obj, cites_obj)
    merged["source_pdf"] = str(pdf_path)

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def iter_pdfs(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


def main() -> int:
    ap = argparse.ArgumentParser()

    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--pdf", help="Single scholarly-work PDF (the one that contains the citations)")
    g_in.add_argument("--input-dir", help="Folder of scholarly-work PDFs to process")

    ap.add_argument("--output", required=True, help="Output root folder for everything")

    ap.add_argument("--crossref-mailto", default=os.getenv("CROSSREF_MAILTO", ""), help="Crossref mailto")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--connect-timeout", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--overwrite", action="store_true", help="Download already downloaded pdf again")
    ap.add_argument("--max-refs-per-work", type=int, default=0)
    ap.add_argument("--crossref-min-score", type=float, default=0.25, help="Minimum CrossRef candidate score to accept a DOI match")

    ap.add_argument("--grobid-url", default="http://localhost:8070", help="GROBID base URL")
    ap.add_argument("--milvus-uri", default="http://localhost:19530", help="Milvus connection URI")
    ap.add_argument("--embed-model", default="sentence-transformers/paraphrase-xlm-r-multilingual-v1")
    ap.add_argument("--device", default="", help="cuda / cpu / empty=auto")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--support-thresh", type=float, default=0.62)
    ap.add_argument("--related-thresh", type=float, default=0.42)

    ap.add_argument(
        "--shared-dir", default="",
        help=(
            "Optional shared directory for PDFs, abstracts, and TEI cache. "
        ),
    )

    ap.add_argument("--skip-detect", action="store_true", help="Skip detect stage")
    ap.add_argument("--skip-download", action="store_true", help="Skip harvest stage")
    ap.add_argument("--skip-verify", action="store_true", help="Skip verify stage")

    ap.add_argument("--protocol", action="store_true", help="Append protocol pages to the end of the input PDF(s)")
    ap.add_argument("--protocol-out-dir", default="", help="Where to write final PDFs (default: <output>/final_pdfs)")
    ap.add_argument("--protocol-max-matches", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser().resolve()
    work_json_dir = out_dir / "work_json"
    enriched_dir = out_dir / "enriched"
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_dir = Path(args.shared_dir).expanduser().resolve() if args.shared_dir else None
    if shared_dir is not None:
        shared_pdf_dir      = shared_dir / "pdfs"
        shared_abstract_dir = shared_dir / "abstracts"
        shared_tei_dir      = shared_dir / "cache" / "tei"
        for d in (shared_pdf_dir, shared_abstract_dir, shared_tei_dir):
            d.mkdir(parents=True, exist_ok=True)
        print(f"[shared] pdfs={shared_pdf_dir}")
        print(f"[shared] abstracts={shared_abstract_dir}")
        print(f"[shared] tei-cache={shared_tei_dir}")
    else:
        shared_pdf_dir = shared_abstract_dir = shared_tei_dir = None

    # ---- Stage 0 ----
    if not args.skip_detect:
        work_json_dir.mkdir(parents=True, exist_ok=True)

        if args.pdf:
            pdfs = [Path(args.pdf).expanduser().resolve()]
        else:
            pdfs = iter_pdfs(Path(args.input_dir).expanduser().resolve())
            if not pdfs:
                raise SystemExit(f"No PDFs found in {args.input_dir}")

        print(f"[stage0] processing {len(pdfs)} work PDF(s) -> {work_json_dir}")
        for i, pdf in enumerate(pdfs, start=1):
            out_json_path = work_json_dir / (pdf.stem + ".json")
            print(f"  [{i}/{len(pdfs)}] {pdf.name} -> {out_json_path.name}")
            build_work_json_from_pdf(pdf_path=pdf, out_json_path=out_json_path)
    else:
        if not work_json_dir.exists():
            raise SystemExit(f"--skip-detect set, but missing {work_json_dir}")

    # ---- Stage 1 ----
    if not args.skip_download:
        enriched_dir.mkdir(parents=True, exist_ok=True)
        print(f"[stage1] downloading/enriching refs -> {enriched_dir}")
        harvest_run(
            input_dir=work_json_dir,
            output_dir=out_dir,
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
            pdf_dir=shared_pdf_dir,
            abstract_dir=shared_abstract_dir,
        )
    else:
        if not enriched_dir.exists():
            raise SystemExit(f"--skip-download set, but missing {enriched_dir}")

    # ---- Stage 2 ----
    if not args.skip_verify:
        verifier_cmd = [
            sys.executable, "-m", "verification.cli",
            "--enriched", str(enriched_dir),
            "--output", str(out_dir),
            "--grobid-url", args.grobid_url,
            "--milvus-uri", args.milvus_uri,
            "--embed-model", args.embed_model,
            "--topk", str(args.topk),
            "--support-thresh", str(args.support_thresh),
            "--related-thresh", str(args.related_thresh),
        ]
        if args.device:
            verifier_cmd += ["--device", args.device]
        if shared_tei_dir is not None:
            verifier_cmd += ["--tei-cache-dir", str(shared_tei_dir)]

        print("[stage2] grobid segment + embed + verify citations")
        _run_command(verifier_cmd)

    if args.protocol:
        protocol_out = (
            Path(args.protocol_out_dir).expanduser().resolve()
            if args.protocol_out_dir
            else out_dir / "final_pdfs"
        )
        protocol_cmd = [
            sys.executable, str(SCRIPT_DIR / "append_protocol.py"),
            "--verified-dir", str(out_dir / "verified_json"),
            "--out-dir", str(protocol_out),
            "--max-matches", str(args.protocol_max_matches),
            "--support-thresh", str(args.support_thresh),
            "--related-thresh", str(args.related_thresh),
        ]
        print("[stage3] appending protocol pages to input PDFs")
        _run_command(protocol_cmd)

    print("\n[done]")
    print(f"  work_json:  {work_json_dir}")
    print(f"  enriched:   {enriched_dir}")
    print(f"  pdfs:       {out_dir / 'pdfs'}")
    print(f"  verified:   {out_dir / 'verified_json'}")
    print(f"  reports:    {out_dir / 'verification_reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
