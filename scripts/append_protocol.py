from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from pypdf import PdfReader, PdfWriter
import matplotlib.pyplot as plt
import pandas as pd

from commons.io import load_json
from analysis.analyze_similarity import extract_result_rows, plot_overall_similarity_histogram


def norm(s: str) -> str:
    return " ".join((s or "").split()).strip()


def short(s: str, n: int = 220) -> str:
    s = norm(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def build_protocol_pdf(
    verified: Dict[str, Any],
    out_pdf: Path,
    support_thresh: float,
    related_thresh: float,
    max_matches: int = 3,
) -> None:
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    title = styles["Title"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    story: List[Any] = []
    story.append(Paragraph("Citation Verification Protocol", title))
    story.append(Spacer(1, 6 * mm))

    source_pdf = verified.get("source_pdf", "")
    story.append(Paragraph(f"<b>Source PDF:</b> {short(str(source_pdf), 400)}", body))

    stats = paper_stats_from_verified(verified)
    plots_dir = out_pdf.parent / "_plots_tmp"
    plots_dir.mkdir(parents=True, exist_ok=True)
    p1 = plots_dir / "occurrence_hist.png"
    p2 = plots_dir / "overall_similarity_histogram.png"
    plot_occurrence_hist(stats, p1)
    rows = extract_result_rows(verified, "", out_pdf.stem)
    df_sims = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["best_sim"])
    if not df_sims.empty:
        plot_overall_similarity_histogram(df_sims, plots_dir, dpi=200, show=False,
                                          support_thresh=support_thresh,
                                          related_thresh=related_thresh)

    story.append(Paragraph("Paper statistics", h2))
    story.append(Spacer(1, 2 * mm))

    summary = verified.get("summary") or {}
    unmatched = verified.get("unmatched_citations") or []
    if not isinstance(unmatched, list):
        unmatched = []

    total_refs = int(summary.get("total_references", 0) or 0)
    cited_refs = int(summary.get("references_cited", 0) or 0)
    total_citations = int(summary.get("total_citations", 0) or 0)
    unused_refs = max(total_refs - cited_refs, 0)
    unmatched_entries = int(summary.get("unmatched_citation_entries", 0) or 0)
    unmatched_labels = len(unmatched)

    data = [
        ["Total references", str(total_refs)],
        ["Cited references", str(cited_refs)],
        ["Unused references", str(unused_refs)],
        ["Total citations (in-text)", str(total_citations)],
        ["Unmatched citation entries", str(unmatched_entries)],
        ["Unmatched citation labels", str(unmatched_labels)],
    ]
    t = Table(data, colWidths=[60 * mm, 110 * mm])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    if p1.exists():
        story.append(Image(str(p1), width=170 * mm, height=85 * mm))

    story.append(PageBreak())

    links = verified.get("links") or []
    total_links = len(links) if isinstance(links, list) else 0

    label_counts = {"supported": 0, "related": 0, "no_support": 0}
    total_checked_sents = 0

    if isinstance(links, list):
        for link in links:
            ver = (link or {}).get("verification") or {}
            if ver.get("status") == "ok":
                for r in (ver.get("results") or []):
                    lab = r.get("label")
                    if lab in label_counts:
                        label_counts[lab] += 1
                    total_checked_sents += 1

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Summary", h2))
    story.append(Paragraph(f"<b>References (links):</b> {total_links}", body))
    story.append(Paragraph(f"<b>Verified citing sentences:</b> {total_checked_sents}", body))
    story.append(Paragraph(
        f"<b>Labels:</b> supported={label_counts['supported']}, related={label_counts['related']}, no_support={label_counts['no_support']}",
        body
    ))
    if p2.exists():
        story.append(Spacer(1, 3 * mm))
        story.append(Image(str(p2), width=170 * mm, height=85 * mm))
    story.append(PageBreak())

    if not isinstance(links, list) or not links:
        story.append(Paragraph("No links found in verified JSON.", body))
        doc.build(story)
        return

    for li, link in enumerate(links, start=1):
        ref = (link or {}).get("reference") or {}
        ver = (link or {}).get("verification") or {}

        idx = link.get("index", li)
        occ = link.get("occurrences", 0)

        doi = ref.get("resolved_doi") or ref.get("doi") or ""
        raw = ref.get("raw") or ""
        dl = ref.get("download") or {}
        dl_status = dl.get("status") or "unknown"
        pdf_path = dl.get("pdf_path") or ""

        story.append(Paragraph(f"Reference [{idx}]", h2))
        story.append(Paragraph(f"<b>Occurrences in work:</b> {occ}", body))
        if doi:
            story.append(Paragraph(f"<b>Resolved DOI:</b> {short(str(doi), 250)}", body))
        crossref_title = ref.get("crossref_title") or ""
        if crossref_title:
            story.append(Paragraph(f"<b>Resolved title:</b> {short(crossref_title, 400)}", body))
        story.append(Paragraph(f"<b>Download status:</b> {dl_status}", body))
        if pdf_path:
            story.append(Paragraph(f"<b>Downloaded PDF:</b> {short(str(pdf_path), 300)}", body))
        if raw:
            story.append(Paragraph(f"<b>Raw reference:</b> {short(str(raw), 500)}", body))

        story.append(Spacer(1, 3 * mm))

        if ver.get("status") != "ok":
            story.append(Paragraph(f"<b>Verification status:</b> {ver.get('status')}", body))
            if ver.get("error"):
                story.append(Paragraph(f"<b>Error:</b> {short(str(ver.get('error')), 500)}", body))
            story.append(PageBreak())
            continue

        results = ver.get("results") or []
        if not results:
            story.append(Paragraph("No citing sentences found for this reference.", body))
            story.append(PageBreak())
            continue

        story.append(Paragraph("Citing sentences and best evidence", h3))
        story.append(Spacer(1, 2 * mm))

        for ri, r in enumerate(results, start=1):
            citing = r.get("citing_sentence") or ""
            label = r.get("label") or "unknown"
            best_sim = r.get("best_sim", 0.0)
            source_hint = r.get("source_hint") or ""
            best_zone = r.get("best_zone") or ""

            story.append(Paragraph(f"<b>{ri}. Label:</b> {label} &nbsp;&nbsp; <b>sim:</b> {best_sim:.3f} "
                                   f"&nbsp;&nbsp; <b>hint:</b> {source_hint} &nbsp;&nbsp; <b>zone:</b> {best_zone}", body))
            story.append(Paragraph(f"<b>Citing:</b> {short(citing, 700)}", body))

            top = (r.get("top_matches") or [])[:max_matches]
            if top:
                for mi, m in enumerate(top, start=1):
                    m_zone = m.get("zone", "")
                    m_level = m.get("level", "")
                    m_sec = m.get("section_path", "")
                    m_sim = m.get("sim", 0.0)
                    m_txt = m.get("text", "")

                    story.append(Paragraph(
                        f"&nbsp;&nbsp;• <b>Match {mi}</b> (sim {m_sim:.3f}) "
                        f"[{m_zone}/{m_level}] {short(m_sec, 180)}: {short(m_txt, 450)}",
                        body
                    ))

            story.append(Spacer(1, 3 * mm))

        story.append(PageBreak())

    doc.build(story)


def paper_stats_from_verified(verified: Dict[str, Any]) -> Dict[str, Any]:
    links = verified.get("links") or []
    if not isinstance(links, list):
        links = []

    summary = verified.get("summary") or {}
    total_refs = int(summary.get("total_references", 0) or 0)
    cited_refs = int(summary.get("references_cited", 0) or 0)
    total_citations = int(summary.get("total_citations", 0) or 0)
    unused_refs = max(total_refs - cited_refs, 0)

    occs = [int((l or {}).get("occurrences", 0) or 0) for l in links]
    occs_pos = [o for o in occs if o > 0]

    return {
        "total_refs": total_refs,
        "cited_refs": cited_refs,
        "unused_refs": unused_refs,
        "total_citations": total_citations,
        "occurrences_list": occs_pos,
    }


def plot_occurrence_hist(stats: Dict[str, Any], out_png: Path) -> None:
    occs = stats["occurrences_list"]
    if not occs:
        # Still create a placeholder so the report layout is stable.
        plt.figure(figsize=(6, 3.2))
        plt.text(0.5, 0.5, "No citation occurrence data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        return

    counts = Counter(occs)
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]

    plt.figure(figsize=(6, 3.2))
    plt.bar(xs, ys)
    plt.xlabel("Times a reference is cited")
    plt.ylabel("Number of references")
    plt.title("Citation occurrence distribution")
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def append_pdf(base_pdf: Path, add_pdf: Path, out_pdf: Path) -> None:
    reader_base = PdfReader(str(base_pdf))
    reader_add = PdfReader(str(add_pdf))
    writer = PdfWriter()

    for p in reader_base.pages:
        writer.add_page(p)
    for p in reader_add.pages:
        writer.add_page(p)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdf, "wb") as f:
        writer.write(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Append a protocol to the end of the input PDF.")
    ap.add_argument("--verified-json", help="Single verified JSON path")
    ap.add_argument("--verified-dir", help="Directory containing verified JSONs")
    ap.add_argument("--out-dir", required=True, help="Output directory for final PDFs")
    ap.add_argument("--max-matches", type=int, default=3, help="How many top matches to print per citing sentence")
    ap.add_argument("--support-thresh", type=float, default=0.62, help="Similarity threshold for 'supported' label")
    ap.add_argument("--related-thresh", type=float, default=0.42, help="Similarity threshold for 'related' label")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths: List[Path] = []
    if args.verified_json:
        json_paths = [Path(args.verified_json).expanduser().resolve()]
    elif args.verified_dir:
        d = Path(args.verified_dir).expanduser().resolve()
        json_paths = sorted([p for p in d.glob("*.json") if p.is_file()])
    else:
        raise SystemExit("Provide --verified-json or --verified-dir")

    for jp in json_paths:
        verified = load_json(jp)
        base_pdf = Path(str(verified.get("source_pdf") or "")).expanduser()
        if not base_pdf.exists():
            print(f"[protocol] SKIP {jp.name}: source_pdf missing or not found: {base_pdf}")
            continue

        tmp_protocol = out_dir / f"{jp.stem}.protocol_only.pdf"
        final_pdf = out_dir / f"{jp.stem}.protocol.pdf"

        build_protocol_pdf(verified, tmp_protocol, max_matches=args.max_matches,
                           support_thresh=args.support_thresh, related_thresh=args.related_thresh)
        append_pdf(base_pdf, tmp_protocol, final_pdf)

        try:
            tmp_protocol.unlink(missing_ok=True)
        except Exception:
            pass

        print(f"[protocol] wrote {final_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
