import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# DOI prefix → publisher name table
# ---------------------------------------------------------------------------

_DOI_PREFIX_MAP: list[tuple[str, str]] = [
    ("10.48550/arxiv", "arXiv"),
    ("10.48550", "arXiv"),
    ("10.1016", "Elsevier"),
    ("10.1007", "Springer Nature"),
    ("10.1109", "IEEE"),
    ("10.1145", "ACM"),
    ("10.1080", "Taylor & Francis"),
    ("10.1017", "Cambridge University Press"),
    ("10.1093", "Oxford University Press"),
    ("10.1515", "De Gruyter"),
    ("10.3390", "MDPI"),
    ("10.1002", "Wiley"),
    ("10.1111", "Wiley"),
    ("10.1038", "Nature / Springer Nature"),
    ("10.1126", "AAAS / Science"),
    ("10.1103", "APS Physics"),
    ("10.1063", "AIP Publishing"),
    ("10.1021", "ACS Publications"),
    ("10.1039", "Royal Society of Chemistry"),
    ("10.1140", "Springer"),
    ("10.1186", "BioMed Central (Springer)"),
    ("10.1371", "PLOS"),
    ("10.3758", "Springer Psychonomic"),
    ("10.2307", "JSTOR"),
    ("10.1353", "Project MUSE"),
    ("10.4324", "Taylor & Francis (Routledge)"),
    ("10.5194", "Copernicus Publications"),
    ("10.3233", "IOS Press"),
    ("10.1162", "MIT Press"),
    ("10.1214", "Institute of Mathematical Statistics"),
    ("10.7717", "PeerJ"),
    ("10.3847", "AAS / IOP"),
    ("10.1088", "IOP Publishing"),
    ("10.1364", "Optica / OSA"),
    ("10.1049", "IET"),
    ("10.1504", "Inderscience"),
    ("10.1142", "World Scientific"),
    ("10.4230", "Schloss Dagstuhl (LIPIcs)"),
    ("10.18653", "ACL Anthology"),
    ("10.1201", "CRC Press / Taylor & Francis"),
    ("10.1163", "Brill"),
    ("10.3403", "BSI"),
    ("10.4135", "SAGE Publications"),
    ("10.17487", "IETF / RFC Editor"),
    ("10.2139", "SSRN (Elsevier)"),
    ("10.1108", "Emerald Publishing"),
]

_DOI_PREFIX_MAP.sort(key=lambda t: len(t[0]), reverse=True)


def _publisher_from_doi(doi: str | None) -> str:
    if not doi:
        return "Unknown (no DOI)"
    doi_lower = doi.lower()
    for prefix, name in _DOI_PREFIX_MAP:
        if doi_lower.startswith(prefix):
            return name
    parts = doi_lower.split("/", 1)
    return parts[0] if parts else "Unknown"


# ---------------------------------------------------------------------------
# Status sets
# ---------------------------------------------------------------------------

_SUCCESS_STATUSES = {"downloaded", "already_indexed"}
_NO_DOI_STATUSES = {"no_doi", "no_valid_doi"}

_BAR_BLUE = "#3a6ea5"


# ---------------------------------------------------------------------------
# URL / domain extraction
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r'https?://[^\s\])"\'<>]+')


_TLD_LIKE = {"co", "ac", "gov", "org", "net", "com", "edu"}


def _extract_domain(raw: str) -> str | None:
    m = _URL_RE.search(raw)
    if not m:
        return None

    netloc = urlparse(m.group()).netloc.removeprefix("www.")

    if not netloc:
        return None

    parts = [p for p in netloc.split(".") if p]
    if len(parts) == 1:
        return parts[0]
    #if its  co.uk, ac.uk, step back one more
    sld = parts[-2]
    if sld in _TLD_LIKE and len(parts) >= 3:
        sld = parts[-3]
    return sld


# ---------------------------------------------------------------------------
# File discovery & loading
# ---------------------------------------------------------------------------

def discover_university_files(in_dir: Path, faculty_name: str) -> dict[str, list[Path]]:
    subdirs = [d for d in sorted(in_dir.iterdir())
               if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"]
    if subdirs:
        result = {}
        for subdir in subdirs:
            files = sorted(subdir.glob("*.json"))
            if files:
                result[subdir.name] = files
        return result
    files = sorted(in_dir.glob("*.json"))
    label = faculty_name
    return {label: files} if files else {}


def load_verified_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Aggregate data collection
# ---------------------------------------------------------------------------

def build_data(university_files: dict[str, list[Path]]) -> tuple[
    Counter,              # status_counts
    Counter,              # all_domains
    int,                  # no_doi_without_url_count
    list[str | None],     # all_dois
]:
    status_counts: Counter = Counter()
    all_domains: Counter = Counter()
    no_doi_without_url_count = 0
    all_dois: list[str | None] = []

    for univ, paths in university_files.items():
        univ_total = 0
        univ_downloaded = 0
        univ_no_doi = 0
        for p in paths:
            data = load_verified_json(p)
            if data is None:
                continue
            for link in data.get("links", []):
                ref = link.get("reference", {})
                dl = ref.get("download", {}) or {}
                status = dl.get("status", "other")
                status_counts[status] += 1
                univ_total += 1
                if status in _SUCCESS_STATUSES:
                    univ_downloaded += 1
                raw = ref.get("raw", "") or ""
                if status in _NO_DOI_STATUSES:
                    univ_no_doi += 1
                    if not _URL_RE.search(raw):
                        no_doi_without_url_count += 1
                    domain = _extract_domain(raw)
                    if domain:
                        all_domains[domain] += 1
                doi = ref.get("resolved_doi") or ref.get("doi")
                if doi:
                    all_dois.append(str(doi).strip())
        print(f"  [{univ}] {univ_total} refs, "
              f"{univ_downloaded} downloaded, {univ_no_doi} no-DOI")

    return status_counts, all_domains, no_doi_without_url_count, all_dois


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def _save(fig: plt.Figure, path: Path, dpi: int, show: bool) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def _add_bar_footer(fig: plt.Figure, footnote: str) -> None:
    fig.text(0.5, 0.012, footnote, ha="center", fontsize=8, color="#555555")
    plt.tight_layout(rect=(0, 0.04, 1, 1))


# ---------------------------------------------------------------------------
# Chart: Undownloaded publisher bar
# ---------------------------------------------------------------------------

def plot_publishers_bar(
    all_dois: list[str | None],
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
    top_n: int = 15,
) -> None:
    pub_counts: Counter = Counter()
    for doi in all_dois:
        if doi:
            pub_counts[_publisher_from_doi(doi)] += 1

    if not pub_counts:
        print("  [providers] No refs with DOI, skipping publisher bar.")
        return

    total_with_doi = sum(pub_counts.values())
    top = pub_counts.most_common(top_n)
    remainder = len(pub_counts) - len(top)

    labels = [pub for pub, _ in top]
    values = [cnt for _, cnt in top]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(fig_width, max(3, n * 0.42 + 1.5)))
    bars = ax.barh(labels[::-1], values[::-1], color=_BAR_BLUE, height=0.65)
    for bar, val in zip(bars, values[::-1]):
        pct = val / total_with_doi * 100
        ax.text(val + total_with_doi * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)", va="center", fontsize=12)

    ax.set_xlabel("Number of references", fontsize=12)
    ax.set_title(f"Top {top_n} Publishers by DOI Count", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlim(0, max(values) * 1.3)

    footnote = f"{total_with_doi:,} refs with DOI"
    if remainder > 0:
        footnote += f"  |  {remainder} further publishers detected"
    _add_bar_footer(fig, footnote)
    _save(fig, out_dir / "publishers_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# Chart: No-DOI URL domain bar
# ---------------------------------------------------------------------------

def plot_no_doi_domains_bar(
    all_domains: Counter,
    no_doi_without_url_count: int,
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
    top_n: int = 15,
) -> None:
    if not all_domains:
        print("  [providers] No no-doi refs found, skipping domain bar.")
        return

    top = all_domains.most_common(top_n)
    remainder = len(all_domains) - len(top)
    labels = [d for d, _ in top]
    values = [c for _, c in top]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(fig_width, max(3, n * 0.38 + 1.5)))
    bars = ax.barh(labels[::-1], values[::-1], color=_BAR_BLUE, height=0.6)
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=12)
    ax.set_xlabel("Number of references", fontsize=12)
    ax.set_title("Domain Distribution of No-DOI References", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlim(0, max(values) * 1.15)

    footnote = f"{no_doi_without_url_count:,} refs did not contain any URL"
    if remainder > 0:
        footnote += f"  |  {remainder} additional domains detected"
    _add_bar_footer(fig, footnote)

    _save(fig, out_dir / "no_doi_domains_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# Chart: Download funnel
# ---------------------------------------------------------------------------

def plot_download_funnel(
    status_counts: Counter,
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    total = sum(status_counts.values())
    if total == 0:
        return

    has_doi = total - sum(status_counts.get(s, 0) for s in _NO_DOI_STATUSES)
    downloaded = sum(status_counts.get(s, 0) for s in _SUCCESS_STATUSES)
    abstract_only = status_counts.get("abstract_saved", 0)

    stages = ["Total references", "Has DOI", "Full text downloaded", "Abstract obtained"]
    values = [total, has_doi, downloaded, abstract_only]
    pcts = [f"{v:,} refs ({v / total * 100:.1f}%)" for v in values]

    fig, ax = plt.subplots(figsize=(fig_width * 0.6, 4))
    colors = ["#4c72b0", "#55a868", "#2ca02c", "#98df8a"]
    bars = ax.barh(stages[::-1], values[::-1], color=colors[::-1], height=0.5)
    for bar, val, pct in zip(bars, values[::-1], pcts[::-1]):
        ax.text(val + total * 0.01, bar.get_y() + bar.get_height() / 2,
                pct, va="center", fontsize=9)
    ax.set_xlabel("Number of references")
    ax.set_title("Download Coverage Funnel")
    ax.set_xlim(0, total * 1.35)
    _save(fig, out_dir / "download_funnel_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# CSV: undownloaded_publishers.csv
# ---------------------------------------------------------------------------

def write_publishers_csv(
    all_dois: list[str | None],
    out_dir: Path,
) -> None:
    publisher_counts: Counter = Counter()
    for doi in all_dois:
        if doi:
            publisher_counts[_publisher_from_doi(doi)] += 1

    total = sum(publisher_counts.values())
    rows = []
    for pub, count in publisher_counts.most_common():
        rows.append({
            "publisher": pub,
            "count": count,
            "pct_of_with_doi": round(count / total * 100, 1) if total else 0.0,
        })

    path = out_dir / "publishers.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["publisher", "count", "pct_of_with_doi"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")

    if rows:
        print("\n  Top publishers:")
        for r in rows[:10]:
            print(f"    {r['publisher']:<40} {r['count']:>5} refs  ({r['pct_of_with_doi']:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Provider share & no-DOI domain analysis")
    ap.add_argument("--in-dir", required=True, help="Directory containing verified JSON files")
    ap.add_argument("--out-dir", required=True, help="Output directory for plots and CSVs")
    ap.add_argument("--show", action="store_true", help="Call plt.show() after each figure")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--fig-width", type=float, default=12)
    ap.add_argument("--faculty-name", default="Faculty",
                    help="Label to use when in-dir has no subdirectories")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    in_dir = Path(args.in_dir)
    university_files = discover_university_files(in_dir, args.faculty_name)
    if not university_files:
        sys.exit(f"No JSON files found in {in_dir}")

    print(f"[analyze_providers] Loading data from {in_dir} ...")
    status_counts, all_domains, no_doi_without_url_count, all_dois = build_data(university_files)

    if sum(status_counts.values()) == 0:
        sys.exit("No valid documents loaded.")

    print(f"[analyze_providers] Generating outputs -> {out_dir}")
    plot_publishers_bar(all_dois, out_dir, args.fig_width, args.dpi, args.show)
    plot_no_doi_domains_bar(
        all_domains,
        no_doi_without_url_count,
        out_dir,
        args.fig_width,
        args.dpi,
        args.show,
    )
    plot_download_funnel(status_counts, out_dir, args.fig_width, args.dpi, args.show)
    write_publishers_csv(all_dois, out_dir)

    print(f"[analyze_providers] Done.")


if __name__ == "__main__":
    main()
