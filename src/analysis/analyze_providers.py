import argparse
import csv
import json
import math
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
# (only needed for undownloaded-publisher analysis)
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
]

# Sorted by prefix length descending so longest match wins
_DOI_PREFIX_MAP.sort(key=lambda t: len(t[0]), reverse=True)


def _publisher_from_doi(doi: str | None) -> str:
    if not doi:
        return "Unknown (no DOI)"
    doi_lower = doi.lower()
    for prefix, name in _DOI_PREFIX_MAP:
        if doi_lower.startswith(prefix):
            return name
    # Fall back to the raw prefix (e.g. "10.XXXX")
    parts = doi_lower.split("/", 1)
    return parts[0] if parts else "Unknown"


# ---------------------------------------------------------------------------
# Category mapping: (status, provider) → display label
# ---------------------------------------------------------------------------

_SUCCESS_STATUSES = {"downloaded", "already_indexed"}

_CATEGORY_COLORS: dict[str, str] = {
    "Springer (downloaded)": "#1a7fc1",
    "IEEE (downloaded)": "#005b96",
    "Elsevier (downloaded)": "#0e4d92",
    "arXiv (downloaded)": "#2494c7",
    "HTTP (downloaded)": "#56b4e9",
    "Abstract only": "#98df8a",
    "Download failed": "#d62728",
    "Not open access": "#ff7f0e",
    "Invalid/missing DOI": "#ffbb78",
    "Work not found": "#9467bd",
    "No DOI": "#c5b0d5",
    "Other": "#7f7f7f",
}

_CATEGORY_ORDER = list(_CATEGORY_COLORS.keys())


def _ref_category(status: str, provider: str | None) -> str:
    if status in _SUCCESS_STATUSES:
        p = (provider or "").lower()
        if p == "springer":
            return "Springer (downloaded)"
        if p == "ieee":
            return "IEEE (downloaded)"
        if p == "elsevier":
            return "Elsevier (downloaded)"
        if p == "arxiv":
            return "arXiv (downloaded)"
        return "HTTP (downloaded)"
    mapping = {
        "abstract_saved": "Abstract only",
        "download_failed": "Download failed",
        "not_open_access": "Not open access",
        "no_valid_doi": "Invalid/missing DOI",
        "work_not_found": "Work not found",
        "no_doi": "No DOI",
        "empty_reference_index": "Other",
        "other": "Other",
    }
    return mapping.get(status, "Other")


# ---------------------------------------------------------------------------
# URL / domain extraction
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r'https?://[^\s\])"\'<>]+')


def _extract_domain(raw: str) -> str | None:
    m = _URL_RE.search(raw)
    if not m:
        return None
    netloc = urlparse(m.group()).netloc
    return netloc.removeprefix("www.") or None


# ---------------------------------------------------------------------------
# File discovery & loading (mirrors analyze_refs.py)
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
# Data extraction
# ---------------------------------------------------------------------------

def extract_provider_counts(data: dict) -> Counter:
    counts: Counter = Counter()
    for link in data.get("links", []):
        ref = link.get("reference", {})
        dl = ref.get("download", {}) or {}
        status = dl.get("status", "other")
        provider = dl.get("provider")
        cat = _ref_category(status, provider)
        counts[cat] += 1
    return counts


def collect_no_doi_domains(data: dict) -> list[str | None]:
    domains = []
    for link in data.get("links", []):
        ref = link.get("reference", {})
        dl = ref.get("download", {}) or {}
        if dl.get("status") == "no_doi":
            raw = ref.get("raw", "")
            domains.append(_extract_domain(raw))
    return domains


def collect_undownloaded_dois(data: dict) -> list[str | None]:
    _FAILED_STATUSES = {"download_failed", "not_open_access", "work_not_found", "no_valid_doi"}
    dois = []
    for link in data.get("links", []):
        ref = link.get("reference", {})
        dl = ref.get("download", {}) or {}
        if dl.get("status") in _FAILED_STATUSES:
            doi = ref.get("resolved_doi") or ref.get("doi")
            dois.append(str(doi).strip() if doi else None)
    return dois


# ---------------------------------------------------------------------------
# Build per-university data
# ---------------------------------------------------------------------------

def build_data(university_files: dict[str, list[Path]]) -> tuple[
    dict[str, Counter],   # per-university category counts
    Counter,              # no-doi domain counts (all universities combined)
    dict[str, Counter],   # per-university no-doi domain counts
    list[str | None],     # undownloaded DOIs (all universities)
    dict[str, int],       # total ref count per university
]:
    univ_counts: dict[str, Counter] = {}
    all_domains: Counter = Counter()
    univ_domains: dict[str, Counter] = {}
    all_undownloaded_dois: list[str | None] = []
    univ_totals: dict[str, int] = {}

    for univ, paths in university_files.items():
        uc: Counter = Counter()
        ud: Counter = Counter()
        for p in paths:
            data = load_verified_json(p)
            if data is None:
                continue
            uc.update(extract_provider_counts(data))
            domains = collect_no_doi_domains(data)
            for d in domains:
                key = d if d else "__no_url__"
                ud[key] += 1
                all_domains[key] += 1
            all_undownloaded_dois.extend(collect_undownloaded_dois(data))

        univ_counts[univ] = uc
        univ_domains[univ] = ud
        univ_totals[univ] = sum(uc.values())
        print(f"  [{univ}] {sum(uc.values())} refs, "
              f"{sum(ud.values())} no-doi refs loaded")

    return univ_counts, all_domains, univ_domains, all_undownloaded_dois, univ_totals


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


# ---------------------------------------------------------------------------
# Chart 1: Provider share pie charts (one per university, combined figure)
# ---------------------------------------------------------------------------

def plot_provider_share_pies(
    univ_counts: dict[str, Counter],
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    univs = sorted(univ_counts.keys())
    n = len(univs)
    if n == 0:
        return

    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(fig_width, fig_width * nrows / ncols * 0.9 + 1.5))
    axes_flat: list[plt.Axes] = np.array(axes).flatten().tolist() if n > 1 else [axes]

    present_cats = [c for c in _CATEGORY_ORDER
                    if any(univ_counts[u].get(c, 0) > 0 for u in univs)]
    colors = [_CATEGORY_COLORS[c] for c in present_cats]

    legend_patches = [
        plt.matplotlib.patches.Patch(color=_CATEGORY_COLORS[c], label=c)
        for c in present_cats
    ]

    for i, univ in enumerate(univs):
        ax = axes_flat[i]
        counts = univ_counts[univ]
        total = sum(counts.values())
        sizes = [counts.get(c, 0) for c in present_cats]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
            startangle=90,
            pctdistance=0.75,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(7)

        ax.set_title(univ, fontsize=10, pad=4)
        ax.text(0, -1.25, f"n={total:,}", ha="center", fontsize=8, color="#555555")

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.legend(handles=legend_patches, title="Category",
               loc="lower center", ncol=min(len(present_cats), 4),
               bbox_to_anchor=(0.5, 0), fontsize=8, title_fontsize=9)
    fig.suptitle("Reference Provider Share by University", fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    _save(fig, out_dir / "provider_share_pies.png", dpi, show)


# ---------------------------------------------------------------------------
# Chart 2a: No-DOI domain bar — aggregated
# ---------------------------------------------------------------------------

def plot_no_doi_domains_bar(
    all_domains: Counter,
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
    top_n: int = 20,
) -> None:
    if not all_domains:
        print("  [providers] No no-doi refs found, skipping domain bar.")
        return

    no_url_count = all_domains.pop("__no_url__", 0)
    top = all_domains.most_common(top_n)

    labels = [d for d, _ in top]
    values = [c for _, c in top]
    if no_url_count:
        labels.append("(no URL in text)")
        values.append(no_url_count)

    # Restore key for reuse
    all_domains["__no_url__"] = no_url_count

    n = len(labels)
    fig, ax = plt.subplots(figsize=(fig_width, max(3, n * 0.38 + 1.5)))
    palette = sns.color_palette("muted", n)
    bars = ax.barh(labels[::-1], values[::-1], color=palette[::-1], height=0.6)
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)
    ax.set_xlabel("Number of references")
    ax.set_title("No-DOI References: Domain Distribution (all universities)")
    ax.set_xlim(0, max(values) * 1.15)
    _save(fig, out_dir / "no_doi_domains_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# Chart 2b: No-DOI domain per university — stacked bar
# ---------------------------------------------------------------------------

def plot_no_doi_domains_per_univ(
    univ_domains: dict[str, Counter],
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
    top_n: int = 10,
) -> None:
    univs = sorted(univ_domains.keys())
    if len(univs) < 2:
        return  # not meaningful with a single university

    # Collect top-N domains across all universities combined (excluding __no_url__)
    combined: Counter = Counter()
    for ud in univ_domains.values():
        for k, v in ud.items():
            if k != "__no_url__":
                combined[k] += v

    top_domains = [d for d, _ in combined.most_common(top_n)]
    palette = sns.color_palette("tab10", len(top_domains) + 1)
    color_map = {d: palette[i] for i, d in enumerate(top_domains)}
    color_map["Other / no URL"] = palette[len(top_domains)]

    rows = []
    for univ in univs:
        ud = univ_domains[univ]
        total_no_doi = sum(ud.values())
        row: dict[str, float] = {}
        other = 0
        for d, c in ud.items():
            if d in top_domains:
                row[d] = c / total_no_doi * 100 if total_no_doi else 0.0
            else:
                other += c
        row["Other / no URL"] = other / total_no_doi * 100 if total_no_doi else 0.0
        rows.append(row)

    all_keys = top_domains + ["Other / no URL"]
    df = pd.DataFrame(rows, index=univs).fillna(0.0)[all_keys]

    n_unis = len(univs)
    fig, ax = plt.subplots(figsize=(fig_width, max(3, n_unis * 0.6 + 1.5)))
    left = np.zeros(n_unis)
    for key in all_keys:
        vals = df[key].values
        bars = ax.barh(univs, vals, left=left, color=color_map[key], label=key, height=0.6)
        for bar, val in zip(bars, vals):
            if val >= 8:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%", ha="center", va="center", fontsize=7, color="white")
        left += vals

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of no-DOI references")
    ax.set_title("No-DOI Reference Domains per University (top 10)")
    ax.legend(title="Domain", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.invert_yaxis()
    _save(fig, out_dir / "no_doi_domains_per_univ_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# Chart 3: Download funnel (overall aggregate)
# ---------------------------------------------------------------------------

def plot_download_funnel(
    univ_counts: dict[str, Counter],
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    combined: Counter = Counter()
    for c in univ_counts.values():
        combined.update(c)

    total = sum(combined.values())
    if total == 0:
        return

    has_doi = total - combined.get("No DOI", 0) - combined.get("Invalid/missing DOI", 0)
    downloaded = (combined.get("Springer (downloaded)", 0)
                  + combined.get("IEEE (downloaded)", 0)
                  + combined.get("Elsevier (downloaded)", 0)
                  + combined.get("arXiv (downloaded)", 0)
                  + combined.get("HTTP (downloaded)", 0))
    abstract_only = combined.get("Abstract only", 0)

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
    ax.set_title("Download Coverage Funnel (all universities)")
    ax.set_xlim(0, total * 1.35)
    _save(fig, out_dir / "download_funnel_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# CSV 1: providers_summary.csv
# ---------------------------------------------------------------------------

def write_providers_summary_csv(
    univ_counts: dict[str, Counter],
    out_dir: Path,
) -> None:
    rows = []
    for univ in sorted(univ_counts.keys()):
        c = univ_counts[univ]
        total = sum(c.values())
        row: dict = {"university": univ, "total_refs": total}
        for cat in _CATEGORY_ORDER:
            key = cat.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            row[key] = c.get(cat, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = out_dir / "providers_summary.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CSV 2: undownloaded_publishers.csv
# ---------------------------------------------------------------------------

def write_undownloaded_publishers_csv(
    all_undownloaded_dois: list[str | None],
    out_dir: Path,
) -> None:
    publisher_counts: Counter = Counter()
    for doi in all_undownloaded_dois:
        pub = _publisher_from_doi(doi)
        publisher_counts[pub] += 1

    total = sum(publisher_counts.values())
    rows = []
    for pub, count in publisher_counts.most_common():
        rows.append({
            "publisher": pub,
            "count": count,
            "pct_of_undownloaded": round(count / total * 100, 1) if total else 0.0,
        })

    path = out_dir / "undownloaded_publishers.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["publisher", "count", "pct_of_undownloaded"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")

    if rows:
        print("\n  Top undownloaded publishers (implement these APIs to increase coverage):")
        for r in rows[:10]:
            print(f"    {r['publisher']:<40} {r['count']:>5} refs  ({r['pct_of_undownloaded']:.1f}%)")


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
    (univ_counts, all_domains, univ_domains,
     all_undownloaded_dois, univ_totals) = build_data(university_files)

    if not any(univ_counts.values()):
        sys.exit("No valid documents loaded.")

    print(f"[analyze_providers] Generating outputs -> {out_dir}")
    plot_provider_share_pies(univ_counts, out_dir, args.fig_width, args.dpi, args.show)
    plot_no_doi_domains_bar(all_domains, out_dir, args.fig_width, args.dpi, args.show)
    plot_no_doi_domains_per_univ(univ_domains, out_dir, args.fig_width, args.dpi, args.show)
    plot_download_funnel(univ_counts, out_dir, args.fig_width, args.dpi, args.show)
    write_providers_summary_csv(univ_counts, out_dir)
    write_undownloaded_publishers_csv(all_undownloaded_dois, out_dir)

    print(f"[analyze_providers] Done.")


if __name__ == "__main__":
    main()
