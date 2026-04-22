import argparse
import json
import sys
import math
from collections import Counter
from pathlib import Path

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DOWNLOAD_STATUS_ORDER = [
    "downloaded",
    "abstract_saved",
    "already_indexed",
    "download_failed",
    "not_open_access",
    "no_valid_doi",
    "no_doi",
    "work_not_found",
    "empty_reference_index",
    "other",
]

REUSE_LABELS = ["1×", "2×", "3×", "4×+"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Reference & citation statistics")
    ap.add_argument("--in-dir", required=True, help="Directory containing verified JSON files")
    ap.add_argument("--out-dir", required=True, help="Output directory for plots and CSV")
    ap.add_argument("--show", action="store_true", help="Call plt.show() after each figure")
    ap.add_argument("--dpi", type=int, default=150, help="Output resolution (default: 150)")
    ap.add_argument("--fig-width", type=float, default=12, help="Base figure width in inches (default: 12)")
    ap.add_argument("--faculty-name", default="", help="Label to use instead of 'default' when in-dir has no subdirectories")
    return ap.parse_args()


def discover_university_files(in_dir: Path, faculty_name: str) -> dict[str, list[Path]]:
    """Return {university_name: [json_paths]} grouping.

    If in_dir has subdirectories, each subdir is a university.
    Otherwise, all *.json files in in_dir are grouped under faculty_name.
    """
    subdirs = [d for d in sorted(in_dir.iterdir())
               if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"]
    if subdirs:
        result = {}
        for subdir in subdirs:
            files = sorted(subdir.glob("*.json"))
            if files:
                result[subdir.name] = files
        return result
    else:
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


def extract_doc_metrics(data: dict) -> dict:
    summary = data.get("summary", {})
    total_refs = summary.get("total_references", 0)
    refs_cited = summary.get("references_cited", 0)
    total_citations = summary.get("total_citations", 0)

    links = data.get("links", [])

    # Reuse buckets (by occurrences)
    reuse = {"1x": 0, "2x": 0, "3x": 0, "4p": 0}
    # Download statuses
    dl_status = Counter()
    # DOI coverage
    doi_count = 0

    for link in links:
        occ = link.get("occurrences", 0)
        if occ > 0:
            if occ == 1:
                reuse["1x"] += 1
            elif occ == 2:
                reuse["2x"] += 1
            elif occ == 3:
                reuse["3x"] += 1
            else:
                reuse["4p"] += 1

        ref = link.get("reference", {})
        dl = ref.get("download", {})
        status = dl.get("status", "other") if dl else "other"
        if status not in DOWNLOAD_STATUS_ORDER[:-1]:
            status = "other"
        dl_status[status] += 1

        doi = ref.get("doi") or ref.get("resolved_doi")
        if doi and str(doi).strip():
            doi_count += 1

    unused_ratio = ((total_refs - refs_cited) / total_refs * 100) if total_refs > 0 else 0.0
    doi_coverage = (doi_count / len(links) * 100) if links else 0.0

    return {
        "total_refs": total_refs,
        "refs_cited": refs_cited,
        "total_citations": total_citations,
        "unused_ratio": unused_ratio,
        "doi_coverage": doi_coverage,
        "reuse": reuse,
        "dl_status": dl_status,
    }


def build_university_frames(university_files: dict[str, list[Path]]) -> dict[str, list[dict]]:
    result = {}
    for univ, paths in university_files.items():
        metrics = []
        for p in paths:
            data = load_verified_json(p)
            if data is not None:
                metrics.append(extract_doc_metrics(data))
        result[univ] = metrics
        print(f"  [{univ}] {len(metrics)} documents loaded")
    return result


# ---------------------------------------------------------------------------
# Style
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
# Plot 1: Refs distribution heatmap
# ---------------------------------------------------------------------------

def plot_refs_distribution_heatmap(
    univ_data: dict[str, list[dict]],
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    univs = sorted(univ_data.keys())
    all_refs = [m["total_refs"] for u in univs for m in univ_data[u]]
    if not all_refs:
        return

    max_refs = max(all_refs)
    # Create bins of width 5, at least 10 bins
    bin_size = max(5, math.ceil(max_refs / 20 / 5) * 5)
    bins = list(range(0, max_refs + bin_size + 1, bin_size))
    bin_labels = [f"{bins[i]}–{bins[i+1]-1}" for i in range(len(bins) - 1)]

    rows = []
    for univ in univs:
        counts = [0] * len(bin_labels)
        for m in univ_data[univ]:
            idx = min(int(m["total_refs"] // bin_size), len(counts) - 1)
            counts[idx] += 1
        rows.append(counts)

    df = pd.DataFrame(rows, index=univs, columns=bin_labels)
    # Drop all-zero columns from the right for compactness
    last_nonzero = max((i for i, col in enumerate(df.columns) if df[col].sum() > 0), default=0)
    df = df.iloc[:, :last_nonzero + 1]

    n_unis = len(univs)
    fig, ax = plt.subplots(figsize=(max(fig_width, len(df.columns) * 0.7), max(3, n_unis * 0.7 + 1.5)))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5,
                cbar_kws={"label": "Document count"}, ax=ax)
    ax.set_title("Total References Distribution by University")
    ax.set_xlabel("Reference count (binned)")
    ax.set_ylabel("University")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, out_dir / "refs_distribution_heatmap.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 2: Reference reuse stacked bar
# ---------------------------------------------------------------------------

def plot_reuse_stacked_bar(
    univ_data: dict[str, list[dict]],
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    univs = sorted(univ_data.keys())
    keys = ["1x", "2x", "3x", "4p"]
    palette = sns.color_palette("Blues_d", len(keys))

    rows = []
    for univ in univs:
        totals = Counter()
        for m in univ_data[univ]:
            for k in keys:
                totals[k] += m["reuse"][k]
        total = sum(totals[k] for k in keys)
        if total > 0:
            rows.append({l: totals[k] / total * 100 for k, l in zip(keys, REUSE_LABELS)})
        else:
            rows.append({l: 0.0 for l in REUSE_LABELS})

    df = pd.DataFrame(rows, index=univs)

    n_unis = len(univs)
    fig, ax = plt.subplots(figsize=(fig_width, max(3, n_unis * 0.6 + 1.5)))
    left = np.zeros(n_unis)
    for label, color in zip(REUSE_LABELS, palette):
        vals = df[label].values
        bars = ax.barh(univs, vals, left=left, color=color, label=label, height=0.6)
        for bar, val in zip(bars, vals):
            if val >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", ha="center", va="center", fontsize=8, color="white")
        left += vals

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of cited references")
    ax.set_title("Reference Reuse Distribution by University")
    ax.legend(title="Times cited", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.invert_yaxis()
    _save(fig, out_dir / "reuse_stacked_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 3: Download status stacked bar
# ---------------------------------------------------------------------------

def plot_download_status_stacked_bar(
    univ_data: dict[str, list[dict]],
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    univs = sorted(univ_data.keys())

    # Aggregate statuses
    all_statuses: dict[str, Counter] = {}
    present_statuses: set[str] = set()
    for univ in univs:
        c = Counter()
        for m in univ_data[univ]:
            c.update(m["dl_status"])
        all_statuses[univ] = c
        present_statuses.update(c.keys())

    col_order = [s for s in DOWNLOAD_STATUS_ORDER if s in present_statuses]
    col_order += [s for s in present_statuses if s not in col_order]

    # Color: green for downloaded, warm for abstract, red for failures
    status_colors = {
        "downloaded": "#2ca02c",
        "abstract_saved": "#98df8a",
        "already_indexed": "#1f77b4",
        "download_failed": "#d62728",
        "not_open_access": "#ff7f0e",
        "no_valid_doi": "#ffbb78",
        "no_doi": "#c5b0d5",
        "work_not_found": "#9467bd",
        "empty_reference_index": "#8c564b",
        "other": "#7f7f7f",
    }
    palette = [status_colors.get(s, "#aec7e8") for s in col_order]

    rows = []
    for univ in univs:
        c = all_statuses[univ]
        total = sum(c.values())
        rows.append({s: c.get(s, 0) / total * 100 if total > 0 else 0.0 for s in col_order})

    df = pd.DataFrame(rows, index=univs)

    n_unis = len(univs)
    fig, ax = plt.subplots(figsize=(fig_width, max(3, n_unis * 0.6 + 1.5)))
    left = np.zeros(n_unis)
    for status, color in zip(col_order, palette):
        vals = df[status].values
        bars = ax.barh(univs, vals, left=left, color=color, label=status, height=0.6)
        for bar, val in zip(bars, vals):
            if val >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", ha="center", va="center", fontsize=8, color="white")
        left += vals

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of references")
    ax.set_title("Download Status Distribution by University")
    ax.legend(title="Status", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.invert_yaxis()
    _save(fig, out_dir / "download_status_stacked_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 4: Refs vs citations scatter
# ---------------------------------------------------------------------------

def plot_refs_citations_scatter(
    univ_data: dict[str, list[dict]],
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    rows = []
    for univ in sorted(univ_data.keys()):
        for m in univ_data[univ]:
            rows.append({"University": univ, "total_refs": m["total_refs"],
                         "total_citations": m["total_citations"]})
    if not rows:
        return

    df = pd.DataFrame(rows)
    univs = sorted(df["University"].unique())
    palette = dict(zip(univs, sns.color_palette("tab10", len(univs))))

    fig, ax = plt.subplots(figsize=(fig_width * 0.75, fig_width * 0.75))
    for univ in univs:
        sub = df[df["University"] == univ]
        ax.scatter(sub["total_refs"], sub["total_citations"],
                   color=palette[univ], label=univ, alpha=0.7, s=40, edgecolors="white", linewidths=0.4)

    # Diagonal y=x reference line -> citations == references would mean every ref cited once
    lim = max(df["total_refs"].max(), df["total_citations"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.3, label="1 cite/ref")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Total references in bibliography")
    ax.set_ylabel("Total in-text citation occurrences")
    ax.set_title("References vs Citations per Document")
    ax.legend(title="University", bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, out_dir / "refs_citations_scatter.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 5: Unused reference ratio
# ---------------------------------------------------------------------------

def plot_unused_ratio_bar(
    univ_data: dict[str, list[dict]],
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    univs = sorted(univ_data.keys())
    means = []
    stds = []
    for univ in univs:
        vals = [m["unused_ratio"] for m in univ_data[univ]]
        means.append(np.mean(vals) if vals else 0.0)
        stds.append(np.std(vals) if len(vals) > 1 else 0.0)

    n_unis = len(univs)
    fig, ax = plt.subplots(figsize=(fig_width, max(3, n_unis * 0.6 + 1.5)))
    colors = sns.color_palette("muted", n_unis)
    bars = ax.barh(univs, means, xerr=stds, color=colors, height=0.5,
                   error_kw={"capsize": 4, "elinewidth": 1.5})
    for bar, val, std in zip(bars, means, stds):
        ax.text(val + std + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% of bibliography never cited (mean ± std)")
    ax.set_title("Unused Reference Ratio by University")
    ax.set_xlim(0, min(100, max(m + s for m, s in zip(means, stds)) * 1.3 + 5))
    ax.invert_yaxis()
    _save(fig, out_dir / "unused_ratio_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 6: DOI coverage
# ---------------------------------------------------------------------------

def plot_doi_coverage_bar(
    univ_data: dict[str, list[dict]],
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    univs = sorted(univ_data.keys())
    means = []
    stds = []
    for univ in univs:
        vals = [m["doi_coverage"] for m in univ_data[univ]]
        means.append(np.mean(vals) if vals else 0.0)
        stds.append(np.std(vals) if len(vals) > 1 else 0.0)

    n_unis = len(univs)
    fig, ax = plt.subplots(figsize=(fig_width, max(3, n_unis * 0.6 + 1.5)))
    colors = sns.color_palette("Blues_d", n_unis)
    bars = ax.barh(univs, means, xerr=stds, color=colors, height=0.5,
                   error_kw={"capsize": 4, "elinewidth": 1.5})
    for bar, val, std in zip(bars, means, stds):
        ax.text(val + std + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% of references with a valid DOI (mean ± std)")
    ax.set_title("DOI Coverage by University")
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    _save(fig, out_dir / "doi_coverage_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(univ_data: dict[str, list[dict]], out_dir: Path) -> None:
    rows = []
    for univ in sorted(univ_data.keys()):
        docs = univ_data[univ]
        if not docs:
            continue
        total_refs = [m["total_refs"] for m in docs]
        total_cits = [m["total_citations"] for m in docs]
        reuse_totals = Counter()
        for m in docs:
            for k, v in m["reuse"].items():
                reuse_totals[k] += v
        reuse_sum = sum(reuse_totals.values()) or 1
        top_dl = Counter()
        for m in docs:
            top_dl.update(m["dl_status"])

        rows.append({
            "university": univ,
            "n_documents": len(docs),
            "median_total_refs": round(float(np.median(total_refs)), 1),
            "mean_total_refs": round(float(np.mean(total_refs)), 1),
            "std_total_refs": round(float(np.std(total_refs)), 1),
            "median_total_citations": round(float(np.median(total_cits)), 1),
            "mean_total_citations": round(float(np.mean(total_cits)), 1),
            "std_total_citations": round(float(np.std(total_cits)), 1),
            "mean_unused_ratio_pct": round(float(np.mean([m["unused_ratio"] for m in docs])), 1),
            "mean_doi_coverage_pct": round(float(np.mean([m["doi_coverage"] for m in docs])), 1),
            "pct_reuse_1x": round(reuse_totals["1x"] / reuse_sum * 100, 1),
            "pct_reuse_2x": round(reuse_totals["2x"] / reuse_sum * 100, 1),
            "pct_reuse_3x": round(reuse_totals["3x"] / reuse_sum * 100, 1),
            "pct_reuse_4p": round(reuse_totals["4p"] / reuse_sum * 100, 1),
            "top_download_status": top_dl.most_common(1)[0][0] if top_dl else "",
        })

    df = pd.DataFrame(rows)
    out_path = out_dir / "refs_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")




def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    in_dir = Path(args.in_dir)
    university_files = discover_university_files(in_dir, args.faculty_name)
    if not university_files:
        sys.exit(f"No JSON files found in {in_dir}")

    print(f"[analyze_refs] Loading data from {in_dir} ...")
    univ_data = build_university_frames(university_files)

    if not any(univ_data.values()):
        sys.exit("No valid documents loaded.")

    print(f"[analyze_refs] Generating plots -> {out_dir}")
    plot_refs_distribution_heatmap(univ_data, out_dir, args.fig_width, args.dpi, args.show)
    plot_reuse_stacked_bar(univ_data, out_dir, args.fig_width, args.dpi, args.show)
    plot_download_status_stacked_bar(univ_data, out_dir, args.fig_width, args.dpi, args.show)
    plot_refs_citations_scatter(univ_data, out_dir, args.fig_width, args.dpi, args.show)
    plot_unused_ratio_bar(univ_data, out_dir, args.fig_width, args.dpi, args.show)
    plot_doi_coverage_bar(univ_data, out_dir, args.fig_width, args.dpi, args.show)
    write_summary_csv(univ_data, out_dir)

    print(f"[analyze_refs] Done.")


if __name__ == "__main__":
    main()
