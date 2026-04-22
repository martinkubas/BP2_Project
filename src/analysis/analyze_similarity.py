import argparse
import json
import sys
from pathlib import Path

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LABEL_ORDER = ["supported", "related", "no_support"]
ZONE_ORDER = ["abstract", "body"]
SIM_BINS = np.linspace(0, 1, 21)  # 20 bins
SIM_BIN_LABELS = [f"{SIM_BINS[i]:.2f}–{SIM_BINS[i+1]:.2f}" for i in range(len(SIM_BINS) - 1)]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Similarity & verification statistics")
    ap.add_argument("--in-dir", required=True, help="Directory containing verified JSON files")
    ap.add_argument("--out-dir", required=True, help="Output directory for plots and CSV")
    ap.add_argument("--show", action="store_true", help="Call plt.show() after each figure")
    ap.add_argument("--support-thresh", type=float, default=0.62)
    ap.add_argument("--related-thresh", type=float, default=0.42)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--fig-width", type=float, default=12)
    ap.add_argument("--faculty-name", default="", help="Label to use instead of 'default' when in-dir has no subdirectories")
    return ap.parse_args()



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
    else:
        files = sorted(in_dir.glob("*.json"))
        label = faculty_name
        return {label: files} if files else {}


def load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}", file=sys.stderr)
        return None


def extract_result_rows(data: dict, university: str, filename: str) -> list[dict]:
    rows = []
    for link in data.get("links", []):
        verification = link.get("verification", {})
        if verification.get("status") != "ok":
            continue
        occurrences = link.get("occurrences", 0)
        link_index = link.get("index", -1)
        for result in verification.get("results", []):
            try:
                best_sim = float(result["best_sim"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({
                "university": university,
                "doc": filename,
                "link_index": link_index,
                "occurrences": occurrences,
                "label": result.get("label", "no_support"),
                "best_sim": best_sim,
                "best_zone": result.get("best_zone", "body"),
                "best_abstract_sim": float(result["best_abstract_sim"]) if result.get("best_abstract_sim") is not None else None,
                "best_body_sim": float(result["best_body_sim"]) if result.get("best_body_sim") is not None else None,
            })
    return rows


def extract_per_doc_stats(data: dict, university: str, filename: str) -> dict | None:
    links = data.get("links", [])
    if not links:
        return None
    total = len(links)
    ok_links = [lk for lk in links if lk.get("verification", {}).get("status") == "ok"]
    verified_ratio = len(ok_links) / total * 100
    sims = [float(r["best_sim"])
            for lk in ok_links
            for r in lk.get("verification", {}).get("results", [])
            if "best_sim" in r]
    mean_sim = float(np.mean(sims)) if sims else 0.0
    return {
        "university": university,
        "doc": filename,
        "verified_ratio": verified_ratio,
        "mean_sim": mean_sim,
    }


def build_dataframes(university_files: dict[str, list[Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    doc_rows = []
    for univ, paths in sorted(university_files.items()):
        for p in paths:
            data = load_json(p)
            if data is None:
                continue
            result_rows.extend(extract_result_rows(data, univ, p.stem))
            doc_stat = extract_per_doc_stats(data, univ, p.stem)
            if doc_stat:
                doc_rows.append(doc_stat)
        print(f"  [{univ}] {len(paths)} documents loaded")

    df_results = pd.DataFrame(result_rows) if result_rows else pd.DataFrame(
        columns=["university", "doc", "link_index", "occurrences", "label",
                 "best_sim", "best_zone", "best_abstract_sim", "best_body_sim"]
    )
    df_docs = pd.DataFrame(doc_rows) if doc_rows else pd.DataFrame(
        columns=["university", "doc", "verified_ratio", "mean_sim"]
    )
    return df_results, df_docs


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
# Plot 1: Similarity density heatmap
# ---------------------------------------------------------------------------

def plot_similarity_density_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    df = df.copy()
    df["sim_bin"] = pd.cut(df["best_sim"], bins=SIM_BINS, labels=SIM_BIN_LABELS, include_lowest=True)
    matrix = pd.crosstab(df["university"], df["sim_bin"])
    matrix = matrix.reindex(columns=SIM_BIN_LABELS, fill_value=0)
    matrix = matrix.reindex(sorted(matrix.index))

    n_unis = len(matrix)
    fig, ax = plt.subplots(figsize=(fig_width * 1.4, max(3, n_unis * 0.65 + 1.5)))
    sns.heatmap(matrix, cmap="viridis", linewidths=0.3, annot=(n_unis <= 10),
                fmt="d", cbar_kws={"label": "Citation count"}, ax=ax)
    ax.set_title("Similarity Score Distribution by University")
    ax.set_xlabel("Best cosine similarity (binned)")
    ax.set_ylabel("University")
    ax.tick_params(axis="x", rotation=60)
    _save(fig, out_dir / "similarity_density_heatmap.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 2: Label stacked bar
# ---------------------------------------------------------------------------

def plot_label_stacked_bar(
    df: pd.DataFrame,
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    univs = sorted(df["university"].unique())
    label_colors = {"supported": "#2ca02c", "related": "#ff7f0e", "no_support": "#d62728"}

    rows = []
    for univ in univs:
        sub = df[df["university"] == univ]
        total = len(sub)
        rows.append({lbl: len(sub[sub["label"] == lbl]) / total * 100 if total > 0 else 0.0
                     for lbl in LABEL_ORDER})

    data = pd.DataFrame(rows, index=univs)
    n_unis = len(univs)
    fig, ax = plt.subplots(figsize=(fig_width, max(3, n_unis * 0.65 + 1.5)))
    left = np.zeros(n_unis)
    for lbl in LABEL_ORDER:
        vals = data[lbl].values
        bars = ax.barh(univs, vals, left=left, color=label_colors[lbl], label=lbl, height=0.6)
        for bar, val in zip(bars, vals):
            if val >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", ha="center", va="center", fontsize=8, color="white")
        left += vals

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of citation results")
    ax.set_title("Citation Support Label Distribution by University")
    ax.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.invert_yaxis()
    _save(fig, out_dir / "label_stacked_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 3: Zone stacked bar
# ---------------------------------------------------------------------------

def plot_zone_stacked_bar(
    df: pd.DataFrame,
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    univs = sorted(df["university"].unique())
    zone_colors = {"abstract": "#1f77b4", "body": "#aec7e8"}

    rows = []
    for univ in univs:
        sub = df[df["university"] == univ]
        total = len(sub)
        rows.append({z: len(sub[sub["best_zone"] == z]) / total * 100 if total > 0 else 0.0
                     for z in ZONE_ORDER})

    data = pd.DataFrame(rows, index=univs)
    n_unis = len(univs)
    fig, ax = plt.subplots(figsize=(fig_width, max(3, n_unis * 0.65 + 1.5)))
    left = np.zeros(n_unis)
    for zone in ZONE_ORDER:
        vals = data[zone].values
        bars = ax.barh(univs, vals, left=left, color=zone_colors[zone], label=zone, height=0.6)
        for bar, val in zip(bars, vals):
            if val >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", ha="center", va="center", fontsize=8, color="white")
        left += vals

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of citation results")
    ax.set_title("Best Match Zone Distribution by University")
    ax.legend(title="Zone", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.invert_yaxis()
    _save(fig, out_dir / "zone_stacked_bar.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 4: Similarity violin
# ---------------------------------------------------------------------------

def plot_similarity_violin(
    df: pd.DataFrame,
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
    support_thresh: float,
    related_thresh: float,
) -> None:
    univs = sorted(df["university"].unique())
    n_unis = len(univs)
    fig, ax = plt.subplots(figsize=(fig_width, max(4, n_unis * 0.8 + 1.5)))

    min_per_univ = min(len(df[df["university"] == u]) for u in univs)
    try:
        if min_per_univ >= 3:
            sns.violinplot(data=df, y="university", x="best_sim", hue="university",
                           inner="box", palette="muted", orient="h", ax=ax,
                           order=univs, legend=False)
        else:
            raise ValueError("Too few points for violin")
    except Exception:
        sns.boxplot(data=df, y="university", x="best_sim", hue="university",
                    palette="muted", orient="h", ax=ax, order=univs, legend=False)
        sns.stripplot(data=df, y="university", x="best_sim", hue="university",
                      palette="muted", orient="h", ax=ax, order=univs,
                      size=3, alpha=0.5, legend=False)

    ax.axvline(support_thresh, color="green", linestyle="--", linewidth=1.5,
               label=f"Supported ≥ {support_thresh}")
    ax.axvline(related_thresh, color="orange", linestyle="--", linewidth=1.5,
               label=f"Related ≥ {related_thresh}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Best cosine similarity")
    ax.set_ylabel("")
    ax.set_title("Best Similarity Distribution per University")
    ax.legend(loc="lower right")
    _save(fig, out_dir / "similarity_violin.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 5: Overall similarity histogram
# ---------------------------------------------------------------------------

def plot_overall_similarity_histogram(
    df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    show: bool,
    support_thresh: float,
    related_thresh: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["best_sim"].dropna(), bins=40, color="#4C72B0", edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvspan(0, related_thresh, alpha=0.07, color="red", label="No support zone")
    ax.axvspan(related_thresh, support_thresh, alpha=0.07, color="orange", label="Related zone")
    ax.axvspan(support_thresh, 1.0, alpha=0.07, color="green", label="Supported zone")
    ax.axvline(related_thresh, color="orange", linestyle="--", linewidth=2,
               label=f"Related threshold ({related_thresh})")
    ax.axvline(support_thresh, color="green", linestyle="--", linewidth=2,
               label=f"Support threshold ({support_thresh})")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Number of citations")
    ax.set_title("Overall Distribution of Best Similarity Scores")
    ax.legend()
    _save(fig, out_dir / "overall_similarity_histogram.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 6: Abstract vs body scatter
# ---------------------------------------------------------------------------

def plot_abstract_body_scatter(
    df: pd.DataFrame,
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    sample = df.dropna(subset=["best_abstract_sim", "best_body_sim"])
    sample = sample[(sample["best_abstract_sim"] > 0.0) & (sample["best_body_sim"] > 0.0)]
    if sample.empty:
        print("  Skipping abstract_body_scatter: no rows with both zones > 0")
        return
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=42)

    label_colors = {"supported": "#2ca02c", "related": "#ff7f0e", "no_support": "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 7))
    for lbl in LABEL_ORDER:
        sub = sample[sample["label"] == lbl]
        ax.scatter(sub["best_abstract_sim"], sub["best_body_sim"],
                   c=label_colors[lbl], label=lbl, alpha=0.4, s=15, edgecolors="none")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="y = x")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Best abstract similarity")
    ax.set_ylabel("Best body similarity")
    ax.set_title("Abstract vs Body Similarity per Citation")
    ax.legend(title="Label")
    _save(fig, out_dir / "abstract_body_scatter.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 7: Similarity vs reference frequency scatter
# ---------------------------------------------------------------------------

def plot_sim_vs_frequency_scatter(
    df: pd.DataFrame,
    out_dir: Path,
    fig_width: float,
    dpi: int,
    show: bool,
) -> None:
    grouped = (df.groupby(["university", "doc", "link_index", "occurrences"])["best_sim"]
                 .mean().reset_index().rename(columns={"best_sim": "mean_sim"}))

    if grouped.empty:
        print("  Skipping sim_vs_frequency: no data")
        return

    univs = sorted(grouped["university"].unique())
    palette = dict(zip(univs, sns.color_palette("tab10", len(univs))))

    fig, ax = plt.subplots(figsize=(fig_width * 0.8, 6))
    for univ in univs:
        sub = grouped[grouped["university"] == univ]
        ax.scatter(sub["occurrences"], sub["mean_sim"],
                   color=palette[univ], label=univ, alpha=0.5, s=20, edgecolors="none")

    x = grouped["occurrences"].values
    y = grouped["mean_sim"].values
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        x_line = np.array([x.min(), x.max()])
        ax.plot(x_line, m * x_line + b, "k--", linewidth=1.5, alpha=0.6,
                label=f"Trend (slope={m:.4f})")

    ax.set_xlabel("Reference occurrence count (times cited)")
    ax.set_ylabel("Mean best cosine similarity")
    ax.set_ylim(0, 1)
    ax.set_title("Similarity vs Citation Frequency per Reference")
    ax.legend(title="University", bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, out_dir / "sim_vs_frequency_scatter.png", dpi, show)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(
    df: pd.DataFrame,
    df_docs: pd.DataFrame,
    university_files: dict[str, list[Path]],
    out_dir: Path,
) -> None:
    rows = []
    for univ in sorted(df["university"].unique()):
        sub = df[df["university"] == univ]
        n_docs = len(university_files.get(univ, []))
        total = len(sub)
        doc_sub = df_docs[df_docs["university"] == univ] if not df_docs.empty else pd.DataFrame()

        label_pcts = {f"pct_{l}": round(len(sub[sub["label"] == l]) / total * 100, 1) if total > 0 else 0.0
                      for l in LABEL_ORDER}
        zone_pcts = {f"pct_zone_{z}": round(len(sub[sub["best_zone"] == z]) / total * 100, 1) if total > 0 else 0.0
                     for z in ZONE_ORDER}
        rows.append({
            "university": univ,
            "n_documents": n_docs,
            "n_citation_results": total,
            "mean_best_sim": round(float(sub["best_sim"].mean()), 4),
            "median_best_sim": round(float(sub["best_sim"].median()), 4),
            "std_best_sim": round(float(sub["best_sim"].std()), 4),
            **label_pcts,
            **zone_pcts,
            "mean_verified_ratio": round(float(doc_sub["verified_ratio"].mean()), 1) if not doc_sub.empty else 0.0,
        })

    out_path = out_dir / "similarity_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    in_dir = Path(args.in_dir)
    university_files = discover_university_files(in_dir, args.faculty_name)
    if not university_files:
        sys.exit(f"No JSON files found in {in_dir}")

    print(f"[analyze_similarity] Loading data from {in_dir} ...")
    df, df_docs = build_dataframes(university_files)

    if df.empty:
        sys.exit("No verified results found. Nothing to plot.")

    print(f"[analyze_similarity] Generating plots -> {out_dir}")
    plot_similarity_density_heatmap(df, out_dir, args.fig_width, args.dpi, args.show)
    plot_label_stacked_bar(df, out_dir, args.fig_width, args.dpi, args.show)
    plot_zone_stacked_bar(df, out_dir, args.fig_width, args.dpi, args.show)
    plot_similarity_violin(df, out_dir, args.fig_width, args.dpi, args.show,
                           args.support_thresh, args.related_thresh)
    plot_overall_similarity_histogram(df, out_dir, args.dpi, args.show,
                                      args.support_thresh, args.related_thresh)
    plot_abstract_body_scatter(df, out_dir, args.fig_width, args.dpi, args.show)
    plot_sim_vs_frequency_scatter(df, out_dir, args.fig_width, args.dpi, args.show)
    write_summary_csv(df, df_docs, university_files, out_dir)
    print(f"[analyze_similarity] Done.")


if __name__ == "__main__":
    main()
