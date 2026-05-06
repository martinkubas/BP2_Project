# analysis (scripts/analysis/)

Post-processing scripts that read `verified_json/` output and produce statistics tables and plots for the corpus.

## What it does

Three independent scripts cover different aspects of the data. They can be run together via `analyze.py` or individually.

---

### analyze_refs.py — Reference and citation statistics

Computes per-document and per-faculty metrics about bibliography structure and source retrieval, then generates plots.

**Metrics computed:**

| Metric            | Description                                     |
|-------------------|-------------------------------------------------|
| `total_refs`      | Number of bibliography entries                  |
| `refs_cited`      | Number of references actually cited in the text |
| `total_citations` | Total in-text citation occurrences              |
| `unused_ratio`    | % of bibliography entries never cited           |
| `doi_coverage`    | % of references with a detected DOI             |
| `dl_status`       | Breakdown of download outcomes per reference    |

**Plots produced:**

| File                              | Description                                                  |
|-----------------------------------|--------------------------------------------------------------|
| `refs_distribution_heatmap.png`   | Bibliography size distribution by faculty                    |
| `reuse_stacked_bar.png`           | Reference citation frequency by faculty                      |
| `download_status_stacked_bar.png` | Download outcome breakdown by faculty                        |
| `refs_citations_scatter.png`      | Total references vs. total citation occurrences per document |
| `unused_ratio_bar.png`            | Mean unused bibliography ratio by faculty                    |
| `doi_coverage_bar.png`            | Mean DOI coverage by faculty                                 |
| `refs_summary.csv`                | Aggregate statistics table                                   |

---

### analyze_similarity.py — Verification and similarity statistics

Reads per-citation verification results and produces plots about similarity score distributions and label breakdowns.

**Metrics computed per citation result:**

| Metric              | Description                                             |
|---------------------|---------------------------------------------------------|
| `label`             | `supported` / `related` / `no_support`                  |
| `best_sim`          | Highest cosine similarity across all retrieved segments |
| `best_zone`         | Whether the best match came from the abstract or body   |
| `best_abstract_sim` | Best similarity within the abstract zone                |
| `best_body_sim`     | Best similarity within the body zone                    |

**Plots produced:**

| File                               | Description                                              |
|------------------------------------|----------------------------------------------------------|
| `similarity_density_heatmap.png`   | Citation count by faculty × similarity bin (20 bins)     |
| `label_stacked_bar.png`            | Label distribution by faculty                            |
| `zone_stacked_bar.png`             | Best-match zone by faculty                               |
| `similarity_violin.png`            | Similarity distribution per faculty with threshold lines |
| `overall_similarity_histogram.png` | Aggregated histogram of all best_sim scores              |
| `abstract_body_scatter.png`        | Abstract similarity vs. body similarity per citation     |
| `sim_vs_frequency_scatter.png`     | Reference reuse count vs. mean similarity                |
| `similarity_summary.csv`           | Per-faculty aggregates                                   |

---

### analyze_providers.py — Publisher and download funnel statistics

Aggregates publisher distribution, no-DOI domain breakdown, and a download funnel across all processed documents.

**Plots and files produced:**

| File                       | Description                                                            |
|----------------------------|------------------------------------------------------------------------|
| `publishers_bar.png`       | Top 15 publishers by DOI count               |
| `no_doi_domains_bar.png`   | Top 15 domains found in references that have no DOI                    |
| `download_funnel_bar.png`  | 4-stage funnel: Total refs → Has DOI → Downloaded → Abstract           |
| `publishers.csv`           | Publisher name, reference count, and percentage of DOI-bearing refs    |

---

## Inputs

```
verified_json/
├──faculty1/
│  └──*.json
└── faculty2/
    └──*.json
```
**or**
`verified_json/` directory. 

University grouping is auto-detected: if subdirectories are present, each is treated as a faculty. 

If no subdirectory is present, all files are grouped under `"default"` - can be changed using `--faculty-name` argument.

## Outputs

```
out/analysis/
├── refs/
│   ├── *.png
│   └── refs_summary.csv
├── similarity/
│   ├── *.png
│   └── similarity_summary.csv
└── providers/
    ├── *.png
    └── publishers.csv
```

## CLI

```bash
# All three scripts in parallel
python src/analysis/analyze.py \
  --in-dir out/verified_json \
  --out-dir out/analysis

# Reference statistics only
python src/analysis/analyze_refs.py \
  --in-dir out/verified_json \
  --out-dir out/analysis/refs

# Similarity statistics only
python src/analysis/analyze_similarity.py \
  --in-dir out/verified_json \
  --out-dir out/analysis/similarity

# Publisher and download funnel only
python src/analysis/analyze_providers.py \
  --in-dir out/verified_json \
  --out-dir out/analysis/providers
```

### Shared flags

| Flag               | Default    | Description                                               |
|--------------------|------------|-----------------------------------------------------------|
| `--show`           | off        | Call `plt.show()` after each figure                       |
| `--dpi`            | `150`      | Output image resolution                                   |
| `--fig-width`      | `12`       | Base figure width in inches                               |
| `--support-thresh` | `0.62`     | Threshold line for *supported* in similarity plots        |
| `--related-thresh` | `0.42`     | Threshold line for *related* in similarity plots          |
| `--faculty-name`   | `""`       | Name for university in case no subdirectories are present |

## Dependencies

- **pandas** — data loading and aggregation
- **numpy** — binning and numeric operations
- **matplotlib** + **seaborn** — all plots
