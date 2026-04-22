

End-to-end orchestrator that runs all three stages of the citation analysis pipeline on one or more input PDFs.

## What it does

Calls `citedetect`, `refdetect`, `harvest`, and `verify` in sequence, writing intermediate outputs to the configured output directory. Each stage can be skipped independently to resume a partial run without repeating completed work.

### Stages

| Stage        | Script                       | Input            | Output                                               |
|--------------|------------------------------|------------------|------------------------------------------------------|
| 0 — Detect   | `citedetect` + `refdetect`   | PDF              | `work_json/<name>.json`                              |
| 1 — Harvest  | `src/harvest/`               | `work_json/`     | `enriched/<name>.json`, `pdfs/`, `abstracts/`        |
| 2 — Verify   | `src/verification/cli.py`    | `enriched/`      | `verified_json/<name>.json`, `verification_reports/` |
| 3 — Protocol | `scripts/append_protocol.py` | `verified_json/` | `final_pdfs/` - **optional**                         |

## Inputs

One of:

| Flag                | Description                            |
|---------------------|----------------------------------------|
| `--pdf <path>`      | Single PDF file                        |
| `--input-dir <dir>` | Directory of PDFs processed one by one |

## Outputs

```
out/
├── work_json/              # Stage 0: detected refs + citation sentences per input PDF
├── enriched/               # Stage 1: work JSONs enriched with download status + file paths
├── pdfs/                   # Downloaded reference PDFs
├── abstracts/              # Downloaded plain-text abstracts
├── cache/
│   └── tei/                # GROBID TEI XML cache - keyed by DOI
├── verified_json/          # Stage 2: enriched JSONs with verification results added
├── final_pdfs/
│   └── *.protocol.pdf      # Per-input generated protocol PDF        
└── verification_reports/
    ├── *.report.json       # Per-input-PDF summary
    └── summary.json        # Aggregate counts across all input PDFs
```

## CLI

```bash
python scripts/pipeline.py \
  --pdf thesis.pdf \
  --output out/ \
  --crossref-mailto your@email.com \
  --grobid-url http://localhost:8070 \
  --milvus-uri http://localhost:19530
```

### Key flags

| Flag                     | Default                                                  | Description                                            |
|--------------------------|----------------------------------------------------------|--------------------------------------------------------|
| `--output`               | `out/`                                                   | Root output directory                                  |
| `--embed-model`          | `sentence-transformers/paraphrase-xlm-r-multilingual-v1` | Embedding model                                        |
| `--device`               | auto                                                     | `cuda` or `cpu`                                        |
| `--support-thresh`       | `0.62`                                                   | Cosine similarity threshold for *supported* label      |
| `--related-thresh`       | `0.42`                                                   | Cosine similarity threshold for *related* label        |
| `----crossref-min-score` | `0.25`                                                   | Minimum CrossRef candidate score to accept a DOI match |
| `--topk`                 | `5`                                                      | Reference segments retrieved per citing sentence       |
| `--max-refs-per-work`    | `0` (all)                                                | Cap on references processed per PDF                    |
| `--skip-detect`          | —                                                        | Skip Stage 0, reuse existing `work_json/`              |
| `--skip-download`        | —                                                        | Skip Stage 1, reuse existing `enriched/`               |
| `--skip-verify`          | —                                                        | Skip Stage 2                                           |
| `--protocol`             | —                                                        | Append protocol pages to input PDFs (Stage 3)          |
| `--protocol-out-dir`     | `<output>/final_pdfs`                                    | Where to write final PDFs                              |
| `--protocol-max-matches` | `3`                                                      | Max citation matches shown per reference in protocol   |

## Dependencies

No direct third-party dependencies beyond the sub-packages
