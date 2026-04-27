

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
| `--shared-dir`           | —                                                        | Shared directory for PDFs, abstracts, and TEI cache across parallel runs |
| `--skip-detect`          | —                                                        | Skip Stage 0, reuse existing `work_json/`              |
| `--skip-download`        | —                                                        | Skip Stage 1, reuse existing `enriched/`               |
| `--skip-verify`          | —                                                        | Skip Stage 2                                           |
| `--protocol`             | —                                                        | Append protocol pages to input PDFs (Stage 3)          |
| `--protocol-out-dir`     | `<output>/final_pdfs`                                    | Where to write final PDFs                              |
| `--protocol-max-matches` | `3`                                                      | Max citation matches shown per reference in protocol   |

## Running multiple instances in parallel

Processing large batches of documents can be sped up by running several pipeline instances simultaneously, each targeting a different subset of input PDFs. A single shared directory eliminates redundant downloads and GROBID processing — any reference PDF or TEI file produced by one instance is immediately reused by the others.

### How sharing works

Without `--shared-dir`, each instance stores PDFs, abstracts, and TEI files under its own `--output` directory. The same reference DOI can then be downloaded and GROBID-processed multiple times. With `--shared-dir` all instances point at the same location for those three artefacts:

```
shared/
├── pdfs/               # Downloaded reference PDFs (keyed by DOI)
├── abstracts/          # Plain-text abstracts
└── cache/tei/          # GROBID TEI XML output (keyed by DOI)

out1/                   # Instance 1 — per-run outputs only
├── work_json/
├── enriched/
├── verified_json/
└── verification_reports/

out2/                   # Instance 2 — per-run outputs only
├── work_json/
...
```

Milvus is already shared across all instances (same `--milvus-uri`). Once any instance indexes a reference, subsequent instances skip GROBID and embedding for that DOI entirely.

If two instances reach the same reference at the same moment, a lock file prevents them from both calling GROBID. The second instance waits until the first has written the TEI file, then reads it from cache.

### Example — 4 parallel terminals

Split your input PDFs into four folders, then run one terminal per folder:

```bash
# Terminal 1
python scripts/pipeline.py \
  --input-dir path/to/batch1/ \
  --output out1/ \
  --shared-dir shared/ \
  --crossref-mailto your@email.com

# Terminal 2
python scripts/pipeline.py \
  --input-dir path/to/batch2/ \
  --output out2/ \
  --shared-dir shared/ \
  --crossref-mailto your@email.com

# Terminal 3
python scripts/pipeline.py \
  --input-dir path/to/batch3/ \
  --output out3/ \
  --shared-dir shared/ \
  --crossref-mailto your@email.com

# Terminal 4
python scripts/pipeline.py \
  --input-dir path/to/batch4/ \
  --output out4/ \
  --shared-dir shared/ \
  --crossref-mailto your@email.com
```

All four instances share `shared/pdfs/`, `shared/abstracts/`, and `shared/cache/tei/`. Each instance writes its own `work_json/`, `enriched/`, `verified_json/`, and `verification_reports/` into its own `outN/` directory.

## Dependencies

No direct third-party dependencies beyond the sub-packages
