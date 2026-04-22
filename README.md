# Citation Analysis Pipeline

Extracts IEEE-style citations from a scholarly PDF, downloads the referenced papers, and verifies how well each citing sentence is supported by the content of the reference it cites.

---

## Overview

The pipeline runs in three stages:

| Stage       | Module                            | What it does                                                                                          |
|-------------|-----------------------------------|-------------------------------------------------------------------------------------------------------|
| 0 - Detect  | `src/refdetect`, `src/citedetect` | Extract bibliography and in-text citations from the input PDF                                         |
| 1 - Harvest | `src/harvest`                     | Resolve DOIs via CrossRef, download reference PDFs or abstracts                                       |
| 2 - Verify  | `src/verification`                | Segment reference text, embed with a sentence model, check cosine similarity against citing sentences |

Results are written to `out/verified_json/` and `out/verification_reports/`.

To analyse results you may use analysis script after the pipeline finishes. Refer to `docs/analysis.md` for more info.

---

## Project structure

```
src/
├── citedetect/       # Stage 0: In-text citation detection
├── refdetect/        # Stage 0: Bibliography extraction
├── harvest/          # Stage 1: DOI resolution + PDF/abstract download
├── verification/     # Stage 2: GROBID segmentation, embedding, Milvus search
├── analysis/         # Post-verification statistics and charts
└── commons/          # Shared utilities

scripts/
├── pipeline.py
└── append_protocol.py  # Append a result report to the input PDF
```

---

## Requirements

- Python **3.13**
- Docker and Docker Compose

---

## Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
pip install -e .
python -m spacy download xx_sent_ud_sm
```

`pip install -e .` registers the packages under `src/` as importable without any PYTHONPATH configuration. Run it once after cloning.

---

## Step 2 — Start services (Milvus + GROBID)

Milvus stores the embeddings of downloaded reference papers persistently across runs. Once a reference has been embedded, it will not be re-processed on the next run even if you analyse a different input PDF. GROBID converts downloaded reference PDFs into structured TEI XML that the pipeline segments into sentences and paragraphs.

```bash
docker compose up -d --wait
```

Milvus will be available at `http://localhost:19530` (GUI at `http://localhost:9091/webui/`) and GROBID at `http://localhost:8070`. Milvus data is persisted in Docker named volumes and survives container restarts. TEI output from GROBID is cached locally under `out/cache/tei/` so each PDF is only sent to GROBID once.

To stop:
```bash
docker compose down
```

To wipe all stored embeddings and start fresh:
```bash
docker compose down -v
```

---

## Step 3 — Set environment variables

The downloader uses the CrossRef API to resolve DOIs. A contact email is required by CrossRef's polite pool policy:

```bash
export CROSSREF_MAILTO="your.email@example.com"
```

Optional API keys for additional download sources:

```bash
export ELSEVIER_API_KEY=""       # Elsevier full-text API
export ELSEVIER_INSTTOKEN=""     # Elsevier institutional token
export IEEE_API_KEY=""           # IEEE Xplore API
export SPRINGER_OA_API_KEY=""    # Springer Open Access API
export SPRINGER_META_API_KEY=""  # Springer Metadata API
```

---

## Running the pipeline

### Full run

```bash
python scripts/pipeline.py \
  --pdf path/to/your/paper.pdf \
  --output out/ \
  --protocol
```

### Multiple input PDFs

```bash
python scripts/pipeline.py \
  --input-dir path/to/pdf/folder/ \
  --output out/ \
  --protocol
```

### Key options
For more key options see `pipeline.md`

| Flag                     | Default                                                  | Description                                              |
|--------------------------|----------------------------------------------------------|----------------------------------------------------------|
| `--embed-model`          | `sentence-transformers/paraphrase-xlm-r-multilingual-v1` | Embedding model for verification                         |
| `--skip-detect`          | -                                                        | Skip Stage 0 -> re-use existing `out/work_json/`         |
| `--skip-download`        | -                                                        | Skip Stage 1 -> re-use existing `out/enriched/`          |
| `--skip-verify`          | -                                                        | Skip Stage 2                                             |

### Switching embedding models

Pass `--embed-model` to the pipeline or verifier to choose a backend. Each model gets its own Milvus collection, so switching models does not overwrite existing embeddings — both are stored in parallel. Re-indexing happens automatically for references not yet embedded with the new model.

| `--embed-model` value                                                | Backend             | Dims | Requirement              |
|----------------------------------------------------------------------|---------------------|------|--------------------------|
| `sentence-transformers/paraphrase-xlm-r-multilingual-v1` *(default)* | SentenceTransformer | 768  | -                        |
| `voyage-3-large`                                                     | Voyage AI           | 1024 | `VOYAGE_API_KEY` env var |
| `voyage-3`                                                           | Voyage AI           | 1024 | `VOYAGE_API_KEY` env var |
| `text-embedding-3-small`                                             | OpenAI              | 1536 | `OPENAI_API_KEY` env var |
| `text-embedding-3-large`                                             | OpenAI              | 3072 | `OPENAI_API_KEY` env var |


**Example — Voyage AI:**
```bash
python scripts/pipeline.py \
  --pdf paper.pdf --output out/ \
  --embed-model voyage-3-large
```

**Example — OpenAI:**
```bash
 python scripts/pipeline.py \
  --pdf paper.pdf --output out/ \
  --embed-model text-embedding-3-large
```

### Limiting stages

```bash
# Detect citations and download sources
python scripts/pipeline.py \
  --pdf path/to/your/paper.pdf \
  --output out/ \
  --skip-verify

# Only verify
python scripts/pipeline.py \
  --pdf path/to/your/paper.pdf \
  --output out/ \
  --skip-detect \
  --skip-download

# Analysis of results
python src/analysis/analyze.py \
  --in-dir out/verified_json \
  --out-dir out/analysis
```

---

## Output structure

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

### Verification labels

Each citing sentence gets one of three labels:

| Label        | Meaning                                                                                                                            |
|--------------|------------------------------------------------------------------------------------------------------------------------------------|
| `supported`  | Cosine similarity ≥ support-thresh — the reference contains text that directly supports the claim                                  |
| `related`    | related-thresh >= Cosine similarity < support-thresh — the reference is topically related but the exact claim is not clearly there |
| `no_support` | Cosine similarity < related-thresh — no strong match found in the reference                                                        |

---

## Cross-run caching

On the first run, each reference PDF is downloaded, sent to GROBID, segmented, and embedded. The vectors are stored in Milvus keyed by DOI.

On subsequent runs:
1. Stage 1 resolves the DOI and checks Milvus - if already indexed, it marks the reference as `already_indexed` and skips the download.
2. Stage 2 sees the `already_indexed` status and skips GROBID + embedding, going directly to vector search.

This means processing a new input PDF that shares references with a previous one is significantly faster after the first run.

---

## Troubleshooting

**GROBID returns an error or times out**
GROBID can take 30–60 s for large PDFs. The default timeout is 180 s. If it times out, restart GROBID and re-run with `--skip-detect --skip-download`.

**Milvus connection refused**
Make sure Milvus is running: `docker compose ps`. The `standalone` container should show `healthy`.

**`xx_sent_ud_sm` not found**
Run `python -m spacy download xx_sent_ud_sm`.
