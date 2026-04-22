# verify (src/verification/cli.py)

Verifies each citation in a corpus by embedding the cited source text and measuring cosine similarity against the citing sentences. Produces a label (`supported`, `related`, or `no_support`) for each citation.

## What it does

For each reference link in an `enriched` JSON file:

1. **Source text extraction** -- if a PDF was downloaded, sends it to **GROBID** to get structured TEI XML, which is split into sentences/paragraphs by zone. If only an abstract is available, uses that directly. TEI output is cached locally so each PDF is sent to GROBID only once.
2. **Embedding** -- segments are embedded with the configured model and stored in **Milvus**, keyed by DOI and model name. Already-indexed references are skipped.
3. **Search** -- each citing sentence from the input thesis is embedded and used to query Milvus for the top-k most similar segments from the cited reference.
4. **Labelling** -- the best cosine similarity score across all retrieved segments determines the label:

| Label        | Threshold |
|--------------|-----------|
| `supported`  | ≥ 0.62    |
| `related`    | ≥ 0.42    |
| `no_support` | < 0.42    |

Both thresholds are configurable before running the verification.

## Inputs

`enriched/` directory containing JSON files from Stage 1. Each file's `links` array must include a `reference.download` block with the PDF or abstract path.

## Outputs

- `verified_json/<name>.json` — enriched JSON with a `verification` block added to each link:
  ```json
  {
    "status": "ok",
    "results": [
      {
        "sentence": "This method was first proposed in [4].",
        "label": "supported",
        "best_sim": 0.71,
        "best_zone": "body",
        "best_abstract_sim": 0.58,
        "best_body_sim": 0.71,
        "top_segments": [...]
      }
    ]
  }
  ```
- `verification_reports/<name>.report.json` — inside of `verification` block added to verified_json
- `verification_reports/summary.json` — aggregate counts across all documents
- `cache/tei/` — GROBID TEI XML cache

## CLI

Called internally by `pipeline.py`. Can also be run standalone:

```bash
python -m verification.cli \
  --enriched out/enriched/ \
  --output out/
```


### Key flags

| Flag               | Default                                                  | Description                            |
|--------------------|----------------------------------------------------------|----------------------------------------|
| `--embed-model`    | `sentence-transformers/paraphrase-xlm-r-multilingual-v1` | Embedding model                        |
| `--device`         | auto                                                     | `cuda` or `cpu`                        |
| `--topk`           | `5`                                                      | Segments retrieved per citing sentence |
| `--support-thresh` | `0.62`                                                   | Threshold for *supported*              |
| `--related-thresh` | `0.52`                                                   | Threshold for *related*                |

### Embedding model options

| Value                                              | Backend             | Notes                                                |
|----------------------------------------------------|---------------------|------------------------------------------------------|
| `sentence-transformers/*`                          | SentenceTransformer | Runs locally; default model is multilingual XLM-R    |
| `voyage-3`, `voyage-3-large`                       | Voyage AI           | Requires `VOYAGE_API_KEY` and `pip install voyageai` |
| `text-embedding-3-small`, `text-embedding-3-large` | OpenAI              | Requires `OPENAI_API_KEY` and `pip install openai`   |

Switching models does not overwrite existing Milvus data — each model uses its own collection.

## Dependencies

- **sentence-transformers** + **torch** — local embedding
- **pymilvus** — vector storage and search
- **spacy** — sentence splitting
- **lxml** — TEI XML parsing
- **reportlab** / **pypdf** — report generation

## Key modules

| File                                    | Role                                                              |
|-----------------------------------------|-------------------------------------------------------------------|
| `src/verification/cli.py`               | Main entry point, per-document orchestration                      |
| `src/verification/embedder.py`          | `SentenceTransformerEmbedder`, `VoyageEmbedder`, `OpenAIEmbedder` |
| `src/verification/milvus_store.py`      | `MilvusSegmentStore` — insert and search segments by DOI          |
| `src/verification/grobid_client.py`     | HTTP wrapper for GROBID TEI extraction                            |
| `src/verification/text_segmenter.py`    | `TEISegmenter`, lxml + spacy, abstract splitter                   |
| `src/verification/citation_verifier.py` | orchestrates extraction -> embedding -> search -> labelling       |
| `src/verification/segment_models.py`    | `Segment` dataclass, DOI slug helper                              |
