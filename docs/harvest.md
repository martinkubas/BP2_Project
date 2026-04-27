# harvest (src/harvest/)

Downloads the source documents for each reference found in Stage 0, producing enriched JSON files with download status and file paths.

## What it does

For each reference link in a `work_json` file, the harvester:

1. Detects whether the reference is an arXiv paper and skips Crossref if so.
2. Validates or resolves the DOI via the **Crossref API**. Falls back to bibliographic full-text search if no DOI is present.
3. Checks **Milvus** if the DOI has already been embedded from a previous run, marks it `already_indexed` and skips download entirely.
4. Attempts to download the full-text PDF through publisher APIs (**Elsevier**, **Springer**, **IEEE Xplore**, **arXiv**) if the required credentials are set.
5. Falls back to generic HTTP download.
6. Falls back to fetching just the **abstract** from Crossref or Elsevier.
7. If nothing is available, marks the reference as `download_failed`.

### Download status values

| Status            | Meaning                                     |
|-------------------|---------------------------------------------|
| `downloaded`      | Full-text PDF saved to `pdfs/`              |
| `abstract_saved`  | Abstract only, saved to `abstracts/`        |
| `already_indexed` | DOI found in Milvus — skip download         |
| `download_failed` | DOI resolved but file not reachable         |
| `no_valid_doi`    | DOI present but not resolvable via Crossref |
| `no_doi`          | No DOI found and full-text search failed    |
| `work_not_found`  | Crossref returned no matching record        |

## Inputs

`work_json/` directory containing JSON files produced by Stage 0. Each file has a `links` array where each entry contains a `reference` object with `raw`, `doi`, and citation sentences.

## Outputs

- `enriched/<name>.json` — copy of the work JSON with a `download` block added to each reference:
  ```json
  {
    "status": "downloaded",
    "pdf_path": "out/pdfs/10.1234__example.pdf",
    "abstract_path": null
  }
  ```
- `out/pdfs/` — downloaded PDF files, named by DOI
- `out/abstracts/` — plain-text abstract files

## CLI

Called internally by `pipeline.py`. Can also be run standalone:

```bash
python -m harvest.cli \
  --input out/work_json/ \
  --output out/ \
  --enriched-output out/enriched/ \
  --crossref-mailto your@email.com \
  --milvus-uri http://localhost:19530 \
  --embed-model sentence-transformers/paraphrase-xlm-r-multilingual-v1
```

> PDFs land in `<output>/pdfs/` and abstracts in `<output>/abstracts/` automatically.

### Environment variables

| Variable                         | Required                     | Description                                                     |
|----------------------------------|------------------------------|-----------------------------------------------------------------|
| `CROSSREF_MAILTO`                | Optional if provided in args | Contact email for Crossref polite pool                          |
| `ELSEVIER_API_KEY`               | Optional                     | Elsevier Full-Text API                                          |
| `ELSEVIER_INSTTOKEN`             | Optional                     | Elsevier institutional token                                    |
| `IEEE_API_KEY`                   | Optional                     | IEEE metadata/search API key                                    |
| `IEEE_FULL_TEXT_SESSION_TOKEN`   | Optional                     | IEEE chargeable full-text session token if your account uses it |
| `IEEE_FULL_TEXT_AUTH_TOKEN`      | Optional                     | IEEE long-lived auth token used to request a temporary `cltoken`|
| `IEEE_FULL_TEXT_AUTH_CODE`       | Optional                     | Backward-compatible alias for `IEEE_FULL_TEXT_AUTH_TOKEN`       |
| `IEEE_FULL_TEXT_AUTH_CODE_PARAM` | Optional                     | IEEE auth-token query param, default `cltoken`                  |
| `IEEE_FULL_TEXT_FORMAT`          | Optional                     | Full-text response format, default `xml`                        |
| `SPRINGER_FULLTEXT_API_KEY`      | Optional                     | Springer full-text API key                                      |
| `SPRINGER_FULLTEXT_URL_TEMPLATE` | Optional                     | DOI-based Springer full-text endpoint from your contract docs   |
| `SPRINGER_FULLTEXT_API_KEY_PARAM` | Optional                     | Query parameter name for the Springer key, default `api_key`    |
| `SPRINGER_FULLTEXT_API_KEY_HEADER`| Optional                     | Header name for the Springer key if your endpoint expects one   |

Notes:
- IEEE full-text access is a different flow from the old metadata-only branch: the harvester resolves the DOI to article metadata, exchanges your auth token for a temporary `cltoken`, then requests full text by article number.
- Springer is no longer wired to the legacy OA/meta APIs in this branch. The full-text endpoint shape is injected through `SPRINGER_FULLTEXT_URL_TEMPLATE` so it can match the endpoint format in your Springer account documentation.

## Dependencies

- **requests** — HTTP downloads and API calls
- **pymilvus** — Milvus deduplication check

## Key modules

| File                 | Role                                                      |
|----------------------|-----------------------------------------------------------|
| `cli.py`             | Argument parsing, file loop, stats reporting              |
| `reference.py`       | Per-reference downloading                                 |
| `providers.py`       | Publisher-specific downloaders                            |
| `arxiv.py`           | arXiv detection and PDF download                          |
| `crossref.py`        | Crossref API client                                       |
| `http_downloader.py` | Generic HTTP PDF download with HTML landing-page handling |
| `utils.py`           | `DownloadConfig`, Milvus checker, I/O helpers             |
