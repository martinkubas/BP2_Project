# refdetect

Extracts the bibliography section from a PDF and parses each entry into a structured reference object with index, raw text, and DOI.

## What it does

Scans a PDF from the end backwards, locating the bibliography by finding a "mostly decreasing" sequence of IEEE-style reference numbers (`[1]`, `[2]`, …) at line beginnings.

Once the bibliography region is found, each numbered entry is split out and parsed:
- The reference index is extracted from the leading `[n]`.
- A DOI is searched for via regex (handles `doi:`, `https://doi.org/`, and bare `10.xxxx/` prefixes).
- The full raw text of the entry is preserved for downstream use.

## Inputs

| Parameter  | Type          | Description                               |
|------------|---------------|-------------------------------------------|
| `pdf_path` | `str \| Path` | Path to the input PDF                     |
| `--out`    | `str \| Path` | If given, the result JSON is written here |

## Outputs

A list of reference objects:

```json
[
  {
    "index": 1,
    "raw": "[6] Martin Potthast et al. “Cross-language plagiarism detection”. In: Language Resources and Evaluation 45.1 (2011),... doi: 10.1007/s10579-009-9114-z ...",
    "doi": "10.1007/s10579-0099114-z"
  }
]
```

## CLI

```bash
python -m refdetect.cli path/to/thesis.pdf --out refs.json
```


| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `refs.json` | Output JSON path |

## Dependencies

- **PyMuPDF** (`fitz`) — PDF text extraction
- **regex** — reference pattern matching and DOI extraction

## Key modules

| File | Role |
|------|------|
| `detector.py` | Bibliography section location, entry splitting, top-level orchestration |
| `ieee.py` | IEEE reference format parsing, DOI regex extraction |
| `models.py` | `Reference` dataclass, JSON serialisation |
| `normalize.py` | Unicode normalisation, whitespace collapsing |
| `cli.py` | Argument parsing, entry point |
