# citedetect

Extracts in-text IEEE-style citations from a PDF and returns the surrounding text passages where each citation appears.

## What it does

Scans a PDF page by page for IEEE numeric citation patterns `[1]`, `[1,3]`, `[1–3]` and for each citation occurrence retrieves the sentence/quoted passage that contains it. The result maps each reference index to a list of citing sentences, which become the "claim" side of the later verification step.

The detector handles multi-number groups `[1,2,3]` and ranges `[1–4]` by expanding them and attributing the same passage to each index in the group. If a citation appears right next to quotation marks, the quoted span is returned instead of the one sentence.

## Inputs

| Parameter  | Type          | Description                               |
|------------|---------------|-------------------------------------------|
| `pdf_path` | `str \| Path` | Path to the input PDF                     |
| `out`      | `str \| Path` | If given, the result JSON is written here |

## Outputs

A list of records, one per citation number, each holding all passages that cite it:

```json
[
  {
    "citation": "[3]",
    "sentences": [
      "This approach was introduced in [3] as a baseline for cross-lingual tasks.",
      "Results in [3] show a significant improvement over prior work."
    ]
  }
]
```

## CLI

```bash
python -m citedetect.cli path/to/thesis.pdf --out cites.json
```


| Flag    | Default        | Description      |
|---------|----------------|------------------|
| `--out` | `"cites.json"` | Output JSON path |

## Dependencies

- **PyMuPDF** (`fitz`) — PDF text extraction
- **regex** — extended regex for citation pattern matching

## Key modules

| File            | Role                                                    |
|-----------------|---------------------------------------------------------|
| `detector.py`   | Top-level: page loop, citation grouping, passage lookup |
| `ieee.py`       | Citation pattern definitions, index expansion           |
| `spans.py`      | Sentence boundary detection, quote span detection       |
| `text_utils.py` | PDF text extraction, whitespace normalisation           |
| `cli.py`        | Argument parsing, entry point                           |
