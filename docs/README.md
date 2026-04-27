# Documentation index

| File | What it covers |
|------|---------------|
| [pipeline.md](pipeline.md) | End-to-end orchestrator — all CLI flags, output layout, and how to run multiple parallel instances with a shared PDF/TEI cache |
| [harvest.md](harvest.md) | Stage 1 in isolation — DOI resolution, provider-specific PDF download, HTTP fallback, abstract fallback, and per-reference status codes |
| [verify.md](verify.md) | Stage 2 in isolation — GROBID segmentation, sentence embedding, Milvus vector search, cosine similarity thresholds, and verification result schema |
| [citedetect.md](citedetect.md) | Stage 0 citation detector — how in-text citation spans are extracted from the input PDF |
| [refdetect.md](refdetect.md) | Stage 0 reference detector — how the bibliography is parsed from the input PDF |
| [analysis.md](analysis.md) | Post-pipeline analysis scripts — aggregating verification results, generating charts and summary statistics |

For a quick-start and project overview see the root [README.md](../README.md).
