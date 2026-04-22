"""
Citation verification logic.

verify_link() is the core function: given a citation link and a source_key, it returns a verdict dict
describing how well each citing sentence is supported by the reference.

extract_context_sentences() is a schema-agnostic helper that extracts the
citing sentences from a link dict regardless of which JSON key they are stored
under.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .embedder import BaseEmbedder
from .milvus_store import MilvusSegmentStore
from commons.text import normalize_whitespace

# How many characters of a matched segment to include in the output.
# Long segments are truncated to keep the JSON report readable.
_MAX_MATCH_TEXT_CHARS = 500


def extract_context_sentences(link: Dict[str, Any]) -> List[str]:
    collected_sentences: List[str] = []

    def add_if_valid(text: str) -> None:
        cleaned = normalize_whitespace(text)
        if len(cleaned) >= 20:
            collected_sentences.append(cleaned)

    for field_name in ("contexts", "sentences", "citation_sentences", "spans"):
        field_value = link.get(field_name)
        if not field_value or not isinstance(field_value, list):
            continue
        for item in field_value:
            if isinstance(item, str):
                add_if_valid(item)
            elif isinstance(item, dict):
                for text_key in ("sentence", "text", "context", "span"):
                    if isinstance(item.get(text_key), str):
                        add_if_valid(item[text_key])
                        break

    seen_sentences: set = set()
    unique_sentences: List[str] = []
    for sentence in collected_sentences:
        if sentence not in seen_sentences:
            seen_sentences.add(sentence)
            unique_sentences.append(sentence)
    return unique_sentences


def verify_link(
    link: Dict[str, Any],
    source_key: str,
    milvus_store: MilvusSegmentStore,
    embedder: BaseEmbedder,
    top_k: int,
    support_threshold: float,
    related_threshold: float,
) -> Dict[str, Any]:
    citing_sentences = extract_context_sentences(link)
    if not citing_sentences:
        return {"status": "no_context_sentences"}

    query_vectors = embedder.encode(citing_sentences, batch_size=32)

    per_sentence_results = []
    for sentence_index, citing_sentence in enumerate(citing_sentences):
        query_vector = query_vectors[sentence_index]
        top_hits = milvus_store.search_similar(source_key, query_vector, top_k=top_k)

        if not top_hits:
            per_sentence_results.append({
                "citing_sentence":  citing_sentence,
                "label":            "no_support",
                "best_sim":         0.0,
                "best_zone":        None,
                "source_hint":      "unclear",
                "best_abstract_sim": 0.0,
                "best_body_sim":    0.0,
                "top_matches":      [],
            })
            continue

        best_overall_sim = top_hits[0]["sim"]
        best_zone = top_hits[0]["zone"]

        # Separate the best similarity by zone to determine where in the
        # reference the support comes from.
        best_abstract_sim = None
        best_body_sim = None
        for hit in top_hits:
            if hit["zone"] == "abstract":
                best_abstract_sim = max(best_abstract_sim or 0.0, hit["sim"])
            else:
                best_body_sim = max(best_body_sim or 0.0, hit["sim"])

        if best_overall_sim >= support_threshold:
            label = "supported"
        elif best_overall_sim >= related_threshold:
            label = "related"
        else:
            label = "no_support"

        # Source hint: tells the user whether the support comes mainly from
        # the abstract or the body.  The +0.03 margin avoids flipping the
        # hint when scores are nearly equal.
        abs_sim = best_abstract_sim or 0.0
        bod_sim = best_body_sim or 0.0
        if abs_sim >= support_threshold and bod_sim + 0.03 < abs_sim:
            source_hint = "mostly_abstract"
        elif bod_sim >= support_threshold:
            source_hint = "body"
        elif abs_sim >= related_threshold and bod_sim + 0.03 < abs_sim:
            source_hint = "weak_abstract"
        else:
            source_hint = "unclear"

        formatted_matches = [
            {
                "sim":          hit["sim"],
                "zone":         hit["zone"],
                "level":        hit["level"],
                "section_path": hit["section_path"],
                "para_i":       hit["para_i"],
                "sent_i":       hit["sent_i"],
                "text":         (hit["text"] or "")[:_MAX_MATCH_TEXT_CHARS],
            }
            for hit in top_hits
        ]

        per_sentence_results.append({
            "citing_sentence":   citing_sentence,
            "label":             label,
            "best_sim":          best_overall_sim,
            "best_zone":         best_zone,
            "source_hint":       source_hint,
            "best_abstract_sim": best_abstract_sim,
            "best_body_sim":     best_body_sim,
            "top_matches":       formatted_matches,
        })

    return {"status": "ok", "results": per_sentence_results}
