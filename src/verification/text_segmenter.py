"""
Text segmentation for reference documents.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, List

import spacy
from lxml import etree

from .segment_models import Segment
from commons.text import normalize_whitespace
if TYPE_CHECKING:
    from .grobid_client import GrobidClient

TEI_NAMESPACE = {"tei": "http://www.tei-c.org/ns/1.0"}

_LOCK_POLL_INTERVAL = 1    # seconds between polls while waiting for another instance
_LOCK_TIMEOUT       = 90  # seconds before assuming the lock holder crashed

# Minimum character counts used to filter out very short fragments that add
# noise to the index without contributing meaningful semantic content.
_MIN_PARAGRAPH_CHARS = 30
_MIN_SECTION_CHARS = 80
_MIN_SENTENCE_CHARS = 20


class SentenceSplitter:

    def __init__(self) -> None:
        try:
            self.nlp = spacy.load("xx_sent_ud_sm")
        except OSError as error:
            raise RuntimeError(
                "spaCy model xx_sent_ud_sm is not installed. "
                "Run: python -m spacy download xx_sent_ud_sm"
            ) from error

    def split(self, text: str) -> List[str]:
        cleaned_text = normalize_whitespace(text)
        if not cleaned_text:
            return []
        doc = self.nlp(cleaned_text)
        return [
            sentence
            for sentence in (normalize_whitespace(span.text) for span in doc.sents)
            if len(sentence) >= _MIN_SENTENCE_CHARS
        ]


class TEISegmenter:
    def __init__(self, splitter: SentenceSplitter, tei_cache_dir: Path) -> None:
        self.splitter = splitter
        self.tei_cache_dir = tei_cache_dir
        tei_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_process_tei(
        self,
        source_key_slug: str,
        pdf_path: Path,
        grobid_client: "GrobidClient",
    ) -> str:
        tei_cache_path = self.tei_cache_dir / f"{source_key_slug}.tei.xml"
        lock_path      = self.tei_cache_dir / f"{source_key_slug}.tei.xml.lock"

        if tei_cache_path.exists():
            print(f"  [TEI cache] {source_key_slug}", flush=True)
            return tei_cache_path.read_text(encoding="utf-8", errors="ignore")

        # Try to acquire an exclusive lock so only one instance calls Grobid.
        lock_acquired = False
        try:
            open(lock_path, "x").close()
            lock_acquired = True
        except FileExistsError:
            pass

        if not lock_acquired:
            print(f"  [Grobid] {source_key_slug} — waiting for another instance to finish", flush=True)
            waited = 0
            while waited < _LOCK_TIMEOUT:
                time.sleep(_LOCK_POLL_INTERVAL)
                waited += _LOCK_POLL_INTERVAL
                if tei_cache_path.exists():
                    print(f"  [TEI cache] {source_key_slug} (waited {waited}s)", flush=True)
                    return tei_cache_path.read_text(encoding="utf-8", errors="ignore")
            # Lock holder appears to have crashed — proceed without the lock.
            print(f"  [Grobid] {source_key_slug} — lock timeout, processing anyway", flush=True)

        try:
            print(f"  [Grobid] processing {source_key_slug}", flush=True)
            tei_xml = grobid_client.process_fulltext_tei(pdf_path)
            # Write to a temp file then rename so the cache path is never partially written.
            tmp_path = tei_cache_path.with_name(tei_cache_path.name + ".tmp")
            tmp_path.write_text(tei_xml, encoding="utf-8")
            tmp_path.replace(tei_cache_path)
            return tei_xml
        finally:
            if lock_acquired:
                lock_path.unlink(missing_ok=True)

    def segments_from_tei(self, tei_xml: str) -> List[Segment]:
        root = etree.fromstring(tei_xml.encode("utf-8", errors="ignore"))
        segments: List[Segment] = []
        self._extract_abstract_segments(root, segments)
        self._extract_body_segments(root, segments)
        return segments

    def segments_from_abstract_text(self, abstract_text: str) -> List[Segment]:
        segments: List[Segment] = []
        cleaned_abstract = normalize_whitespace(abstract_text)
        if not cleaned_abstract:
            return segments

        for sentence_index, sentence in enumerate(self.splitter.split(cleaned_abstract)):
            segments.append(Segment(
                zone="abstract",
                level="sentence",
                text=sentence,
                section_path="abstract",
                para_i=0,
                sent_i=sentence_index,
            ))

        segments.append(Segment(
            zone="abstract",
            level="paragraph",
            text=cleaned_abstract,
            section_path="abstract",
            para_i=0,
            sent_i=-1,
        ))
        return segments

    def _node_text(self, xml_node) -> str:
        if xml_node is None:
            return ""
        return normalize_whitespace("".join(xml_node.itertext()))

    def _build_section_path(self, heading_stack: List[str]) -> str:
        non_empty_headings = [normalize_whitespace(h) for h in heading_stack if normalize_whitespace(h)]
        return "/".join(non_empty_headings)

    def _extract_abstract_segments(self, root, segments: List[Segment]) -> None:
        abstract_paragraphs = root.xpath(".//tei:abstract//tei:p", namespaces=TEI_NAMESPACE)
        full_abstract_text = normalize_whitespace(
            "\n".join(self._node_text(paragraph) for paragraph in abstract_paragraphs)
        )
        if not full_abstract_text:
            return

        for sentence_index, sentence in enumerate(self.splitter.split(full_abstract_text)):
            segments.append(Segment(
                zone="abstract",
                level="sentence",
                text=sentence,
                section_path="abstract",
                para_i=0,
                sent_i=sentence_index,
            ))

        segments.append(Segment(
            zone="abstract",
            level="paragraph",
            text=full_abstract_text,
            section_path="abstract",
            para_i=0,
            sent_i=-1,
        ))

    def _extract_body_segments(self, root, segments: List[Segment]) -> None:
        body_nodes = root.xpath(".//tei:text/tei:body", namespaces=TEI_NAMESPACE)
        if not body_nodes:
            return
        body = body_nodes[0]

        def walk_div(div_node, heading_stack: List[str]) -> None:
            heading_nodes = div_node.xpath("./tei:head", namespaces=TEI_NAMESPACE)
            heading_text = self._node_text(heading_nodes[0]) if heading_nodes else ""
            new_heading_stack = heading_stack + ([heading_text] if heading_text else [])
            section_path = self._build_section_path(new_heading_stack)

            all_paragraph_nodes = div_node.xpath(".//tei:p", namespaces=TEI_NAMESPACE)
            paragraph_texts = [
                text for text in (self._node_text(p) for p in all_paragraph_nodes)
                if len(text) >= _MIN_PARAGRAPH_CHARS
            ]
            if paragraph_texts:
                section_text = normalize_whitespace("\n".join(paragraph_texts))
                if len(section_text) >= _MIN_SECTION_CHARS:
                    segments.append(Segment(
                        zone="body",
                        level="section",
                        text=section_text,
                        section_path=section_path,
                        para_i=-1,
                        sent_i=-1,
                    ))

            direct_paragraph_nodes = div_node.xpath("./tei:p", namespaces=TEI_NAMESPACE)
            for paragraph_index, paragraph_node in enumerate(direct_paragraph_nodes):
                paragraph_text = self._node_text(paragraph_node)
                if len(paragraph_text) < _MIN_PARAGRAPH_CHARS:
                    continue

                segments.append(Segment(
                    zone="body",
                    level="paragraph",
                    text=paragraph_text,
                    section_path=section_path,
                    para_i=paragraph_index,
                    sent_i=-1,
                ))
                for sentence_index, sentence in enumerate(self.splitter.split(paragraph_text)):
                    segments.append(Segment(
                        zone="body",
                        level="sentence",
                        text=sentence,
                        section_path=section_path,
                        para_i=paragraph_index,
                        sent_i=sentence_index,
                    ))

            for child_div in div_node.xpath("./tei:div", namespaces=TEI_NAMESPACE):
                walk_div(child_div, new_heading_stack)

        for top_level_div in body.xpath("./tei:div", namespaces=TEI_NAMESPACE):
            walk_div(top_level_div, [])

        # Fallback: if GROBID produced no structured <div>s, fall back to all <p> tags.
        if not any(segment.zone == "body" for segment in segments):
            for paragraph_index, paragraph_node in enumerate(body.xpath(".//tei:p", namespaces=TEI_NAMESPACE)):
                paragraph_text = self._node_text(paragraph_node)
                if len(paragraph_text) < _MIN_PARAGRAPH_CHARS:
                    continue
                segments.append(Segment(
                    zone="body",
                    level="paragraph",
                    text=paragraph_text,
                    section_path="",
                    para_i=paragraph_index,
                    sent_i=-1,
                ))
                for sentence_index, sentence in enumerate(self.splitter.split(paragraph_text)):
                    segments.append(Segment(
                        zone="body",
                        level="sentence",
                        text=sentence,
                        section_path="",
                        para_i=paragraph_index,
                        sent_i=sentence_index,
                    ))
