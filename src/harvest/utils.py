from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import requests

from commons.io import iter_json_files, write_json_atomic


@dataclass(frozen=True)
class DownloadConfig:
    crossref_mailto: str
    connect_timeout: int
    read_timeout: int
    sleep_between_calls: float
    overwrite: bool
    max_refs_per_work: int
    crossref_min_score: float
    pdf_dir: Path
    abstract_dir: Path
    log_dir: Path
    transmission_log_path: Path
    output_dir: Path
    enriched_dir: Path


def log(message: str) -> None:
    print(message, flush=True)


def bump(stats: Dict[str, Any], key: str, increment: int = 1) -> None:
    stats[key] = int(stats.get(key, 0)) + increment


def bump_both(stats: Dict[str, Any], work_stats: Dict[str, Any], key: str) -> None:
    bump(stats, key)
    work_stats[key] = work_stats.get(key, 0) + 1


def initial_stats() -> Dict[str, Any]:
    return {
        "works": 0,
        "references": 0,
        "doi_found": 0,
        "no_doi": 0,
        "pdf_ok": 0,
        "pdf_fail": 0,
        "abstract_ok": 0,
        "abstract_fail": 0,
        "by_provider_pdf_ok": {
            "arxiv": 0,
            "elsevier": 0,
            "springer": 0,
            "ieee": 0,
            "http": 0,
        },
        "by_provider_pdf_fail": {
            "arxiv": 0,
            "elsevier": 0,
            "springer": 0,
            "ieee": 0,
            "http": 0,
        },
    }


def set_download_info(reference: Dict[str, Any], **fields: Any) -> None:
    download_dict = reference.get("download")
    if not isinstance(download_dict, dict):
        download_dict = {}
        reference["download"] = download_dict
    for key, value in fields.items():
        if value is None:
            download_dict.pop(key, None)
        else:
            download_dict[key] = value


def safe_filename(text: str, max_length: int = 180) -> str:
    name = (text or "").strip()
    name = re.sub(r"^https?://(dx\.)?doi\.org/", "", name, flags=re.I)
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if len(name) > max_length:
        name = name[:max_length].rstrip("_")
    return name or "file"


def log_provider_transmission(
    log_path: Path,
    provider: str,
    doi: str,
    status_code: int | str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = ",".join([
        datetime.now(timezone.utc).isoformat(),
        _csv_escape(provider),
        _csv_escape(doi),
        _csv_escape(str(status_code)),
    ])
    with open(log_path, "a", encoding="utf-8", newline="") as log_file:
        log_file.write(line)
        log_file.write("\n")


def _csv_escape(value: str) -> str:
    if any(ch in value for ch in [",", "\"", "\n", "\r"]):
        return "\"" + value.replace("\"", "\"\"") + "\""
    return value


def write_response_pdf(response: requests.Response, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as pdf_file:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                pdf_file.write(chunk)


def find_first_string(payload: Any, field_names: Iterable[str]) -> Optional[str]:
    wanted = {field.lower() for field in field_names}

    def visit(node: Any) -> Optional[str]:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in wanted and isinstance(value, str) and value.strip():
                    return value.strip()
            for value in node.values():
                found = visit(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = visit(item)
                if found:
                    return found
        return None

    return visit(payload)


def extract_pdf_url_from_text(text: str) -> Optional[str]:
    for match in re.finditer(r"https?://[^\s\"'<>]+", text or "", flags=re.I):
        candidate = match.group(0).rstrip(".,);")
        if candidate.lower().endswith(".pdf") or "/pdf" in candidate.lower():
            return candidate
    return None


def extract_token_from_text(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"[A-Za-z0-9._=-]{16,}", line):
            return line
    return ""


def extract_pdf_url_from_xml(xml_text: str) -> Optional[str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    for element in root.iter():
        text_value = (element.text or "").strip()
        if text_value:
            candidate = extract_pdf_url_from_text(text_value)
            if candidate:
                return candidate
        for attr_value in element.attrib.values():
            candidate = extract_pdf_url_from_text(str(attr_value))
            if candidate:
                return candidate
    return None


def extract_pdf_url_from_response(response: requests.Response) -> Optional[str]:
    content_type = (response.headers.get("Content-Type") or "").lower()

    try:
        if "json" in content_type:
            payload = response.json()
            return find_first_string(
                payload,
                ("pdf_url", "pdfUrl", "pdf", "url", "href", "value"),
            )
    except Exception:
        pass

    try:
        text = response.text
    except Exception:
        return None

    if "xml" in content_type:
        candidate = extract_pdf_url_from_xml(text)
        if candidate:
            return candidate

    return extract_pdf_url_from_text(text)


def extract_token_from_response(
    response: requests.Response,
    field_names: Iterable[str],
) -> str:
    content_type = (response.headers.get("Content-Type") or "").lower()

    try:
        if "json" in content_type:
            payload = response.json()
            token = find_first_string(payload, field_names)
            if token:
                return token
    except Exception:
        pass

    try:
        text = response.text
    except Exception:
        return ""

    return extract_token_from_text(text)



def build_milvus_checker(
    milvus_uri: str,
    embed_model: str,
) -> Callable[[str], bool]:
    if not (milvus_uri.strip() and embed_model.strip()):
        return lambda _doi: False

    from verification.milvus_store import MilvusSegmentStore
    from commons.text import slugify

    model_slug = embed_model.strip()
    uri = milvus_uri.strip()
    log(f"[Milvus] cache check enabled — URI={uri} model={model_slug}")

    def check_milvus(doi: str) -> bool:
        return MilvusSegmentStore.source_exists(uri, model_slug, slugify(doi))

    return check_milvus
