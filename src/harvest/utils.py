from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from commons.io import iter_json_files, write_json_atomic  # noqa: F401  re-exported


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
            "springer_oa": 0,
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
