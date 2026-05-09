"""
Persistent vector store for reference document segments using Milvus.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from .segment_models import Segment
from commons.text import slugify


_MAX_SOURCE_KEY_LENGTH = 256
_MAX_SECTION_PATH_LENGTH = 512
_MAX_TEXT_LENGTH = 4096


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Truncate text so its UTF-8 encoding fits within max_bytes.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")

# HNSW index build parameters.
# M=16 and efConstruction=200 give a good accuracy/build-time trade-off for
# small-to-medium collections (thousands to tens of thousands of segments).
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 200

# ef at query time — must be >= top_k to avoid quality degradation.
_HNSW_EF_SEARCH = 64


class MilvusSegmentStore:

    def __init__(
        self,
        milvus_uri: str,
        model_slug: str,
        embedding_dim: int,
        alias: str = "default",
    ) -> None:
        # Milvus collection names may only contain alphanumerics and underscores.
        safe_model_slug = slugify(model_slug).replace("-", "_").replace(".", "_")
        self.collection_name = f"segments_{safe_model_slug}"[:255]
        self.embedding_dim = embedding_dim
        self._alias = alias

        connections.connect(alias=self._alias, uri=milvus_uri)
        self._collection: Collection = self._get_or_create_collection()

    def has_source(self, source_key: str) -> bool:
        existing_rows = self._collection.query(
            expr=f'source_key == "{source_key}"',
            output_fields=["source_key"],
            limit=1,
        )
        return len(existing_rows) > 0

    def store_segments(
        self,
        source_key: str,
        source_type: str,
        segments: List[Segment],
        embedding_matrix: np.ndarray,
    ) -> None:
        if not segments:
            return

        # Truncate text and section_path to fit Milvus VARCHAR limits.
        rows = [
            {
                "source_key":   source_key,
                "source_type":  source_type,
                "zone":         seg.zone,
                "level":        seg.level,
                "section_path": _truncate_utf8(seg.section_path, _MAX_SECTION_PATH_LENGTH),
                "para_i":       seg.para_i,
                "sent_i":       seg.sent_i,
                "text":         _truncate_utf8(seg.text, _MAX_TEXT_LENGTH),
                "embedding":    embedding_matrix[i].tolist(),
            }
            for i, seg in enumerate(segments)
        ]
        self._collection.insert(rows)
        # flush() blocks until Milvus has persisted the data so that a
        # subsequent has_source() call in the same run sees the new rows.
        self._collection.flush()

    def search_similar(
        self,
        source_key: str,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[dict]:
        search_params = {
            "metric_type": "IP",
            "params": {"ef": max(_HNSW_EF_SEARCH, top_k)},
        }
        search_results = self._collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=f'source_key == "{source_key}"',
            output_fields=["zone", "level", "section_path", "para_i", "sent_i", "text"],
        )

        hits = []
        for hit in search_results[0]:
            entity = hit.entity
            hits.append({
                "sim":          float(hit.score),
                "zone":         entity.get("zone"),
                "level":        entity.get("level"),
                "section_path": entity.get("section_path"),
                "para_i":       entity.get("para_i"),
                "sent_i":       entity.get("sent_i"),
                "text":         entity.get("text"),
            })
        return hits

    def close(self) -> None:
        try:
            self._collection.release()
        except Exception:
            pass
        connections.disconnect(self._alias)

    @staticmethod
    def source_exists(milvus_uri: str, model_slug: str, source_key: str) -> bool:
        checker = MilvusSourceChecker(milvus_uri, model_slug)
        try:
            return checker.has_source(source_key)
        finally:
            checker.close()

    def _get_or_create_collection(self) -> Collection:
        if utility.has_collection(self.collection_name, using=self._alias):
            existing_collection = Collection(self.collection_name, using=self._alias)
            existing_collection.load()
            return existing_collection

        schema = self._build_collection_schema()
        new_collection = Collection(name=self.collection_name, schema=schema, using=self._alias)
        new_collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {
                    "M": _HNSW_M,
                    "efConstruction": _HNSW_EF_CONSTRUCTION,
                },
            },
        )
        new_collection.load()
        return new_collection

    def _build_collection_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema("id",           DataType.INT64,        is_primary=True, auto_id=True),
            FieldSchema("source_key",   DataType.VARCHAR,      max_length=_MAX_SOURCE_KEY_LENGTH),
            FieldSchema("source_type",  DataType.VARCHAR,      max_length=16),
            FieldSchema("zone",         DataType.VARCHAR,      max_length=16),
            FieldSchema("level",        DataType.VARCHAR,      max_length=16),
            FieldSchema("section_path", DataType.VARCHAR,      max_length=_MAX_SECTION_PATH_LENGTH),
            FieldSchema("para_i",       DataType.INT32),
            FieldSchema("sent_i",       DataType.INT32),
            FieldSchema("text",         DataType.VARCHAR,      max_length=_MAX_TEXT_LENGTH),
            FieldSchema("embedding",    DataType.FLOAT_VECTOR, dim=self.embedding_dim),
        ]
        return CollectionSchema(
            fields=fields,
            description="Citation verification segments indexed by reference document DOI",
        )


class MilvusSourceChecker:

    def __init__(
        self,
        milvus_uri: str,
        model_slug: str,
        alias: str = "existence_check",
    ) -> None:
        safe_model_slug = slugify(model_slug).replace("-", "_").replace(".", "_")
        self.collection_name = f"segments_{safe_model_slug}"[:255]
        self._alias = alias

        connections.connect(alias=self._alias, uri=milvus_uri)
        self._collection: Collection | None = None
        if utility.has_collection(self.collection_name, using=self._alias):
            self._collection = Collection(self.collection_name, using=self._alias)
            self._collection.load()

    def has_source(self, source_key: str) -> bool:
        if self._collection is None:
            return False
        existing_rows = self._collection.query(
            expr=f'source_key == "{source_key}"',
            output_fields=["source_key"],
            limit=1,
        )
        return len(existing_rows) > 0

    def close(self) -> None:
        try:
            if self._collection is not None:
                self._collection.release()
        except Exception:
            pass
        connections.disconnect(self._alias)
