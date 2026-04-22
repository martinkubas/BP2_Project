from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from commons.text import slugify


class BaseEmbedder(ABC):
    """Abstract base for all embedding backends.

    Implementations must return L2-normalised float32 vectors so that inner
    product (IP) in Milvus equals cosine similarity without any extra step.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors this model produces."""

    @property
    @abstractmethod
    def model_slug(self) -> str:
        """Filesystem- and Milvus-safe identifier for this model."""

    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into L2-normalised float32 embedding vectors."""


class SentenceTransformerEmbedder(BaseEmbedder):

    def __init__(self, model_name: str, device: str = "") -> None:
        self.model_name = model_name
        chosen_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = chosen_device
        self._model = SentenceTransformer(model_name, device=chosen_device)
        self._embedding_dim = int(self._model.get_sentence_embedding_dimension())
        self._model_slug = slugify(model_name)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_slug(self) -> str:
        return self._model_slug

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # normalise so IP == cosine similarity
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


_XLM_R_DEFAULT = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
_FALLBACK_HINT = f"Use --embed-model {_XLM_R_DEFAULT} for an offline alternative."


class VoyageEmbedder(BaseEmbedder):

    _MAX_BATCH_SIZE = 128 # Voyage API hard limit

    _DIMENSIONS = {
        "voyage-3":       1024,
        "voyage-3-large": 1024,
    }

    def __init__(self, model_name: str = "voyage-3-large") -> None:
        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "VOYAGE_API_KEY environment variable is not set. "
                f"{_FALLBACK_HINT}"
            )

        import voyageai
        self._client = voyageai.Client(api_key=api_key)
        self.model_name = model_name
        self._model_slug = slugify(model_name)
        self._embedding_dim = self._DIMENSIONS.get(model_name, 1024)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_slug(self) -> str:
        return self._model_slug

    def encode(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        # Respect the Voyage API per-call limit.
        effective_batch = min(batch_size, self._MAX_BATCH_SIZE)
        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), effective_batch):
            batch = texts[start : start + effective_batch]
            result = self._client.embed(batch, model=self.model_name, input_type="document")
            all_vectors.extend(result.embeddings)

        vectors = np.array(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-10)


class OpenAIEmbedder(BaseEmbedder):
    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                f"{_FALLBACK_HINT}"
            )

        import openai  # noqa: PLC0415 — optional dependency
        self._client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self._model_slug = slugify(model_name)
        self._embedding_dim = self._DIMENSIONS.get(model_name, 1536)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_slug(self) -> str:
        return self._model_slug

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self._client.embeddings.create(input=batch, model=self.model_name)
            all_vectors.extend(item.embedding for item in response.data)

        vectors = np.array(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-10)


def create_embedder(model_name: str, device: str = "") -> BaseEmbedder:
    lower_name = model_name.lower()

    if lower_name.startswith("voyage"):
        return VoyageEmbedder(model_name)

    if lower_name.startswith("text-embedding"):
        return OpenAIEmbedder(model_name)

    return SentenceTransformerEmbedder(model_name, device=device)
