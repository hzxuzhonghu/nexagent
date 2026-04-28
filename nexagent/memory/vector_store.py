"""Lightweight numpy-based vector store.

No external dependencies beyond numpy. Suitable for up to ~500K vectors before
O(n) search becomes the bottleneck. The interface is designed so that swapping
in an ANN index (FAISS, HNSW) requires only replacing this file.

Usage::

    store = VectorStore(dim=768)
    store.add("doc-1", embedding, {"source": "web", "text": "hello world"})
    results = store.search(query_embedding, top_k=5)
    store.save("/path/to/store.npz")
    store2 = VectorStore.load("/path/to/store.npz")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """Cosine-similarity vector store backed by a numpy matrix.

    Parameters
    ----------
    dim:
        Embedding dimension. All vectors must match this dimension.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._matrix: FloatArray = np.empty((0, dim), dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int:
        return self._dim

    def add(
        self,
        doc_id: str,
        vector: FloatArray | list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a single vector. Overwrites an existing entry with the same id."""
        vec = self._normalise(np.asarray(vector, dtype=np.float32))
        if vec.shape != (self._dim,):
            raise ValueError(f"Expected vector of dim {self._dim}, got {vec.shape}")

        if doc_id in self._ids:
            idx = self._ids.index(doc_id)
            self._matrix[idx] = vec
            self._metadata[idx] = metadata or {}
        else:
            self._ids.append(doc_id)
            self._metadata.append(metadata or {})
            self._matrix = np.vstack([self._matrix, vec[np.newaxis, :]])

    def add_batch(
        self,
        ids: list[str],
        vectors: FloatArray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add multiple vectors at once (more efficient than repeated add)."""
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"Expected shape (n, {self._dim}), got {vectors.shape}")
        if metadata is None:
            metadata = [{} for _ in ids]
        normed = self._normalise_matrix(vectors.astype(np.float32))
        for doc_id, vec, meta in zip(ids, normed, metadata):
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                self._matrix[idx] = vec
                self._metadata[idx] = meta
            else:
                self._ids.append(doc_id)
                self._metadata.append(meta)
        if self._matrix.shape[0] == 0:
            self._matrix = normed
        else:
            # Rebuild from scratch to keep consistent with per-doc logic
            pass
        # Simpler: rebuild entire matrix from lists
        self._matrix = np.vstack([self._matrix, normed]) if self._matrix.shape[0] > 0 else normed

    def search(
        self,
        query: FloatArray | list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Return top-k most similar documents by cosine similarity."""
        if self.size == 0:
            return []

        q = self._normalise(np.asarray(query, dtype=np.float32))
        if q.shape != (self._dim,):
            raise ValueError(f"Query dim {q.shape[0]} != store dim {self._dim}")

        scores: FloatArray = self._matrix @ q
        effective_k = min(top_k, self.size)
        # np.argpartition is O(n) vs O(n log n) for full sort
        top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_score:
                results.append(
                    SearchResult(
                        id=self._ids[idx],
                        score=score,
                        metadata=self._metadata[idx],
                    )
                )
        return results

    def delete(self, doc_id: str) -> bool:
        """Remove a document. Returns True if found and removed."""
        if doc_id not in self._ids:
            return False
        idx = self._ids.index(doc_id)
        self._ids.pop(idx)
        self._metadata.pop(idx)
        self._matrix = np.delete(self._matrix, idx, axis=0)
        return True

    def save(self, path: str | Path) -> None:
        """Persist the store to a .npz file (metadata as a sidecar JSON)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(p), matrix=self._matrix)
        sidecar = p.with_suffix(".json")
        with sidecar.open("w") as f:
            json.dump(
                {"dim": self._dim, "ids": self._ids, "metadata": self._metadata}, f
            )
        logger.debug("VectorStore saved %d vectors to %s", self.size, p)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Load a store from disk."""
        p = Path(path)
        # np.savez_compressed appends .npz automatically if not present
        npz_path = p if p.suffix == ".npz" else p.with_suffix(".npz")
        sidecar = p.with_suffix(".json")
        with sidecar.open() as f:
            meta = json.load(f)
        store = cls(dim=meta["dim"])
        data = np.load(str(npz_path))
        store._matrix = data["matrix"].astype(np.float32)
        store._ids = meta["ids"]
        store._metadata = meta["metadata"]
        logger.debug("VectorStore loaded %d vectors from %s", store.size, p)
        return store

    @staticmethod
    def _normalise(v: FloatArray) -> FloatArray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return v
        return v / norm

    @staticmethod
    def _normalise_matrix(m: FloatArray) -> FloatArray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return m / norms
