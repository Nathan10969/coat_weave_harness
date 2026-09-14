"""Small, exact embedding index backed by NumPy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


INDEX_VERSION = 2
EMBEDDINGS_FILE = "embeddings.npy"
MANIFEST_FILE = "index.json"


@dataclass(frozen=True)
class SearchHit:
    text: str
    reference: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class IndexStatus:
    exists: bool
    valid: bool
    message: str
    rows: int = 0
    dimension: int = 0


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing parsed chunks: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("embedding_text") or not record.get("reference"):
                raise ValueError(
                    f"Parsed chunk line {line_number} lacks embedding_text/reference"
                )
            records.append(record)
    if not records:
        raise ValueError(f"No parsed chunks in {path}")
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_index(
    index_dir: Path,
    records_path: Path,
    *,
    model_id: str | None = None,
    max_seq_length: int | None = None,
) -> IndexStatus:
    manifest_path = index_dir / MANIFEST_FILE
    embeddings_path = index_dir / EMBEDDINGS_FILE
    if not manifest_path.is_file() and not embeddings_path.is_file():
        return IndexStatus(False, False, "not built")
    if not manifest_path.is_file() or not embeddings_path.is_file():
        return IndexStatus(True, False, "index files are incomplete")
    if not records_path.is_file():
        return IndexStatus(True, False, f"parsed chunks are missing: {records_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        return IndexStatus(True, False, f"cannot read index: {exc}")
    if not isinstance(manifest, dict):
        return IndexStatus(True, False, "index manifest must be an object")

    try:
        rows = int(manifest.get("rows", 0))
        dimension = int(manifest.get("dimension", 0))
        with records_path.open(encoding="utf-8") as handle:
            record_count = sum(1 for line in handle if line.strip())
    except Exception as exc:
        return IndexStatus(True, False, f"invalid index metadata: {exc}")
    checks = [
        (manifest.get("version") == INDEX_VERSION, "unsupported index version"),
        (embeddings.ndim == 2, "embeddings must be a matrix"),
        (embeddings.shape == (rows, dimension), "manifest and embedding shape differ"),
        (rows == record_count, "manifest and parsed row count differ"),
        (
            manifest.get("embeddings_sha256") == file_sha256(embeddings_path),
            "embedding file checksum differs",
        ),
        (
            manifest.get("source_sha256") == file_sha256(records_path),
            "parsed chunks changed",
        ),
    ]
    if model_id is not None:
        checks.append((manifest.get("model_id") == model_id, "embedding model changed"))
    if max_seq_length is not None:
        checks.append(
            (
                int(manifest.get("max_seq_length", 0)) == int(max_seq_length),
                "embedding max sequence length changed",
            )
        )
    for passed, message in checks:
        if not passed:
            return IndexStatus(True, False, message, rows, dimension)
    return IndexStatus(True, True, "ready", rows, dimension)


class EmbeddingIndex:
    """Exact cosine search over parsed patent chunks."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        embeddings: np.ndarray,
        manifest: dict[str, Any],
    ) -> None:
        self.records = records
        self.embeddings = embeddings
        self.manifest = manifest

    @classmethod
    def build(
        cls,
        *,
        records_path: Path,
        index_dir: Path,
        encoder: Any,
        model_id: str,
        max_seq_length: int,
        batch_size: int,
    ) -> "EmbeddingIndex":
        source_sha256 = file_sha256(records_path)
        records = load_records(records_path)
        if file_sha256(records_path) != source_sha256:
            raise RuntimeError("Parsed chunks changed while they were being loaded")
        texts = [str(record["embedding_text"]) for record in records]
        vectors = np.asarray(
            encoder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(records):
            raise ValueError(
                f"Encoder returned shape {vectors.shape}; expected ({len(records)}, dimension)"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Encoder returned NaN or infinite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Encoder returned a zero vector")
        vectors /= norms
        if file_sha256(records_path) != source_sha256:
            raise RuntimeError(
                "Parsed chunks changed while embeddings were being generated"
            )

        index_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = index_dir / EMBEDDINGS_FILE
        manifest_path = index_dir / MANIFEST_FILE
        temp_embeddings = index_dir / f".{EMBEDDINGS_FILE}.tmp"
        temp_manifest = index_dir / f".{MANIFEST_FILE}.tmp"
        with temp_embeddings.open("wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
        manifest = {
            "version": INDEX_VERSION,
            "rows": len(records),
            "dimension": int(vectors.shape[1]),
            "source_sha256": source_sha256,
            "embeddings_sha256": file_sha256(temp_embeddings),
            "model_id": model_id,
            "max_seq_length": int(max_seq_length),
            "normalized": True,
        }
        temp_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_embeddings.replace(embeddings_path)
        temp_manifest.replace(manifest_path)
        return cls(records, vectors, manifest)

    @classmethod
    def load(
        cls,
        *,
        records_path: Path,
        index_dir: Path,
        model_id: str,
        max_seq_length: int,
    ) -> "EmbeddingIndex":
        status = inspect_index(
            index_dir,
            records_path,
            model_id=model_id,
            max_seq_length=max_seq_length,
        )
        if not status.valid:
            raise RuntimeError(
                f"Embedding index is not usable ({status.message}); run index --rebuild"
            )
        manifest = json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        embeddings = np.load(
            index_dir / EMBEDDINGS_FILE, mmap_mode="r", allow_pickle=False
        )
        records = load_records(records_path)
        if len(records) != embeddings.shape[0]:
            raise RuntimeError(
                "Embedding index and parsed chunks have different row counts"
            )
        return cls(records, embeddings, manifest)

    def search(self, vector: Sequence[float], top_k: int) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.size != self.embeddings.shape[1]:
            raise ValueError(
                f"Query dimension {query.size} does not match index {self.embeddings.shape[1]}"
            )
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Query embedding must be finite and non-zero")
        scores = np.asarray(self.embeddings @ (query / norm), dtype=np.float32)
        limit = min(int(top_k), len(self.records))
        if limit == len(self.records):
            indices = np.argsort(scores)[::-1]
        else:
            subset = np.argpartition(scores, -limit)[-limit:]
            indices = subset[np.argsort(scores[subset])[::-1]]

        hits: list[SearchHit] = []
        for index in indices:
            record = self.records[int(index)]
            metadata = dict(record.get("metadata") or {})
            metadata.update(
                {
                    "title": record.get("title"),
                    "section_title": record.get("section_title"),
                    "section_path": record.get("section_path", []),
                }
            )
            hits.append(
                SearchHit(
                    text=str(record["content"]),
                    reference=str(record["reference"]),
                    metadata=metadata,
                    score=float(scores[int(index)]),
                )
            )
        return hits
