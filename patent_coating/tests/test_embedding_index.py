from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from patent_coating.search.embedding_index import (
    EMBEDDINGS_FILE,
    MANIFEST_FILE,
    EmbeddingIndex,
    file_sha256,
    inspect_index,
)


class FakeEncoder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append((list(texts), dict(kwargs)))
        return self.vectors.copy()


class EmbeddingIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.records_path = self.root / "patent_chunks.jsonl"
        self.index_dir = self.root / "index"
        self.records = [
            {
                "embedding_text": "alpha embedding",
                "content": "Alpha coating passage",
                "reference": "US-A",
                "title": "Alpha Patent",
                "section_title": "Abstract",
                "section_path": ["Abstract"],
                "metadata": {"chunk_index": 7, "source": "parser"},
            },
            {
                "embedding_text": "beta embedding",
                "content": "Beta coating passage",
                "reference": "US-B",
                "title": "Beta Patent",
                "section_title": "Examples",
                "section_path": ["Description", "Examples"],
                "metadata": {"chunk_index": 3},
            },
            {
                "embedding_text": "opposite embedding",
                "content": "Opposite passage",
                "reference": "US-C",
                "title": "Opposite Patent",
                "section_title": "Claims",
                "section_path": ["Claims"],
                "metadata": {},
            },
        ]
        self._write_records()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_records(self) -> None:
        self.records_path.write_text(
            "".join(json.dumps(record) + "\n" for record in self.records),
            encoding="utf-8",
        )

    def _build(self) -> tuple[EmbeddingIndex, FakeEncoder]:
        encoder = FakeEncoder([[3.0, 0.0], [0.0, 4.0], [-2.0, 0.0]])
        index = EmbeddingIndex.build(
            records_path=self.records_path,
            index_dir=self.index_dir,
            encoder=encoder,
            model_id="fake/embedding-v1",
            max_seq_length=256,
            batch_size=2,
        )
        return index, encoder

    def test_build_load_and_cosine_search(self) -> None:
        index, encoder = self._build()

        self.assertTrue((self.index_dir / EMBEDDINGS_FILE).is_file())
        self.assertTrue((self.index_dir / MANIFEST_FILE).is_file())
        self.assertEqual(
            encoder.calls[0][0],
            [record["embedding_text"] for record in self.records],
        )
        self.assertEqual(encoder.calls[0][1]["batch_size"], 2)
        self.assertIs(encoder.calls[0][1]["normalize_embeddings"], True)
        self.assertIs(encoder.calls[0][1]["convert_to_numpy"], True)
        self.assertAlmostEqual(float(np.linalg.norm(index.embeddings[0])), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(index.embeddings[1])), 1.0)

        status = inspect_index(
            self.index_dir,
            self.records_path,
            model_id="fake/embedding-v1",
            max_seq_length=256,
        )
        self.assertTrue(status.exists)
        self.assertTrue(status.valid)
        self.assertEqual((status.rows, status.dimension), (3, 2))

        loaded = EmbeddingIndex.load(
            records_path=self.records_path,
            index_dir=self.index_dir,
            model_id="fake/embedding-v1",
            max_seq_length=256,
        )
        hits = loaded.search([10.0, 0.0], top_k=2)

        self.assertEqual([hit.reference for hit in hits], ["US-A", "US-B"])
        self.assertAlmostEqual(hits[0].score, 1.0)
        self.assertAlmostEqual(hits[1].score, 0.0)
        self.assertEqual(hits[0].text, "Alpha coating passage")
        self.assertEqual(hits[0].metadata["title"], "Alpha Patent")
        self.assertEqual(hits[0].metadata["section_title"], "Abstract")
        self.assertEqual(hits[0].metadata["section_path"], ["Abstract"])
        self.assertEqual(hits[0].metadata["chunk_index"], 7)
        self.assertEqual(hits[0].metadata["source"], "parser")

    def test_source_hash_change_invalidates_built_index(self) -> None:
        self._build()
        original_hash = file_sha256(self.records_path)

        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")

        self.assertNotEqual(file_sha256(self.records_path), original_hash)
        status = inspect_index(
            self.index_dir,
            self.records_path,
            model_id="fake/embedding-v1",
            max_seq_length=256,
        )
        self.assertTrue(status.exists)
        self.assertFalse(status.valid)
        self.assertEqual(status.message, "parsed chunks changed")
        with self.assertRaisesRegex(RuntimeError, "parsed chunks changed"):
            EmbeddingIndex.load(
                records_path=self.records_path,
                index_dir=self.index_dir,
                model_id="fake/embedding-v1",
                max_seq_length=256,
            )

    def test_search_rejects_invalid_query_or_top_k(self) -> None:
        index, _ = self._build()

        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            index.search([1.0, 0.0], top_k=0)
        with self.assertRaisesRegex(ValueError, "dimension"):
            index.search([1.0], top_k=1)
        with self.assertRaisesRegex(ValueError, "finite and non-zero"):
            index.search([0.0, 0.0], top_k=1)


if __name__ == "__main__":
    unittest.main()
