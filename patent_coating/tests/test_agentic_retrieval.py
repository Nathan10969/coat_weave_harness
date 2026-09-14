from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from patent_coating.search.agentic_retrieval import (
    SandboxFileTools,
    SandboxPathError,
    prepare_agentic_workspace,
    retrieve_document_candidate_stages,
)
from patent_coating.search.embedding_index import SearchHit


class FakeIndex:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.records = [object() for _ in hits]
        self.requested_top_k: list[int] = []

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        self.requested_top_k.append(top_k)
        return self.hits[:top_k]


class RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def __call__(self, queries: list[str], documents: list[str]) -> list[float]:
        self.calls.append((list(queries), list(documents)))
        return [0.95 if "Patent: US-B" in document else 0.10 for document in documents]


def make_hit(
    reference: str,
    score: float,
    text: str,
    title: str,
    section: str,
    chunk_index: int,
) -> SearchHit:
    return SearchHit(
        text=text,
        reference=reference,
        score=score,
        metadata={
            "title": title,
            "section_path": [section],
            "chunk_index": chunk_index,
        },
    )


class DocumentRetrievalTests(unittest.TestCase):
    def test_aggregates_chunks_by_reference_before_reranking(self) -> None:
        index = FakeIndex(
            [
                make_hit("US-A", 0.99, "A strongest", "Alpha", "Abstract", 0),
                make_hit("US-A", 0.90, "A example", "Alpha", "Examples", 1),
                make_hit("US-A", 0.80, "A claim", "Alpha", "Claims", 2),
                make_hit("US-B", 0.75, "B decisive", "Beta", "Results", 0),
                make_hit("US-C", 0.70, "C passage", "Gamma", "Abstract", 0),
            ]
        )
        reranker = RecordingReranker()

        embedding_candidates, final_candidates = retrieve_document_candidate_stages(
            index=index,
            query="durable coating",
            query_vector=[1.0, 0.0],
            reranker=reranker,
            candidate_documents=2,
            final_documents=1,
            chunk_pool_top_k=2,
            max_chunk_pool_top_k=5,
            representative_chunks=2,
            rerank_batch_size=8,
        )

        self.assertEqual(index.requested_top_k, [5])
        self.assertEqual(
            [candidate["reference"] for candidate in embedding_candidates],
            ["US-A", "US-B"],
        )
        self.assertEqual(len(embedding_candidates[0]["matched_chunks"]), 3)
        self.assertEqual(embedding_candidates[0]["embedding_score"], 0.99)
        self.assertIn("Title: Alpha", embedding_candidates[0]["rerank_text"])
        self.assertIn("A strongest", embedding_candidates[0]["rerank_text"])
        self.assertIn("A example", embedding_candidates[0]["rerank_text"])
        self.assertNotIn("A claim", embedding_candidates[0]["rerank_text"])

        self.assertEqual(reranker.calls[0][0], ["durable coating", "durable coating"])
        self.assertEqual(
            [candidate["reference"] for candidate in final_candidates], ["US-B"]
        )
        self.assertEqual(final_candidates[0]["rerank_score"], 0.95)


class WorkspaceAndSandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.cleaned = self.root / "cleaned"
        self.workspace = self.root / "workspace"
        self.cleaned.mkdir()
        (self.cleaned / "US-A.md").write_text(
            "# Alpha Patent\nAlpha COATING result\nthird line\n",
            encoding="utf-8",
        )
        (self.cleaned / "US-B.md").write_text(
            "# Beta Patent\nBeta barrier result\n",
            encoding="utf-8",
        )
        self.candidates = [
            {
                "reference": "US-A",
                "title": "Alpha Patent",
                "embedding_score": 0.9,
                "rerank_score": 0.8,
                "matched_chunks": [{"section_path": ["Examples"]}],
            },
            {
                "reference": "US-B",
                "title": "Beta Patent",
                "embedding_score": 0.7,
                "rerank_score": 0.95,
                "matched_chunks": [{"section_path": ["Results"]}],
            },
        ]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_workspace_copy_manifest_and_read_only_tools(self) -> None:
        self.workspace.mkdir()
        unrelated = self.workspace / "keep.me"
        unrelated.write_text("do not remove", encoding="utf-8")

        manifest = prepare_agentic_workspace(
            self.candidates,
            cleaned_md_root=self.cleaned,
            workspace=self.workspace,
        )

        self.assertEqual(manifest["candidate_count"], 2)
        self.assertEqual(
            [document["reference"] for document in manifest["documents"]],
            ["US-A", "US-B"],
        )
        self.assertTrue((self.workspace / "manifest.json").is_file())
        self.assertEqual(
            (self.workspace / "docs" / "US-A.md").read_text(encoding="utf-8"),
            (self.cleaned / "US-A.md").read_text(encoding="utf-8"),
        )
        self.assertTrue(unrelated.is_file())

        tools = SandboxFileTools(self.workspace, max_read_lines=2)
        self.assertEqual(
            tools.file_find("docs/**/*.md"),
            "docs/US-A.md\ndocs/US-B.md",
        )
        self.assertEqual(
            tools.file_grep("alpha coating", regex=False),
            "docs/US-A.md:2:Alpha COATING result",
        )
        self.assertEqual(
            tools.file_grep("alpha coating", regex=False, case_sensitive=True),
            "NO_MATCHES",
        )
        self.assertEqual(
            tools.file_grep(r"Beta\s+barrier", regex=True),
            "docs/US-B.md:2:Beta barrier result",
        )
        self.assertEqual(
            tools.file_read("docs/US-A.md", start_line=1, end_line=99),
            "FILE docs/US-A.md LINES 1-2\n1: # Alpha Patent\n2: Alpha COATING result",
        )

    def test_find_grep_and_read_cannot_escape_workspace(self) -> None:
        prepare_agentic_workspace(
            self.candidates,
            cleaned_md_root=self.cleaned,
            workspace=self.workspace,
        )
        outside = self.root / "outside.md"
        outside.write_text("SECRET OUTSIDE", encoding="utf-8")
        leak = self.workspace / "docs" / "leak.md"
        leak.symlink_to(outside)
        tools = SandboxFileTools(self.workspace)

        self.assertNotIn("leak.md", tools.file_find("docs/*.md"))
        self.assertNotIn("SECRET OUTSIDE", tools.file_grep("SECRET", path="docs"))
        with self.assertRaises(SandboxPathError):
            tools.file_find("../*.md")
        with self.assertRaises(SandboxPathError):
            tools.file_read("../outside.md")
        with self.assertRaises(SandboxPathError):
            tools.file_read(str(outside))
        with self.assertRaises(SandboxPathError):
            tools.file_grep("SECRET", path="..")
        with self.assertRaises(SandboxPathError):
            tools.file_grep("anything", path="docs", file_glob="../*.md")

    def test_workspace_rejects_reference_traversal_and_stale_patents(self) -> None:
        bad_candidate = {"reference": "../outside", "title": "bad"}
        with self.assertRaisesRegex(ValueError, "plain publication identifier"):
            prepare_agentic_workspace(
                [bad_candidate],
                cleaned_md_root=self.cleaned,
                workspace=self.workspace,
            )

        docs = self.workspace / "fresh" / "docs"
        docs.mkdir(parents=True)
        (docs / "STALE.md").write_text("stale", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "stale candidate files"):
            prepare_agentic_workspace(
                self.candidates,
                cleaned_md_root=self.cleaned,
                workspace=self.workspace / "fresh",
            )


if __name__ == "__main__":
    unittest.main()
