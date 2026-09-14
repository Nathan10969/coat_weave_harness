from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path

from patent_coating.patent_rag_parser import load_content_list, main as parser_main


def write_minimal_patent(folder: Path) -> None:
    folder.mkdir(parents=True)
    records = [
        {
            "type": "text",
            "text": "Durable Coating Patent",
            "text_level": 1,
            "page_idx": 1,
        },
        {
            "type": "text",
            "text": "Detailed Description",
            "text_level": 1,
            "page_idx": 1,
        },
        {
            "type": "text",
            "text": "A durable coating formulation provides measured barrier performance.",
            "page_idx": 1,
        },
    ]
    (folder / "document_content_list.json").write_text(
        json.dumps(records),
        encoding="utf-8",
    )


def run_parser(input_root: Path, output_root: Path) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = parser_main(
            [
                str(input_root),
                "--output",
                str(output_root),
                "--max-tokens",
                "100",
                "--overlap-tokens",
                "0",
            ]
        )
    return result, stdout.getvalue(), stderr.getvalue()


class ParserPathStabilityTests(unittest.TestCase):
    def test_identical_image_records_under_different_roots_make_identical_relative_blocks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "machine-a" / "WO2024000001A1"
            second = root / "machine-b" / "WO2024000001A1"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            image_record = [
                {
                    "type": "image",
                    "page_idx": 4,
                    "img_path": "images/figure-01.png",
                    "image_caption": ["Figure 1. Barrier-layer cross section"],
                    "image_footnote": ["Not to scale"],
                }
            ]
            for folder in (first, second):
                (folder / "document_content_list.json").write_text(
                    json.dumps(image_record),
                    encoding="utf-8",
                )

            first_blocks, first_raw = load_content_list(
                first / "document_content_list.json"
            )
            second_blocks, second_raw = load_content_list(
                second / "document_content_list.json"
            )

            self.assertEqual(first_blocks, second_blocks)
            self.assertEqual(first_raw, second_raw)
            self.assertEqual(len(first_blocks), 1)
            block = first_blocks[0]
            self.assertEqual(block.asset_path, "images/figure-01.png")
            self.assertFalse(Path(block.asset_path).is_absolute())
            self.assertEqual(block.kind, "image")
            self.assertEqual(
                block.caption, "Figure 1. Barrier-layer cross section Not to scale"
            )
            serialized = json.dumps(dataclasses.asdict(block), ensure_ascii=False)
            self.assertNotIn(str(first), serialized)
            self.assertNotIn(str(second), serialized)


class ParserAtomicOutputTests(unittest.TestCase):
    def test_one_patent_failure_leaves_existing_chunks_and_markdown_unchanged(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "mineru"
            output_root = root / "output"
            md_root = output_root / "md"
            input_root.mkdir()
            md_root.mkdir(parents=True)

            write_minimal_patent(input_root / "WO2024000001A1")
            bad_folder = input_root / "WO2024000002A1"
            bad_folder.mkdir()
            (bad_folder / "broken_content_list.json").write_text("{", encoding="utf-8")

            chunks_path = output_root / "patent_chunks.jsonl"
            report_path = output_root / "report.json"
            existing_md = md_root / "WO2024000001A1.md"
            stale_md = md_root / "STALE.md"
            chunks_path.write_bytes(b"KEEP-EXISTING-CHUNKS\n")
            report_path.write_bytes(b"KEEP-EXISTING-REPORT\n")
            existing_md.write_bytes(b"KEEP-EXISTING-PATENT\n")
            stale_md.write_bytes(b"KEEP-EXISTING-STALE\n")

            result, _, stderr = run_parser(input_root, output_root)

            self.assertEqual(result, 1)
            self.assertIn("Parsing had failures", stderr)
            self.assertEqual(chunks_path.read_bytes(), b"KEEP-EXISTING-CHUNKS\n")
            self.assertEqual(report_path.read_bytes(), b"KEEP-EXISTING-REPORT\n")
            self.assertEqual(existing_md.read_bytes(), b"KEEP-EXISTING-PATENT\n")
            self.assertEqual(stale_md.read_bytes(), b"KEEP-EXISTING-STALE\n")

    def test_interrupted_directory_swap_restores_backup_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "mineru"
            output_root = root / "output"
            backup_root = root / ".output.backup"
            input_root.mkdir()
            backup_root.mkdir()
            write_minimal_patent(input_root / "WO2024000001A1")
            (backup_root / "sentinel.txt").write_text(
                "old generation", encoding="utf-8"
            )

            bad_folder = input_root / "WO2024000002A1"
            bad_folder.mkdir()
            (bad_folder / "broken_content_list.json").write_text("{", encoding="utf-8")

            result, _, stderr = run_parser(input_root, output_root)

            self.assertEqual(result, 1)
            self.assertIn("Parsing had failures", stderr)
            self.assertFalse(backup_root.exists())
            self.assertEqual(
                (output_root / "sentinel.txt").read_text(encoding="utf-8"),
                "old generation",
            )

    def test_successful_parse_replaces_chunks_and_removes_stale_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "mineru"
            output_root = root / "output"
            md_root = output_root / "md"
            input_root.mkdir()
            md_root.mkdir(parents=True)
            write_minimal_patent(input_root / "WO2024000001A1")
            stale_md = md_root / "STALE.md"
            stale_md.write_text("obsolete", encoding="utf-8")
            (output_root / "patent_chunks.jsonl").write_text(
                "obsolete chunks\n",
                encoding="utf-8",
            )

            result, _, stderr = run_parser(input_root, output_root)

            self.assertEqual(result, 0, stderr)
            self.assertFalse(stale_md.exists())
            generated_md = md_root / "WO2024000001A1.md"
            self.assertTrue(generated_md.is_file())
            self.assertIn(
                "Durable Coating Patent", generated_md.read_text(encoding="utf-8")
            )
            lines = (
                (output_root / "patent_chunks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertGreaterEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["reference"], "WO2024000001A1")
            self.assertIn("barrier performance", record["content"])


if __name__ == "__main__":
    unittest.main()
