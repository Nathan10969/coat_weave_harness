#!/usr/bin/env python3
"""Clean MinerU patent output and build heading-aware RAG chunks.

The parser prefers MinerU's ``*_content_list.json`` because it contains page
numbers and element types.  ``full.md`` is supported as a fallback.  It uses
only the Python standard library; an optional local HuggingFace tokenizer can
be supplied when exact model token counts are required.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence


SPACE_RE = re.compile(r"[\t\u00a0 ]+")
PATENT_NO_RE = re.compile(
    r"\b(?:WO|US|EP)\s*\d{2,4}\s*/?\s*\d{4,10}\s*[A-Z]\d?\b", re.I
)
MARKER_RE = re.compile(
    r"(?:[（(]\s*)?(10|21|22|25|26|30|43|51|54|57|71|72|74|81|84)\s*[）)]", re.I
)
TITLE_MARKER_RE = re.compile(
    r"(?:[（(]\s*)?54\s*[）)]\s*(?:tit(?:le|ie))?\s*:?\s*", re.I
)
ABSTRACT_MARKER_RE = re.compile(r"(?:[（(]\s*)?57\s*[）)]\s*abstract\s*:?\s*", re.I)


MAJOR_HEADINGS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"^(?:\(?57\)?\s*)?abstract\b", re.I)),
    (
        "field",
        re.compile(r"^(?:technical\s+)?field(?:\s+of\s+the\s+invention)?\b", re.I),
    ),
    (
        "background",
        re.compile(
            r"^(?:background(?:\s+of\s+the\s+invention)?|background\s+art|prior\s+art|state\s+of\s+the\s+art)\b",
            re.I,
        ),
    ),
    (
        "summary",
        re.compile(r"^(?:brief\s+)?summary(?:\s+of\s+the\s+invention)?\b", re.I),
    ),
    ("problem", re.compile(r"^(?:technical\s+)?problem\b", re.I)),
    (
        "objective",
        re.compile(
            r"^(?:object(?:ive)?s?|purpose)(?:\s+of\s+the\s+invention)?\b", re.I
        ),
    ),
    ("solution", re.compile(r"^(?:technical\s+)?solution\b", re.I)),
    (
        "drawings",
        re.compile(
            r"^(?:brief\s+)?description\s+of\s+the\s+(?:figures|drawings)\b", re.I
        ),
    ),
    (
        "description",
        re.compile(
            r"^(?:detailed\s+description(?:\s+of\s+the\s+invention)?|description)\b",
            re.I,
        ),
    ),
    ("embodiments", re.compile(r"^(?:preferred\s+)?embodiments?\b", re.I)),
    ("examples", re.compile(r"^(?:reference\s+)?examples?\s*:?$", re.I)),
    (
        "claims",
        re.compile(
            r"^(?:patent\s+claims?|claims?|we\s+claim|in\s+the\s+claims)\s*:?$", re.I
        ),
    ),
]

NOISE_HEADING_RE = re.compile(
    r"^(?:international\s+(?:application|search)|for\s+the\s+purposes\s+of\s+information|"
    r"substitute\s+sheet|published|declarations?\s+under|world\s+intellectual|wipo\s*\|?\s*pct|"
    r"pct\s*/|wo\s+\d|box\s+no\.|further\s+information|[ivx]+\.?\s*observations?)",
    re.I,
)
FRONT_MATTER_RE = re.compile(
    r"^(?:international\s+(?:publication|application|filing|patent\s+classification)|"
    r"filing\s+language|publication\s+language|priority\s+data|designated\s+states|"
    r"applicants?|inventors?|agents?|published|declarations?\s+under)",
    re.I,
)
SEARCH_REPORT_RE = re.compile(r"international\s*search\s*report", re.I)
IMAGE_MD_RE = re.compile(r"^!\[[^]]*]\([^)]*\)\s*$")


@dataclass
class Block:
    text: str
    page: int | None
    kind: str = "text"
    text_level: int | None = None
    raw_type: str = "text"
    asset_path: str | None = None
    caption: str = ""


@dataclass
class Section:
    heading: str
    section_type: str
    path: list[str]
    blocks: list[Block] = field(default_factory=list)


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None:
            value = normalize_inline("".join(self.cell))
            if self.row is not None:
                self.row.append(value)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if any(self.row):
                self.rows.append(self.row)
            self.row = None


def normalize_inline(text: str) -> str:
    text = html.unescape(text).replace("\u200b", "")
    return SPACE_RE.sub(" ", text.replace("\r", " ").replace("\n", " ")).strip()


def html_table_rows(value: str) -> list[list[str]]:
    parser = _TableParser()
    try:
        parser.feed(value)
    except Exception:
        return [[normalize_inline(re.sub(r"<[^>]+>", " ", value))]]
    return parser.rows


def table_to_markdown(value: str) -> str:
    rows = html_table_rows(value)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]

    def escape(value: str) -> str:
        return value.replace("|", "\\|")

    lines = ["| " + " | ".join(escape(x) for x in padded[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    lines.extend("| " + " | ".join(escape(x) for x in row) + " |" for row in padded[1:])
    return "\n".join(lines)


def table_to_text(value: str) -> str:
    return "\n".join(
        " | ".join(cell for cell in row if cell) for row in html_table_rows(value)
    )


def load_content_list(path: Path) -> tuple[list[Block], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks: list[Block] = []
    all_raw: list[str] = []
    for item in data:
        item_type = item.get("type", "text")
        raw = str(item.get("text") or "")
        if item_type == "table":
            body = str(item.get("table_body") or "")
            raw = table_to_text(body)
            rendered = table_to_markdown(body)
        elif item_type == "image":
            captions = item.get("image_caption") or []
            footnotes = item.get("image_footnote") or []
            raw = " ".join(str(x) for x in [*captions, *footnotes])
            caption = (
                normalize_inline(raw)
                or f"Figure on page {(item.get('page_idx') or 0) + 1}"
            )
            relative_image = str(item.get("img_path") or "")
            asset_path = Path(relative_image).as_posix() if relative_image else None
            # The text RAG pipeline cannot inspect image bytes. Keep the useful
            # caption in the corpus and the stable source-relative path in
            # metadata; absolute paths would make chunking machine-dependent.
            rendered = f"[Figure: {caption}]"
        elif item_type == "equation":
            rendered = raw.strip()
        else:
            rendered = normalize_inline(raw)
        if raw.strip():
            all_raw.append(normalize_inline(raw))
        # Discarded MinerU elements are useful for title extraction, not body RAG.
        if item_type == "discarded" or not rendered.strip():
            continue
        blocks.append(
            Block(
                text=rendered.strip(),
                page=item.get("page_idx"),
                kind="table" if item_type == "table" else item_type,
                text_level=item.get("text_level"),
                raw_type=item_type,
                asset_path=asset_path if item_type == "image" else None,
                caption=caption if item_type == "image" else "",
            )
        )
    return blocks, all_raw


def load_markdown(path: Path) -> tuple[list[Block], list[str]]:
    blocks: list[Block] = []
    raw_values: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if not paragraph:
            return
        value = normalize_inline(" ".join(paragraph))
        paragraph.clear()
        if value:
            blocks.append(Block(value, None))
            raw_values.append(value)

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            value = normalize_inline(stripped[level:])
            blocks.append(Block(value, None, text_level=level))
            raw_values.append(value)
        elif stripped.startswith("<table"):
            flush()
            raw_values.append(table_to_text(stripped))
            blocks.append(
                Block(table_to_markdown(stripped), None, kind="table", raw_type="table")
            )
        elif IMAGE_MD_RE.match(stripped):
            flush()
            match = re.match(r"^!\[([^]]*)]\(([^)]*)\)\s*$", stripped)
            if match:
                caption, image_path = match.groups()
                label = normalize_inline(caption) or "Figure"
                blocks.append(
                    Block(
                        f"[Figure: {label}]",
                        None,
                        kind="image",
                        raw_type="image",
                        asset_path=Path(image_path).as_posix() if image_path else None,
                        caption=label,
                    )
                )
        else:
            paragraph.append(stripped)
    flush()
    return blocks, raw_values


def clean_heading(value: str) -> str:
    value = normalize_inline(value).strip("# ")
    value = re.sub(
        r"^(?:WO\s+\d{4}/\d+|PCT/[A-Z]{2}\d{2,4}/\d+)\s+", "", value, flags=re.I
    )
    value = re.sub(r"^\s*\d{1,3}\s*(?=[A-Za-z])", "", value)
    return value.strip(" -:")


def major_type(heading: str) -> str | None:
    normalized = clean_heading(heading)
    normalized = re.sub(r"\bbackqround\b", "background", normalized, flags=re.I)
    normalized = re.sub(r"\bpield\b", "field", normalized, flags=re.I)
    normalized = re.sub(r"\bsunmary\b", "summary", normalized, flags=re.I)
    normalized = re.sub(
        r"^(technical|detailed|brief)(field|description|summary)\b",
        r"\1 \2",
        normalized,
        flags=re.I,
    )
    for name, pattern in MAJOR_HEADINGS:
        if pattern.search(normalized):
            return name
    return None


def is_noise_heading(value: str, reference: str) -> bool:
    value = clean_heading(value)
    compact = re.sub(r"\s+", "", value).upper()
    ref_compact = re.sub(r"\s+", "", reference).upper()
    if not value or len(value) > 180 or value in {"and", "and/or", "or"}:
        return True
    if re.fullmatch(r"[-–—]?\d{1,4}[-–—]?|[ivxlcdm]+[.)]?", value, re.I):
        return True
    if ref_compact and (
        compact == ref_compact or compact.startswith(ref_compact + "PATENT")
    ):
        return True
    if PATENT_NO_RE.fullmatch(value) or NOISE_HEADING_RE.search(value):
        return True
    if MARKER_RE.match(value) and major_type(value) is None:
        return True
    return False


def is_heading(block: Block, reference: str) -> bool:
    value = clean_heading(block.text)
    if is_noise_heading(value, reference):
        return False
    if major_type(value):
        return True
    if block.text_level is None:
        return False
    if value.endswith((".", ";", ",")) or len(value.split()) > 18:
        return False
    if re.fullmatch(r"(?:pounds|units|streams|others|percent by volume)", value, re.I):
        return False
    return len(value) >= 3


def find_title(
    raw_values: Sequence[str], blocks: Sequence[Block], reference: str
) -> str:
    for index, value in enumerate(raw_values):
        match = TITLE_MARKER_RE.search(value)
        if not match:
            continue
        candidate = value[match.end() :]
        candidate = ABSTRACT_MARKER_RE.split(candidate, maxsplit=1)[0]
        candidate = re.split(r"[（(]\s*57\s*[）)]", candidate, maxsplit=1, flags=re.I)[
            0
        ]
        candidate = normalize_inline(candidate).strip(" :-")
        candidate = re.sub(
            r"\s+FIG(?:URE)?\.?\s*\d.*$", "", candidate, flags=re.I
        ).strip()
        if len(re.findall(r"[A-Za-z]", candidate)) >= 10:
            candidate = re.sub(r"\s+[\u3400-\u9fff]{1,2}$", "", candidate).strip()
        # Old PCT covers sometimes continue a long title in the next H1 element.
        if (
            candidate
            and index + 1 < len(raw_values)
            and re.search(r"\b(?:and|or|of|for|with)$", candidate, re.I)
        ):
            continuation = clean_heading(raw_values[index + 1])
            if (
                continuation
                and len(continuation) < 120
                and not MARKER_RE.match(continuation)
            ):
                candidate = f"{candidate} {continuation}"
        if 4 <= len(candidate) <= 350:
            return candidate
    # Fall back to a repeated body title near the beginning.
    for block in blocks[:80]:
        value = clean_heading(block.text)
        if (
            block.text_level
            and not is_noise_heading(value, reference)
            and not major_type(value)
        ):
            if 4 <= len(value) <= 250 and len(value.split()) >= 2:
                return value
    return reference


def find_abstract(raw_values: Sequence[str]) -> str:
    for index, value in enumerate(raw_values):
        match = ABSTRACT_MARKER_RE.search(value)
        if not match:
            continue
        candidate = normalize_inline(value[match.end() :])
        if len(candidate) >= 30:
            return candidate
        for following in raw_values[index + 1 : index + 4]:
            following = normalize_inline(following)
            if len(following) >= 30 and not MARKER_RE.match(following):
                return following
    # Some early PCT covers put (54) in a table but omit the (57) label from
    # MinerU's table extraction.  The next substantial cover element is then
    # the abstract.
    for index, value in enumerate(raw_values):
        if not TITLE_MARKER_RE.search(value):
            continue
        for following in raw_values[index + 1 : index + 4]:
            following = normalize_inline(following)
            if len(following) >= 60 and not MARKER_RE.match(following):
                return following
    return ""


def infer_reference(folder: Path, raw_values: Sequence[str]) -> str:
    name = folder.name.strip()
    if re.fullmatch(r"(?:WO|US|EP)\d+[A-Z]\d?", name, re.I):
        return name.upper()
    for value in raw_values:
        match = PATENT_NO_RE.search(value)
        if match:
            return re.sub(r"[\s/]", "", match.group(0)).upper()
    return name


def looks_like_front_matter(value: str) -> bool:
    cleaned = clean_heading(value)
    return bool(MARKER_RE.match(cleaned) or FRONT_MATTER_RE.match(cleaned))


def build_sections(
    blocks: Sequence[Block], reference: str, patent_title: str, abstract: str
) -> list[Section]:
    sections: list[Section] = []
    if abstract:
        sections.append(
            Section("Abstract", "abstract", ["Abstract"], [Block(abstract, 0)])
        )
    current = Section("Description", "description", ["Description"])
    current_major = "Description"
    skip_info_page: int | None = None
    seen_body = False

    def commit() -> None:
        nonlocal current
        if current.blocks:
            sections.append(current)

    for block in blocks:
        value = clean_heading(block.text)
        if block.text_level is not None and re.match(
            r"^international\s*search\s*report\b", value, re.I
        ):
            break
        if block.kind == "table" and (
            re.match(r"^\|?\s*international\s*search\s*report\b", block.text, re.I)
            or (
                re.search(r"fields\s*searched", block.text, re.I)
                and re.search(r"documents\s+considered", block.text, re.I)
            )
        ):
            break
        if ABSTRACT_MARKER_RE.match(value):
            continue
        if TITLE_MARKER_RE.match(value) or (
            not seen_body and looks_like_front_matter(value)
        ):
            continue
        if re.match(r"^for\s+the\s+purposes\s+of\s+information", value, re.I):
            skip_info_page = block.page
            continue
        if skip_info_page is not None:
            if block.page == skip_info_page:
                continue
            skip_info_page = None
        if block.text_level is not None and is_noise_heading(value, reference):
            continue
        if block.kind == "text" and (
            re.fullmatch(r"[\d,]+", value) or PATENT_NO_RE.fullmatch(value)
        ):
            continue
        # Cover pages are metadata.  Modern documents normally repeat the
        # patent title at the start of the specification; early documents may
        # begin with prose on page 2 after a one-page PCT information sheet.
        if not seen_body:
            if block.page == 0:
                continue
            same_as_title = (
                re.sub(r"\W", "", value).casefold()
                == re.sub(r"\W", "", patent_title).casefold()
            )
            if same_as_title:
                seen_body = True
                continue
            if (
                block.page is not None
                and block.page <= 1
                and not is_heading(block, reference)
            ):
                continue
        if is_heading(block, reference):
            seen_body = True
            commit()
            kind = major_type(value)
            if kind:
                current_major = value
                path = [value]
                section_type = kind
            else:
                path = [current_major, value] if current_major != value else [value]
                section_type = (
                    "example"
                    if re.match(
                        r"^(?:inventive\s+|comparative\s+|reference\s+)?example\b",
                        value,
                        re.I,
                    )
                    else "subsection"
                )
            current = Section(value, section_type, path)
            continue
        # Skip page headers and bare line numbers even when MinerU called them text.
        if re.fullmatch(
            r"(?:WO\s+\d{4}/\d+|PCT/[A-Z]{2}\d{2,4}/\d+|\d{1,3})", value, re.I
        ):
            continue
        seen_body = True
        current.blocks.append(block)
    commit()
    return sections


def count_approx_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    non_cjk = re.sub(r"[\u3400-\u9fff]", " ", text)
    pieces = re.findall(r"[A-Za-z0-9]+|[^\w\s]", non_cjk)
    # Long OCR-joined words consume multiple subword tokens.
    long_word_extra = sum(max(0, (len(x) - 1) // 6) for x in pieces if x.isalnum())
    return cjk + len(pieces) + long_word_extra


def make_counter(tokenizer_path: str | None) -> tuple[Callable[[str], int], str]:
    if not tokenizer_path:
        return count_approx_tokens, "approx"
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "--tokenizer requires transformers; omit it to use the built-in estimator"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False)), str(
        tokenizer_path
    )


def split_long_text(
    text: str, max_tokens: int, count_tokens: Callable[[str], int]
) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    sentences = re.split(r"(?<=[.!?。！？;；])\s+", text)
    if len(sentences) == 1:
        words = text.split()
        if len(words) == 1:
            ratio = max(1, len(text) * max_tokens // max(count_tokens(text), 1))
            return [text[i : i + ratio] for i in range(0, len(text), ratio)]
        sentences = words
    result: list[str] = []
    current: list[str] = []
    for part in sentences:
        trial = " ".join(current + [part]).strip()
        if current and count_tokens(trial) > max_tokens:
            result.append(" ".join(current).strip())
            current = [part]
        else:
            current.append(part)
    if current:
        value = " ".join(current).strip()
        if count_tokens(value) > max_tokens and value != text:
            result.extend(split_long_text(value, max_tokens, count_tokens))
        else:
            result.append(value)
    final: list[str] = []
    for value in result:
        if count_tokens(value) <= max_tokens:
            final.append(value)
            continue
        # Last-resort monotonic split for OCR-joined chemical names and very
        # large tables. Binary search keeps the guarantee independent of the
        # tokenizer used.
        remaining = value
        while remaining:
            low, high = 1, len(remaining)
            while low < high:
                mid = (low + high + 1) // 2
                if count_tokens(remaining[:mid]) <= max_tokens:
                    low = mid
                else:
                    high = mid - 1
            cut = max(1, low)
            if cut < len(remaining):
                boundary = max(
                    remaining.rfind(" ", 0, cut), remaining.rfind("\n", 0, cut)
                )
                if boundary >= cut // 2:
                    cut = boundary
            final.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
    return [x for x in final if x]


def chunk_section(
    section: Section,
    max_tokens: int,
    overlap_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[tuple[str, int | None, int | None, int, list[dict]]]:
    units: list[Block] = []
    for block in section.blocks:
        for part in split_long_text(block.text, max_tokens, count_tokens):
            units.append(
                Block(
                    part,
                    block.page,
                    block.kind,
                    block.text_level,
                    block.raw_type,
                    block.asset_path,
                    block.caption,
                )
            )
    chunks: list[tuple[str, int | None, int | None, int, list[dict]]] = []
    current: list[Block] = []

    def render(values: Sequence[Block]) -> str:
        return "\n\n".join(x.text for x in values).strip()

    def emit() -> None:
        nonlocal current
        if not current:
            return
        body = render(current)
        pages = [x.page for x in current if x.page is not None]
        images = [
            {"path": x.asset_path, "caption": x.caption, "page": x.page}
            for x in current
            if x.asset_path
        ]
        chunks.append(
            (
                body,
                min(pages) if pages else None,
                max(pages) if pages else None,
                count_tokens(body),
                images,
            )
        )
        if overlap_tokens <= 0:
            current = []
            return
        tail: list[Block] = []
        for unit in reversed(current):
            if tail and count_tokens(render([unit] + tail)) > overlap_tokens:
                break
            tail.insert(0, unit)
        current = tail if len(tail) < len(current) else []

    for unit in units:
        trial = render(current + [unit])
        if current and count_tokens(trial) > max_tokens:
            emit()
        if current and count_tokens(render(current + [unit])) > max_tokens:
            # The next unit already fills a chunk, so overlap must yield.
            current = []
        current.append(unit)
    emit()
    return chunks


def clean_cross_page_hyphenation(section: Section) -> None:
    merged: list[Block] = []
    for block in section.blocks:
        if (
            merged
            and merged[-1].page != block.page
            and merged[-1].kind == block.kind == "text"
        ):
            previous = merged[-1].text
            if re.search(r"[A-Za-z]-$", previous) and re.match(r"^[a-z]", block.text):
                merged[-1].text = previous[:-1] + block.text
                continue
        merged.append(block)
    section.blocks = merged


def iter_patent_dirs(input_path: Path) -> Iterator[Path]:
    if input_path.is_file():
        yield input_path.parent
        return
    if any(input_path.glob("*_content_list.json")) or (input_path / "full.md").exists():
        yield input_path
        return
    for child in sorted(input_path.iterdir()):
        if child.is_dir() and (
            any(child.glob("*_content_list.json")) or (child / "full.md").exists()
        ):
            yield child


def parse_patent(folder: Path) -> tuple[dict, list[Section]]:
    content_files = sorted(folder.glob("*_content_list.json"))
    if content_files:
        source = content_files[0]
        blocks, raw_values = load_content_list(source)
    elif (folder / "full.md").exists():
        source = folder / "full.md"
        blocks, raw_values = load_markdown(source)
    else:
        raise FileNotFoundError(f"No MinerU content list or full.md in {folder}")
    reference = infer_reference(folder, raw_values)
    title = find_title(raw_values, blocks, reference)
    abstract = find_abstract(raw_values)
    sections = build_sections(blocks, reference, title, abstract)
    for section in sections:
        clean_cross_page_hyphenation(section)
    metadata = {
        "reference": reference,
        "patent_title": title,
        "abstract": abstract,
        "source_path": f"{folder.name}/{source.name}",
        "source_format": "mineru_content_list" if content_files else "mineru_markdown",
    }
    return metadata, sections


def section_markdown(metadata: dict, sections: Sequence[Section]) -> str:
    lines = [
        f"# {metadata['patent_title']}",
        "",
        f"- Patent: {metadata['reference']}",
        "",
    ]
    for section in sections:
        if not section.blocks:
            continue
        level = 2 if len(section.path) == 1 else 3
        lines.extend(["#" * level + " " + section.heading, ""])
        lines.extend([block.text + "\n" for block in section.blocks])
    return "\n".join(lines).rstrip() + "\n"


def build_records(
    metadata: dict,
    sections: Sequence[Section],
    max_tokens: int,
    overlap_tokens: int,
    count_tokens: Callable[[str], int],
    tokenizer_name: str,
) -> list[dict]:
    records: list[dict] = []
    for section_index, section in enumerate(sections):
        chunks = chunk_section(section, max_tokens, overlap_tokens, count_tokens)
        for chunk_index, (
            content,
            page_start,
            page_end,
            token_count,
            images,
        ) in enumerate(chunks):
            path = " > ".join(section.path)
            embedding_content = re.sub(
                r"!\[([^]]*)]\([^)]*\)", lambda m: f"[Figure: {m.group(1)}]", content
            )
            embedding_text = f"Patent Title: {metadata['patent_title']}\nSection: {path}\n\n{embedding_content}"
            digest = hashlib.sha1(
                f"{metadata['reference']}|{section_index}|{chunk_index}|{content}".encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
            records.append(
                {
                    "id": f"{metadata['reference']}-{digest}",
                    # ``title`` is intentionally the complete document title.
                    # Section identity stays separate so document-level recall and
                    # heading-aware expansion do not overload one field.
                    "title": metadata["patent_title"],
                    "section_title": section.heading,
                    "section_path": section.path,
                    "content": content,
                    "embedding_text": embedding_text,
                    "reference": metadata["reference"],
                    "images": images,
                    "metadata": {
                        "patent_title": metadata["patent_title"],
                        "title": metadata["patent_title"],
                        "section_title": section.heading,
                        "section_path": section.path,
                        "section_type": section.section_type,
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                        "chunk_total": len(chunks),
                        "page_start": page_start,
                        "page_end": page_end,
                        "token_count": token_count,
                        "tokenizer": tokenizer_name,
                        "has_images": bool(images),
                        "image_count": len(images),
                        "source_path": metadata["source_path"],
                    },
                }
            )
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    return count


def _remove_output_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _recover_interrupted_output(output_dir: Path) -> Path:
    """Recover the old output if a previous directory swap was interrupted."""
    backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if backup_dir.exists():
        if output_dir.exists():
            _remove_output_path(backup_dir)
        else:
            backup_dir.replace(output_dir)
    return backup_dir


def _commit_outputs(
    output_dir: Path,
    summary: dict,
    records: Iterable[dict],
    markdown_documents: dict[str, str],
    *,
    write_markdown: bool,
) -> Path:
    """Stage a complete generation, then swap the output directory."""
    backup_dir = _recover_interrupted_output(output_dir)
    stage_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    committed = False
    try:
        (stage_dir / "report.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        jsonl_path = stage_dir / "patent_chunks.jsonl"
        write_jsonl(jsonl_path, records)
        if write_markdown:
            md_dir = stage_dir / "md"
            md_dir.mkdir()
            for reference, markdown in markdown_documents.items():
                (md_dir / f"{reference}.md").write_text(
                    markdown, encoding="utf-8", newline="\n"
                )

        if output_dir.exists():
            output_dir.replace(backup_dir)
        try:
            stage_dir.replace(output_dir)
            committed = True
        except Exception:
            if backup_dir.exists() and not output_dir.exists():
                backup_dir.replace(output_dir)
            raise
    finally:
        if stage_dir.exists():
            _remove_output_path(stage_dir)

    if committed and backup_dir.exists():
        try:
            _remove_output_path(backup_dir)
        except OSError as exc:
            print(f"[WARN] Could not remove old output backup: {exc}", file=sys.stderr)
    return output_dir / "patent_chunks.jsonl"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="MinerU root, one patent directory, content JSON, or full.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("patent_rag_output"),
        help="output directory",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=900, help="maximum body tokens per chunk"
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=100,
        help="paragraph overlap inside one section",
    )
    parser.add_argument("--tokenizer", help="optional local HuggingFace tokenizer path")
    parser.add_argument(
        "--no-md", action="store_true", help="do not write per-patent cleaned Markdown"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.max_tokens <= 0
        or args.overlap_tokens < 0
        or args.overlap_tokens >= args.max_tokens
    ):
        raise SystemExit("require max_tokens > 0 and 0 <= overlap_tokens < max_tokens")
    count_tokens, tokenizer_name = make_counter(args.tokenizer)
    patent_dirs = list(iter_patent_dirs(args.input.resolve()))
    if not patent_dirs:
        raise SystemExit(f"No MinerU patent output found under {args.input}")
    output_dir = args.output.expanduser().resolve()
    _recover_interrupted_output(output_dir)

    all_records: list[dict] = []
    markdown_documents: dict[str, str] = {}
    report: list[dict] = []
    failures = 0
    for folder in patent_dirs:
        try:
            metadata, sections = parse_patent(folder)
            records = build_records(
                metadata,
                sections,
                args.max_tokens,
                args.overlap_tokens,
                count_tokens,
                tokenizer_name,
            )
            all_records.extend(records)
            if not args.no_md:
                markdown_documents[metadata["reference"]] = section_markdown(
                    metadata, sections
                )
            report.append(
                {
                    "reference": metadata["reference"],
                    "patent_title": metadata["patent_title"],
                    "abstract_found": bool(metadata["abstract"]),
                    "sections": len(sections),
                    "chunks": len(records),
                    "section_types": dict(Counter(s.section_type for s in sections)),
                }
            )
            print(
                f"[OK] {metadata['reference']}: {len(sections)} sections, {len(records)} chunks"
            )
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {folder.name}: {exc}", file=sys.stderr)

    summary = {
        "input": str(args.input.resolve()),
        "patents": len(report),
        "failures": failures,
        "chunks": len(all_records),
        "max_tokens": args.max_tokens,
        "overlap_tokens": args.overlap_tokens,
        "tokenizer": tokenizer_name,
        "documents": report,
    }
    if failures:
        print(
            "Parsing had failures; existing parser outputs were left unchanged.",
            file=sys.stderr,
        )
        return 1

    jsonl_path = _commit_outputs(
        output_dir,
        summary,
        all_records,
        markdown_documents,
        write_markdown=not args.no_md,
    )
    print(f"Wrote {len(all_records)} chunks to {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
