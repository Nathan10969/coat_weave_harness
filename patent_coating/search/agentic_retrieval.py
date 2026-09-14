"""Document-level candidate ranking and sandbox preparation for patent RAG."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

from .agent_runtime import AgentEvent, RegisteredTool, ToolCallingAgent, ToolRegistry
from .embedding_index import EmbeddingIndex, SearchHit


AGENTIC_PATENT_RETRIEVAL_PROMPT = r"""
You are an evidence-first patent research agent. You work inside a restricted
filesystem sandbox containing only the candidate patents selected by embedding
retrieval and a reranker.

## Research question
{query}

## Sandbox
- Candidate root: {candidate_root}
- Candidate manifest: {manifest_path}
- Candidate count: {candidate_count}

## Available tools
You have three actual read-only function tools. Call them yourself; do not only
describe a shell command:
1. `find(pattern, max_results)`: find sandbox files by glob.
2. `grep(pattern, path, regex, case_sensitive, file_glob, max_results)`:
   grep files and return `file:line:text` matches.
3. `read(path, start_line, end_line)`: sed-like targeted line reading.

Do not use network access, shell scripting, Python, file writes, or knowledge
outside the candidate corpus. Do not infer visual facts from an image path; you
may use only its caption unless a separate vision tool is supplied.

## Required search procedure
1. Call `read` on the manifest first. Treat embedding/reranker ranks as
   priors, not proof.
2. Call `find` to enumerate the available `.md` patents. Never search outside
   `{candidate_root}`.
3. Build a compact search plan from the question:
   - exact technical terms, formulas, material names and abbreviations;
   - spelling variants, synonyms, singular/plural forms and useful word stems;
   - patent section signals such as Abstract, Background, Detailed Description,
     Examples, Comparative Examples, Tables, Results and Claims.
4. Call case-insensitive `grep` across all candidate files. Start broad, then
   refine with regex. MinerU OCR may join words, so also search distinctive
   stems and nearby technical terms instead of relying on one exact phrase.
5. Call `read` to read context around every promising match. Expand the range
   until the governing heading, experimental conditions, comparison baseline,
   units and result are all visible. Do not answer from grep snippets alone.
6. For performance, formulation or "best" questions, verify the evidence in
   Examples/Tables and distinguish inventive examples from comparative ones.
   For scope questions, verify Claims. For definitions, verify Detailed
   Description. Cross-check important conclusions in more than one passage or
   patent when the corpus permits.
7. Continue searching until either the evidence is sufficient or at least two
   distinct search strategies fail. Record important contradictions and missing
   information; never fill gaps with assumptions.

## Evidence and citation rules
- Every technical claim, numeric value and comparison must cite a source as
  `[PUBLICATION_NUMBER, lines START-END]`.
- Obtain line numbers from tool output. Include the complete patent title on
  first citation of a document.
- A citation must support the sentence immediately before it.
- Clearly label statements found only in an abstract or only in claims.
- Never cite the manifest or a JSONL chunk as substantive patent evidence when
  the full Markdown is available.

## Final response
Answer the research question directly, then provide:
1. the supported conclusion;
2. the strongest formulations/examples/results, including conditions and units;
3. relevant claim scope or limitations;
4. contradictions, uncertainty and evidence gaps;
5. a short source list containing publication number and complete title.

If the candidate corpus does not support an answer, say so explicitly and list
the searches performed. Be concise, but do not omit evidence necessary to audit
the conclusion.
""".strip()


SANDBOX_FILE_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Find files inside the candidate patent sandbox using a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Sandbox-relative glob, normally docs/**/*.md or manifest.json.",
                        "default": "docs/**/*.md",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "default": 200,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search candidate patent files and return matching lines with file names and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Literal text or regular expression.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Sandbox-relative file or directory to search.",
                        "default": "docs",
                    },
                    "regex": {"type": "boolean", "default": True},
                    "case_sensitive": {"type": "boolean", "default": False},
                    "file_glob": {"type": "string", "default": "*.md"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "default": 200,
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a line-numbered range from one sandbox file, like sed -n START,ENDp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Sandbox-relative file path.",
                    },
                    "start_line": {"type": "integer", "minimum": 1, "default": 1},
                    "end_line": {"type": "integer", "minimum": 1, "default": 200},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


class SandboxPathError(ValueError):
    """Raised when a tool attempts to escape the candidate workspace."""


class SandboxFileTools:
    """Safe Python implementations of find, grep and sed-like file reading."""

    allowed_suffixes = {".md", ".json"}

    def __init__(self, root: Path, max_read_lines: int = 400) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.max_read_lines = max(1, max_read_lines)

    def _inside_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
            return True
        except ValueError:
            return False

    def _resolve(self, value: str, *, must_be_file: bool = False) -> Path:
        requested = Path(value)
        candidate = requested if requested.is_absolute() else self.root / requested
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Sandbox path does not exist: {value}") from exc
        if not self._inside_root(resolved):
            raise SandboxPathError(f"Path escapes sandbox: {value}")
        if must_be_file and not resolved.is_file():
            raise FileNotFoundError(f"Not a sandbox file: {value}")
        if resolved.is_file() and resolved.suffix.lower() not in self.allowed_suffixes:
            raise SandboxPathError(
                f"File type is not readable by sandbox tools: {resolved.suffix}"
            )
        return resolved

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _files(self, target: Path, file_glob: str) -> Iterable[Path]:
        if target.is_file():
            yield target
            return
        if ".." in Path(file_glob).parts:
            raise SandboxPathError("file_glob cannot contain '..'")
        for path in sorted(target.rglob(file_glob)):
            if not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            if (
                self._inside_root(resolved)
                and resolved.suffix.lower() in self.allowed_suffixes
            ):
                yield resolved

    def file_find(self, pattern: str = "docs/**/*.md", max_results: int = 200) -> str:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise SandboxPathError("find pattern must stay inside the sandbox")
        limit = min(max(1, int(max_results)), 500)
        matches = []
        for path in sorted(self.root.glob(pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            if (
                not self._inside_root(resolved)
                or resolved.suffix.lower() not in self.allowed_suffixes
            ):
                continue
            matches.append(self._relative(resolved))
            if len(matches) >= limit:
                break
        if not matches:
            return "NO_FILES_FOUND"
        return "\n".join(matches)

    def file_grep(
        self,
        pattern: str,
        path: str = "docs",
        regex: bool = True,
        case_sensitive: bool = False,
        file_glob: str = "*.md",
        max_results: int = 200,
    ) -> str:
        if not pattern or len(pattern) > 1000:
            raise ValueError("grep pattern must contain 1-1000 characters")
        target = self._resolve(path)
        expression = pattern if regex else re.escape(pattern)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(expression, flags)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {exc}") from exc
        limit = min(max(1, int(max_results)), 500)
        matches = []
        for file_path in self._files(target, file_glob):
            for line_number, line in enumerate(
                file_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if compiled.search(line):
                    compact = line.strip()
                    if len(compact) > 800:
                        compact = compact[:797] + "..."
                    matches.append(
                        f"{self._relative(file_path)}:{line_number}:{compact}"
                    )
                    if len(matches) >= limit:
                        return "\n".join(matches) + "\n[RESULT_LIMIT_REACHED]"
        return "\n".join(matches) if matches else "NO_MATCHES"

    def file_read(self, path: str, start_line: int = 1, end_line: int = 200) -> str:
        file_path = self._resolve(path, must_be_file=True)
        start = max(1, int(start_line))
        end = max(start, int(end_line))
        if end - start + 1 > self.max_read_lines:
            end = start + self.max_read_lines - 1
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if start > len(lines):
            return f"{self._relative(file_path)} has {len(lines)} lines; requested start line {start}."
        end = min(end, len(lines))
        body = "\n".join(
            f"{number}: {lines[number - 1]}" for number in range(start, end + 1)
        )
        return f"FILE {self._relative(file_path)} LINES {start}-{end}\n{body}"


def _rerank_documents(
    query: str,
    candidates: List[Dict[str, Any]],
    reranker: Any,
    final_documents: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    if reranker is None:
        return candidates[:final_documents]
    scored: List[Dict[str, Any]] = []
    for start in range(0, len(candidates), max(1, batch_size)):
        batch = candidates[start : start + max(1, batch_size)]
        documents = [candidate["rerank_text"] for candidate in batch]
        scores = reranker([query] * len(documents), documents)
        if len(scores) != len(batch):
            raise ValueError("Reranker returned a different number of scores")
        for score, candidate in zip(scores, batch):
            candidate = dict(candidate)
            candidate["rerank_score"] = float(score)
            scored.append(candidate)
    scored.sort(key=lambda item: item["rerank_score"], reverse=True)
    return scored[:final_documents]


def retrieve_document_candidate_stages(
    index: EmbeddingIndex,
    query: str,
    query_vector: Sequence[float],
    reranker: Any,
    *,
    candidate_documents: int = 100,
    final_documents: int = 50,
    chunk_pool_top_k: int = 500,
    max_chunk_pool_top_k: int = 5000,
    representative_chunks: int = 3,
    rerank_batch_size: int = 8,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Retrieve chunks, aggregate unique patents, then rerank patent candidates.

    A chunk index cannot directly return N unique documents.  We therefore
    over-retrieve chunks, aggregate by publication number, and use the best
    passages plus the complete patent title as the document rerank text.

    Returns both stages so callers can audit that ``candidate_documents`` and
    ``final_documents`` count publication references, not chunks.
    """
    if candidate_documents <= 0 or final_documents <= 0:
        raise ValueError("candidate_documents and final_documents must be positive")
    if final_documents > candidate_documents:
        raise ValueError("final_documents cannot exceed candidate_documents")
    if chunk_pool_top_k <= 0 or max_chunk_pool_top_k < chunk_pool_top_k:
        raise ValueError("Require max_chunk_pool_top_k >= chunk_pool_top_k > 0")
    if representative_chunks <= 0 or rerank_batch_size <= 0:
        raise ValueError("representative_chunks and rerank_batch_size must be positive")

    # A popular patent can occupy many top chunk hits. Search the exact index
    # once, then expand the visible prefix until enough unique patents appear.
    max_pool = min(max_chunk_pool_top_k, len(index.records))
    all_hits = index.search(query_vector, top_k=max_pool)
    pool_limit = min(max(candidate_documents, chunk_pool_top_k), len(all_hits))
    while pool_limit < len(all_hits):
        unique_references = {hit.reference for hit in all_hits[:pool_limit]}
        if len(unique_references) >= candidate_documents:
            break
        pool_limit = min(pool_limit * 2, len(all_hits))
    hits: List[SearchHit] = all_hits[:pool_limit]
    grouped: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        reference = str(hit.reference)
        metadata = hit.metadata or {}
        candidate = grouped.setdefault(
            reference,
            {
                "reference": reference,
                "title": metadata.get("title")
                or metadata.get("patent_title")
                or reference,
                "embedding_score": hit.score,
                "matched_chunks": [],
            },
        )
        candidate["embedding_score"] = max(candidate["embedding_score"], hit.score)
        candidate["matched_chunks"].append(
            {
                "score": hit.score,
                "section_path": metadata.get("section_path", []),
                "chunk_index": metadata.get("chunk_index"),
                "text": hit.text,
            }
        )

    candidates = sorted(
        grouped.values(), key=lambda item: item["embedding_score"], reverse=True
    )[:candidate_documents]
    for candidate in candidates:
        chunks = sorted(
            candidate["matched_chunks"], key=lambda item: item["score"], reverse=True
        )[: max(1, representative_chunks)]
        passages = []
        for chunk in chunks:
            path = chunk["section_path"]
            if isinstance(path, list):
                path = " > ".join(str(part) for part in path)
            passages.append(f"Section: {path}\n{chunk['text']}")
        candidate["rerank_text"] = (
            f"Patent: {candidate['reference']}\nTitle: {candidate['title']}\n\n"
            + "\n\n".join(passages)
        )
    final_candidates = _rerank_documents(
        query, candidates, reranker, final_documents, rerank_batch_size
    )
    return candidates, final_candidates


def prepare_agentic_workspace(
    candidates: Sequence[Dict[str, Any]],
    cleaned_md_root: Path,
    workspace: Path,
) -> Dict[str, Any]:
    """Copy complete candidate Markdown files into a scoped read-only corpus.

    Existing unrelated files are never removed. Candidate Markdown and the
    manifest are overwritten deterministically when the same workspace is used.
    """
    cleaned_md_root = cleaned_md_root.resolve()
    workspace = workspace.resolve()
    docs_dir = workspace / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    references = [str(candidate["reference"]) for candidate in candidates]
    if any(Path(reference).name != reference for reference in references):
        raise ValueError("Candidate reference must be a plain publication identifier")
    expected_files = {f"{reference}.md" for reference in references}
    stale_files = [
        path for path in docs_dir.glob("*.md") if path.name not in expected_files
    ]
    if stale_files:
        names = ", ".join(path.name for path in stale_files[:5])
        raise FileExistsError(
            f"Workspace contains stale candidate files ({names}); use a fresh per-query workspace"
        )
    manifest_documents = []
    for rank, candidate in enumerate(candidates, 1):
        reference = str(candidate["reference"])
        source = (cleaned_md_root / f"{reference}.md").resolve()
        try:
            source.relative_to(cleaned_md_root)
        except ValueError as exc:
            raise ValueError(
                f"Candidate source escapes Markdown root: {reference}"
            ) from exc
        if not source.is_file():
            raise FileNotFoundError(f"Missing cleaned patent Markdown: {source}")
        destination = docs_dir / source.name
        shutil.copy2(source, destination)
        manifest_documents.append(
            {
                "rank": rank,
                "reference": reference,
                "title": candidate.get("title", reference),
                "embedding_score": candidate.get("embedding_score"),
                "rerank_score": candidate.get("rerank_score"),
                "file": str(destination),
                "matched_sections": [
                    chunk.get("section_path", [])
                    for chunk in candidate.get("matched_chunks", [])
                ],
            }
        )
    manifest = {
        "candidate_count": len(manifest_documents),
        "candidate_root": str(docs_dir),
        "documents": manifest_documents,
    }
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_agentic_prompt(query: str, manifest: Dict[str, Any]) -> str:
    """Render the tool-using research prompt for a prepared workspace."""
    return AGENTIC_PATENT_RETRIEVAL_PROMPT.format(
        query=query,
        candidate_root=manifest["candidate_root"],
        manifest_path=manifest["manifest_path"],
        candidate_count=manifest["candidate_count"],
    )


def build_sandbox_tool_registry(root: Path) -> ToolRegistry:
    """Register the read-only patent tools."""
    sandbox = SandboxFileTools(root)
    registry = ToolRegistry()
    for schema in SANDBOX_FILE_TOOL_SCHEMAS:
        function = schema["function"]
        name = function["name"]
        handler = {
            "find": sandbox.file_find,
            "grep": sandbox.file_grep,
            "read": sandbox.file_read,
        }[name]
        registry.register(
            RegisteredTool(
                name=name,
                description=function["description"],
                parameters=function["parameters"],
                execute=handler,
            )
        )
    return registry


def create_patent_agent(
    client: Any,
    model: str,
    query: str,
    manifest: Dict[str, Any],
    *,
    max_rounds: int = 24,
    max_tool_calls: int = 80,
    max_total_tool_output_chars: int = 300_000,
    request_retries: int = 3,
    tool_retries: int = 2,
    retry_base_seconds: float = 1.0,
    require_tool_use: bool = True,
    model_kwargs: Dict[str, Any] | None = None,
) -> ToolCallingAgent:
    workspace_root = Path(manifest["manifest_path"]).resolve().parent
    return ToolCallingAgent(
        client=client,
        model=model,
        registry=build_sandbox_tool_registry(workspace_root),
        system_prompt=build_agentic_prompt(query, manifest),
        max_rounds=max_rounds,
        max_tool_calls=max_tool_calls,
        max_total_tool_output_chars=max_total_tool_output_chars,
        request_retries=request_retries,
        tool_retries=tool_retries,
        retry_base_seconds=retry_base_seconds,
        require_tool_use=require_tool_use,
        model_kwargs=model_kwargs,
    )


def stream_agentic_research(
    client: Any,
    model: str,
    query: str,
    manifest: Dict[str, Any],
    **kwargs: Any,
) -> Iterator[AgentEvent]:
    agent = create_patent_agent(client, model, query, manifest, **kwargs)
    yield from agent.stream(
        "Begin the patent investigation now. Call the sandbox tools yourself, "
        "then return the evidence-backed final answer."
    )
