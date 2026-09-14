#!/usr/bin/env python3
"""MinerU parsing -> embedding search -> reranking -> tool-agent CLI."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml
from dotenv import load_dotenv

from .patent_rag_parser import iter_patent_dirs, main as parse_patents
from .search.agentic_retrieval import (
    prepare_agentic_workspace,
    retrieve_document_candidate_stages,
    stream_agentic_research,
)
from .search.embedding_index import EmbeddingIndex, inspect_index, load_records


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "patent_rag_config.yaml"
PATH_KEYS = ("mineru_root", "parsed_dir", "index_dir", "workspace_root", "env_file")


def load_config(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a YAML object: {config_path}")
    for section in ("paths", "models", "parsing", "index", "retrieval", "agent"):
        if not isinstance(data.get(section), dict):
            raise ValueError(f"Configuration lacks section: {section}")

    for key in PATH_KEYS:
        if key not in data["paths"]:
            raise ValueError(f"Configuration lacks paths.{key}")
        value = Path(os.path.expandvars(str(data["paths"][key]))).expanduser()
        if not value.is_absolute():
            value = config_path.parent / value
        data["paths"][key] = str(value.resolve())
    data["_config_path"] = str(config_path)
    load_dotenv(data["paths"]["env_file"], override=False)

    retrieval = data["retrieval"]
    candidate_count = int(retrieval["candidate_patents"])
    final_count = int(retrieval["final_patents"])
    if candidate_count <= 0 or not 0 < final_count <= candidate_count:
        raise ValueError("Require candidate_patents >= final_patents > 0")
    return data


def _torch_runtime(config: dict[str, Any]):
    import torch

    device = str(config["models"].get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configuration requires CUDA, but CUDA is unavailable")
    dtype_name = str(config["models"].get("dtype", "bfloat16")).lower()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return torch, device, dtype


def load_embedding_model(config: dict[str, Any]):
    from sentence_transformers import SentenceTransformer

    _, device, dtype = _torch_runtime(config)
    model_path = str(config["models"]["embedding_path"])
    kwargs = {
        "trust_remote_code": True,
        "device": device,
        "model_kwargs": {"dtype": dtype},
    }
    try:
        encoder = SentenceTransformer(model_path, **kwargs)
    except TypeError:
        kwargs.pop("model_kwargs")
        encoder = SentenceTransformer(model_path, **kwargs)
        encoder.to(dtype=dtype)
    encoder.max_seq_length = int(config["models"]["embedding_max_seq_length"])
    return encoder


def load_reranker(config: dict[str, Any]):
    from .reranker.reranker import QwenReranker

    _, device, dtype = _torch_runtime(config)
    return QwenReranker(
        str(config["models"]["reranker_path"]),
        device=device,
        max_length=int(config["models"]["reranker_max_length"]),
        dtype=dtype,
    )


def create_llm_client(config: dict[str, Any]):
    from openai import OpenAI

    env_name = str(config["models"]["api_key_env"])
    api_key = os.getenv(env_name, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing {env_name}; configure it in {config['paths']['env_file']}"
        )
    return OpenAI(
        api_key=api_key,
        base_url=str(config["models"]["llm_base_url"]),
    )


def _records_path(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["parsed_dir"]) / "patent_chunks.jsonl"


def _release_cuda(config: dict[str, Any]) -> None:
    gc.collect()
    torch, _, _ = _torch_runtime(config)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_corpus(config: dict[str, Any]) -> int:
    input_root = Path(config["paths"]["mineru_root"])
    output_dir = Path(config["paths"]["parsed_dir"])
    print(f"[parse] MinerU root: {input_root}")
    code = parse_patents(
        [
            str(input_root),
            "--output",
            str(output_dir),
            "--max-tokens",
            str(config["parsing"]["max_tokens"]),
            "--overlap-tokens",
            str(config["parsing"]["overlap_tokens"]),
        ]
    )
    if code:
        raise RuntimeError(
            "Parsing failed; refusing to build an index from a partial corpus"
        )
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    print(
        f"[parse] done: {report['patents']} patents, {report['chunks']} chunks, "
        f"{report['failures']} failures"
    )
    return 0


def index_corpus(config: dict[str, Any], *, rebuild: bool) -> int:
    records_path = _records_path(config)
    index_dir = Path(config["paths"]["index_dir"])
    model_id = str(config["models"]["embedding_path"])
    max_seq_length = int(config["models"]["embedding_max_seq_length"])
    status = inspect_index(
        index_dir,
        records_path,
        model_id=model_id,
        max_seq_length=max_seq_length,
    )
    if status.valid and not rebuild:
        print(
            f"[index] ready: {status.rows} chunks x {status.dimension}; no rebuild needed"
        )
        return status.rows
    if status.exists and not rebuild:
        raise RuntimeError(
            f"Existing index is stale ({status.message}); pass --rebuild to replace it"
        )

    records = load_records(records_path)
    references = {str(record["reference"]) for record in records}
    print(f"[index] encoding {len(records)} chunks from {len(references)} patents...")
    encoder = load_embedding_model(config)
    index = EmbeddingIndex.build(
        records_path=records_path,
        index_dir=index_dir,
        encoder=encoder,
        model_id=model_id,
        max_seq_length=max_seq_length,
        batch_size=int(config["index"]["embedding_batch_size"]),
    )
    del encoder
    _release_cuda(config)
    print(
        f"[index] done: {len(index.records)} chunks x "
        f"{index.embeddings.shape[1]} dimensions (exact cosine)"
    )
    return len(index.records)


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    chunks = sorted(
        candidate.get("matched_chunks", []),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    return {
        "reference": candidate["reference"],
        "title": candidate.get("title"),
        "embedding_score": candidate.get("embedding_score"),
        "rerank_score": candidate.get("rerank_score"),
        "matched_chunk_count": len(chunks),
        "top_matched_sections": [
            {
                "score": chunk.get("score"),
                "section_path": chunk.get("section_path"),
                "chunk_index": chunk.get("chunk_index"),
            }
            for chunk in chunks[:3]
        ],
    }


def _new_workspace(config: dict[str, Any], query: str) -> Path:
    root = Path(config["paths"]["workspace_root"])
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return root / f"{stamp}-{digest}"


def run_question(
    config: dict[str, Any],
    query: str,
    *,
    show_tool_output: bool = False,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    client = create_llm_client(config)
    records_path = _records_path(config)
    index = EmbeddingIndex.load(
        records_path=records_path,
        index_dir=Path(config["paths"]["index_dir"]),
        model_id=str(config["models"]["embedding_path"]),
        max_seq_length=int(config["models"]["embedding_max_seq_length"]),
    )

    print("[retrieve] embedding query...")
    encoder = load_embedding_model(config)
    query_vector = encoder.encode(
        query,
        prompt_name="query",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    del encoder
    _release_cuda(config)

    print("[rerank] selecting unique patent documents...")
    reranker = load_reranker(config)
    retrieval = config["retrieval"]
    embedding_candidates, final_candidates = retrieve_document_candidate_stages(
        index=index,
        query=query,
        query_vector=query_vector.tolist(),
        reranker=reranker,
        candidate_documents=int(retrieval["candidate_patents"]),
        final_documents=int(retrieval["final_patents"]),
        chunk_pool_top_k=int(retrieval["chunk_pool_top_k"]),
        max_chunk_pool_top_k=int(retrieval["max_chunk_pool_top_k"]),
        representative_chunks=int(retrieval["representative_chunks"]),
        rerank_batch_size=int(retrieval["rerank_batch_size"]),
    )
    del reranker
    _release_cuda(config)

    print(f"[retrieve] {len(embedding_candidates)} embedding candidates")
    for rank, candidate in enumerate(embedding_candidates, 1):
        print(
            f"  E{rank:02d} {candidate['reference']} "
            f"score={float(candidate['embedding_score']):.6f} {candidate.get('title', '')}"
        )
    print(f"[rerank] {len(final_candidates)} final candidates")
    for rank, candidate in enumerate(final_candidates, 1):
        print(
            f"  R{rank:02d} {candidate['reference']} "
            f"score={float(candidate.get('rerank_score', 0.0)):.6f} "
            f"{candidate.get('title', '')}"
        )

    workspace = _new_workspace(config, query)
    manifest = prepare_agentic_workspace(
        final_candidates,
        cleaned_md_root=Path(config["paths"]["parsed_dir"]) / "md",
        workspace=workspace,
    )
    retrieval_trace = {
        "query": query,
        "embedding_candidates": [
            _candidate_summary(item) for item in embedding_candidates
        ],
        "reranked_candidates": [_candidate_summary(item) for item in final_candidates],
        "manifest_path": manifest["manifest_path"],
    }
    (workspace / "retrieval.json").write_text(
        json.dumps(retrieval_trace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[agent] tool sandbox contains {len(final_candidates)} complete patents")
    final_result = None
    agent_config = config["agent"]
    for event in stream_agentic_research(
        client=client,
        model=str(config["models"]["llm_model"]),
        query=query,
        manifest=manifest,
        max_rounds=int(agent_config["max_rounds"]),
        max_tool_calls=int(agent_config["max_tool_calls"]),
        max_total_tool_output_chars=int(agent_config["max_total_tool_output_chars"]),
        request_retries=int(agent_config["request_retries"]),
        tool_retries=int(agent_config["tool_retries"]),
        retry_base_seconds=float(agent_config["retry_base_seconds"]),
        require_tool_use=bool(agent_config["require_tool_use"]),
        model_kwargs={
            "temperature": float(agent_config["temperature"]),
            "max_tokens": int(agent_config["max_tokens"]),
        },
    ):
        if event.kind == "round_start":
            print(f"\n[agent round {event.data['round']}]", flush=True)
        elif event.kind == "assistant_delta":
            print(event.data["text"], end="", flush=True)
        elif event.kind == "tool_start":
            print(
                f"\n[tool:start] {event.data['name']} {event.data['arguments']}",
                flush=True,
            )
        elif event.kind == "tool_end":
            output = str(event.data.get("output", ""))
            if not show_tool_output and len(output) > 400:
                output = output[:400].replace("\n", " ") + " ..."
            print(f"\n[tool:end] {output}", flush=True)
        elif event.kind == "retry":
            print(
                f"\n[retry:{event.data['scope']}] attempt={event.data['attempt']} "
                f"sleep={event.data['delay_seconds']:.2f}s",
                flush=True,
            )
        elif event.kind == "warning":
            print(f"\n[warning] {event.data['message']}", flush=True)
        elif event.kind == "final":
            final_result = event.data["result"]
    if final_result is None:
        raise RuntimeError("Agent ended without a final answer")

    output_path = trace_path.resolve() if trace_path else workspace / "agent_trace.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final_result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[done] tool calls={final_result.tool_call_count}")
    print(f"[done] workspace={workspace}")
    print(f"[done] trace={output_path}")
    return {
        "answer": final_result.answer,
        "workspace": str(workspace),
        "trace": str(output_path),
        "tool_call_count": final_result.tool_call_count,
    }


def print_status(config: dict[str, Any]) -> bool:
    input_root = Path(config["paths"]["mineru_root"])
    patent_dirs = list(iter_patent_dirs(input_root)) if input_root.is_dir() else []
    records_path = _records_path(config)
    records = load_records(records_path) if records_path.is_file() else []
    references = {str(record["reference"]) for record in records}
    md_dir = Path(config["paths"]["parsed_dir"]) / "md"
    md_count = len(list(md_dir.glob("*.md"))) if md_dir.is_dir() else 0
    index_status = inspect_index(
        Path(config["paths"]["index_dir"]),
        records_path,
        model_id=str(config["models"]["embedding_path"]),
        max_seq_length=int(config["models"]["embedding_max_seq_length"]),
    )
    print(f"MinerU patents: {len(patent_dirs)}")
    print(
        f"Parsed corpus: {len(references)} patents / {len(records)} chunks / {md_count} Markdown"
    )
    print(
        f"Embedding index: {index_status.message} / "
        f"{index_status.rows} rows x {index_status.dimension}"
    )
    print(
        f"Retrieval: exact cosine -> {config['retrieval']['candidate_patents']} patents "
        f"-> reranker -> {config['retrieval']['final_patents']} patents -> tools"
    )
    print(f"LLM: {config['models']['llm_model']} @ {config['models']['llm_base_url']}")
    ready = (
        bool(patent_dirs)
        and len(patent_dirs) == len(references) == md_count
        and index_status.valid
        and index_status.rows == len(records)
    )
    print(f"Pipeline: {'ready' if ready else 'not ready'}")
    return ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("parse", help="clean and chunk the MinerU corpus")
    index_parser = subparsers.add_parser("index", help="embed parsed chunks")
    index_parser.add_argument("--rebuild", action="store_true")
    build = subparsers.add_parser("build", help="parse, then build the embedding index")
    build.add_argument("--rebuild", action="store_true")
    ask = subparsers.add_parser("ask", help="embedding -> reranker -> tool-agent")
    ask.add_argument("--query", "-q", required=True)
    ask.add_argument("--show-tool-output", action="store_true")
    ask.add_argument("--trace", type=Path)
    subparsers.add_parser("status", help="validate corpus and embedding index")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "parse":
        return parse_corpus(config)
    if args.command == "index":
        index_corpus(config, rebuild=bool(args.rebuild))
        return 0
    if args.command == "build":
        parse_corpus(config)
        index_corpus(config, rebuild=bool(args.rebuild))
        return 0
    if args.command == "ask":
        run_question(
            config,
            args.query,
            show_tool_output=bool(args.show_tool_output),
            trace_path=args.trace,
        )
        return 0
    if args.command == "status":
        return 0 if print_status(config) else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
