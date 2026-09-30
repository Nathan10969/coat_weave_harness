<p align="center"><img src="docs/assets/hero.svg" alt="CoatWeave Harness — find patents, investigate evidence" width="100%" /></p>

<p align="center">
  <a href="https://github.com/Nathan10969/coat_weave_harness/actions/workflows/offline-tests.yml"><img src="https://github.com/Nathan10969/coat_weave_harness/actions/workflows/offline-tests.yml/badge.svg" alt="Offline CI tests" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-14b8a6" alt="MIT license" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3b82f6" alt="Python 3.10 or newer" />
  <img src="https://img.shields.io/badge/runtime-Pi%20SDK-334155" alt="Pi SDK service layer" />
</p>

# CoatWeave Harness

**Use chunks to find patents. Use full-text tools to investigate evidence.**

[中文说明](README.zh-CN.md) · [Pi service](pi-harness/README.md) · [Tool contracts](docs/TOOLS.md) · [Roadmap](docs/ROADMAP.md) · [Offline example](examples/README.md)

CoatWeave Harness aims to build a **complete tool-using coating research agent**, inspired by the interaction model of Codex and Claude Code: working within a task workspace, choosing tools, investigating evidence and making its work inspectable.

**The repository includes the Python retrieval foundation and a separate Pi SDK service layer.** The Python CLI combines chunk-level discovery with selected-patent full-text tools. `pi-harness/` adds native sessions, HTTP/SSE, a completion-compatible subset, an explicit Milvus patent tool, and optional workspace writing. It is not a complete autonomous coating R&D system. Codex and Claude Code are design inspirations, not integrations, affiliations or capability-equivalence claims.

Created and maintained by [Nathan10969](https://github.com/Nathan10969).

## Why go beyond retrieved chunks?

An evidence trail can be distributed across a patent: a formulation in an example, a test procedure elsewhere, a result in a table, and a comparison in another passage. A retrieved chunk can locate the document without containing everything needed to interpret the result.

This harness separates **document discovery** from **evidence investigation**. Once a patent is selected, the agent is no longer restricted to its retrieved chunks. It can search the complete cleaned Markdown, read surrounding sections and compare passages before composing an answer.

<img src="docs/assets/architecture.svg" alt="Chunk discovery followed by selected-patent full-text investigation; complete agent remains a roadmap goal" width="100%" />

The architecture describes implemented code paths, not a measured accuracy or production-readiness result.

## Pi Service Layer

The sanitized [Pi Harness service](pi-harness/README.md) carries the authorized
adapter and backend service code, using the official SDK as a pinned dependency.
It defaults to loopback access and read-only tools, with explicit opt-in writing
to workspace `outputs/`, isolated SDK state and 11-field metadata-only logs.
Production hosts, accounts, private prompts, collection audits, buckets, keys,
documents and runtime records are excluded.

The Milvus backend is independent of the NumPy CLI below; this update does not
wire the existing CLI or a hypergraph engine into Pi. Native sessions expose
tool events; the completion endpoint streams text and heartbeats only. See the
service guide for configuration, authentication, persistence and shell risks.

```bash
cd pi-harness
npm ci --ignore-scripts
npm run check
npm test
npm run build
npm start
```

Configure your own provider and database for real research. Initial greeting and
health checks are not model-connectivity tests. Publication checks are recorded
in [PUBLICATION_CHECKS.md](docs/PUBLICATION_CHECKS.md).

## Implemented today

The following table describes the existing Python CLI, not the separate Pi service.

| Layer | Implementation | Boundary |
|---|---|---|
| Document preparation | Consume MinerU JSON, with Markdown fallback; clean text and split within sections | No OCR engine or automatic OCR-quality validation |
| Candidate discovery | Qwen-compatible embeddings and NumPy exact cosine search; aggregate by publication reference | Chunk-level retrieval still determines which patents can enter |
| Candidate selection | Rerank patent titles plus representative matching chunks | Not full-document reranking |
| Evidence investigation | Actual `find`, `grep`, `read` function tools over selected full-text files | Selected cleaned Markdown, not the whole library or original PDF visuals |
| Runtime | Streaming tool-call loop, retries, budgets and inspectable traces | Single investigation loop; not persistent cross-session memory |
| Answers | Prompt-guided conditions, comparison checks and line-number citations | No automatic claim-support or citation-correctness validator |

The supplied CLI configuration selects up to **10 unique patent candidates**, then up to **5 final patents**. These are configurable publication counts, not chunk counts. The lower-level retrieval function has different defaults; the CLI explicitly passes configuration values.

The agent can revisit sections that never appeared in the initial chunk hits. However, it cannot retrieve a patent excluded by candidate selection: the current agent exposes only the three file tools, not a dynamic corpus-search tool.

## Run the offline checks first

Python 3.10+. The publication checks use an isolated Python environment and the existing offline tests. The lightweight setup below deliberately omits model-runtime dependencies; it is sufficient for tests, CLI help and synthetic file tools, **not real embedding or question answering**.

```bash
git clone git@github.com:Nathan10969/coat_weave_harness.git
cd coat_weave_harness
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy PyYAML python-dotenv
python -m pip install --no-deps -e .
python -m unittest discover -s patent_coating/tests -v
coat-weave-harness --help
```

See the [synthetic walkthrough](examples/README.md) to explore a formulation, test method and result in different sections without any model, GPU or paid API call. Its records are invented demonstrations, not patent data or scientific findings.

Initial publication verification: the unchanged suite passed **16/16 tests on WSL Ubuntu / Python 3.12**. On Windows, 15 passed and one errored while creating its symlink fixture because the account lacked permission; no test was modified or silently skipped. Editable installation and CLI help passed. These are offline engineering checks, not model-quality or real-ingestion acceptance. The included Linux CI workflow repeats the tests on Python 3.11 and 3.12; its result is separate from these local checks.

The [initial publication CI run](https://github.com/Nathan10969/coat_weave_harness/actions/runs/34808374527) passed for commit `6f7f3766`. The badge above reflects subsequent workflow status, not a model benchmark.

## Run with your own corpus and models

Install the full dependencies in an environment appropriate for your hardware:

```bash
python -m pip install -e .
cp patent_coating/.env.example patent_coating/.env
```

Edit [the configuration](patent_coating/patent_rag_config.yaml): provide your MinerU output, local embedding/reranker paths and an OpenAI-compatible tool-calling provider. Add the selected provider's key only to your local `.env`. The bundled provider values are examples, not a fresh compatibility or availability guarantee.

With the example relative model paths, run from the repository root. `paths.*` resolves relative to the configuration file; model paths are passed through unchanged. For other working directories or installed distributions, use your own explicit configuration and absolute model paths.

```bash
coat-weave-harness parse
coat-weave-harness index --rebuild
coat-weave-harness status
coat-weave-harness ask --query "Which examples report weathering performance, under what conditions?"
```

`build --rebuild` combines parsing and indexing. All commands also work as `python -m patent_coating.patent_rag_app ...`; the historical shell launcher remains available but assumes a configured Conda environment.

Real embedding/reranking loads local models and may require substantial compute. `ask` sends the question and retrieved patent text to the configured provider and may incur charges. Check data permissions and provider policies first. No model weights, corpus or production deployment are bundled.

## Inspect the work

Each question gets a fresh workspace containing selected full-text documents and:

- `manifest.json`: document identity and ranking metadata.
- `retrieval.json`: both candidate stages and matched-section summaries.
- `agent_trace.json`: final answer, message history and tool arguments/results.

The index fingerprints the source JSONL and vector file, records model identity and sequence length, and refuses stale reuse. This does not establish semantic correctness or bind the full-text Markdown to the index with its own content hash.

## Known limits

- Full-text access is not exhaustive reading: tool-output and round budgets apply.
- `require_tool_use` enforces a nonzero attempted-tool count, not successful use of all three tools or a verified manifest-first procedure.
- Search strategy, experiment/comparison checks and citation support are currently prompt requirements, not programmatic acceptance gates.
- Text tools use path-containment and file-type checks. This is not an OS/container security sandbox; untrusted documents and queries still require care.
- Tables and images depend on upstream parsing. Complex table structure is not fully recovered; image captions do not provide visual understanding.
- No retrieval-recall, answer-accuracy, autonomous research or production-readiness benchmark is claimed by this release.

## Toward a complete coating agent

The longer-term goal extends beyond patent QA: a coating research workspace with task planning, reusable tools and explicit verification. Pi now has an independent Milvus `patent_search` tool. Connecting the Python retrieval pipeline, hypergraph retrieval, stronger evidence gates and broader coating workflows remains planned. See [the roadmap](docs/ROADMAP.md).

## Development and license

The public project is **CoatWeave Harness**; Python imports remain `patent_coating`. See [contribution guidance](CONTRIBUTING.md) and [security guidance](SECURITY.md).

[MIT](LICENSE) · Copyright 2026 Nathan10969. See [third-party notices](THIRD_PARTY_NOTICES.md) for Pi attribution. The license does not cover third-party patents, model weights or service terms.
