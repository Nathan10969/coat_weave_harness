# Toward a complete coating research agent

The intended destination is a complete, tool-using coating research agent—not only a patent question-answering endpoint. Codex and Claude Code inspire the task-workspace and inspectable tool-interaction model; they are not integrated runtimes or capability benchmarks for this release.

Everything under **planned** is future work, not implemented functionality or a promised delivery date.

## Current foundation

- Chapter-aware patent text preparation and exact chunk retrieval.
- Unique-publication aggregation and representative-chunk reranking.
- Selected-patent full-text investigation through `find`, `grep`, `read`.
- Streaming tool calls, per-question workspaces and execution traces.

## Planned: make retrieval a reusable tool

- Define a stable retrieval request/result contract for the outer agent.
- Return selected documents, source identity and auditable retrieval metadata.
- Allow query refinement and explicit candidate expansion when evidence is insufficient.
- Bind full-text content versions to the retrieval artifacts.

## Planned: stronger evidence and evaluation

- Check source membership, line ranges and citation formatting.
- Evaluate whether cited passages support claims, including conditions, units and comparisons.
- Build challenge cases for distributed evidence, candidate omissions, OCR and tables.
- Separate retrieval recall, evidence completeness and answer correctness from API success.

## Planned: the wider coating agent

- Task planning and inspectable progress in a research workspace.
- Reusable tools for literature/patent discovery and coating research artifacts.
- Explicit approval boundaries for network access, paid calls and external changes.
- Durable task state, reproducible evidence bundles and verification gates.

Domain workflows beyond patent retrieval require their own tool contracts and validation. Autonomous formulation design, experiment execution and production deployment are not capabilities of the current release.
