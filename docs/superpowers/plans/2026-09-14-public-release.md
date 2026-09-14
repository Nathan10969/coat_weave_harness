# CoatWeave Harness Public Release Implementation Plan

> For agentic workers: use subagent-driven-development for bounded source verification and Git publication; keep documentation changes separate.

**Goal:** Publish the approved professional, honest public retrieval foundation.

**Architecture:** Root documentation surrounds the unchanged `patent_coating` Python package. Build metadata exposes its existing CLI without reorganizing business modules. Local audit receipts remain outside the public tree.

**Tech Stack:** Python, NumPy, Qwen-compatible embedding/reranking, OpenAI-compatible tool calling, unittest, GitHub, SVG.

## Task 1: Safe source import

- [x] Verify ZIP identity and entry paths; import only `.py`, `requirements.txt`, launcher, and safe template/configuration into `patent_coating/`.
- [x] Exclude all corpora, secrets and generated artifacts; preserve original archive unchanged.
- [x] Replace personal model paths with local `models/` examples; put MIT, ignore rules and build metadata at repository root.

## Task 2: Public documentation

- [x] Create `README.md`, `README.zh-CN.md`, `docs/TOOLS.md`, `docs/ROADMAP.md`, `CONTRIBUTING.md`, `SECURITY.md` and original SVG assets.
- [x] Describe actual CLI configuration 10-to-5 patent selection, representative-chunk reranking, candidate-full-text tools and prompt-only evidence constraints.
- [x] Create synthetic Markdown examples and an offline tool walkthrough; retain original tests unchanged.

## Task 3: Verification

- [x] Run `python -m unittest discover -s patent_coating/tests -v` from repository root: original 16/16 tests pass on WSL/Linux without model calls; Windows symlink-permission error retained.
- [x] Run `python -m patent_coating.patent_rag_app --help`; verify editable installation and CLI entrypoint in an isolated environment.
- [x] Parse all Python and SVG files, check local Markdown references and execute synthetic file tools.
- [x] Run redacted Gitleaks directory scan and inspect publication files for forbidden data; inspect the staged tree again before push.

## Task 4: Publication

- [ ] Create public repository only if absent, set description/topics, commit as Nathan10969 with real date and push normal `main`.
- [ ] Verify remote SHA, README, MIT and visuals; save local sanitized audit receipt and report verification limits.
