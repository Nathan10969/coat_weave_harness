# CoatWeave Harness public release design

Approved scope: publish an independent public repository named `Nathan10969/coat_weave_harness`, without changing retrieval business logic. Preserve Python package `patent_coating` and the supplied tests.

## Positioning

Vision: a complete tool-using coating research agent inspired by Codex and Claude Code's interaction model. Current implementation: chunk-based candidate patent retrieval followed by tool-driven investigation of selected patents' complete cleaned Markdown. Clearly distinguish roadmap from implemented capability.

## Public distribution

Import only source, tests and safe configuration from the supplied archive. Exclude `.env`, original PDFs, OCR data, parsed corpora, embeddings and generated runs. Replace personal model paths with portable examples. MIT copyright 2026 Nathan10969. No invented attribution, benchmark numbers or production acceptance.

## Documentation and visuals

English README and Chinese companion; original SVG hero and architecture; tool contracts and limitations; roadmap, contribution and security guidance. A small synthetic corpus illustrates file tools without third-party patent data or paid calls.

## Verification

Run the original offline unit tests, AST parsing, CLI help, synthetic tool walkthrough, package installation smoke, Gitleaks, documentation links and SVG parsing. Preserve discovered failures rather than repairing retrieval behavior silently. Verify remote main and public README/license after normal push. Do not change CoatWeave's existing snapshot repository or automation.
