# Public Pi Harness Implementation Plan

**Goal:** Publishable service code with neutral configuration and offline verification.

**Architecture:** Python retrieval remains independent. A pinned Pi SDK drives
the migrated HTTP adapter and explicit patent-search extension. State and
credentials stay outside the tracked public tree.

**Tech Stack:** TypeScript, Node.js, Pi SDK, Python, FastAPI, Milvus and optional TOS.

- [x] Migrate adapter and patent backend source, preserving regression coverage.
- [x] Add tests for nested tool metadata and workspace tool policy.
- [x] Implement metadata-only logging and explicit read/write/shell modes.
- [x] Remove internal identity, collection audits and storage defaults.
- [x] Document synthetic setup, authentication, state retention and boundaries.
- [x] Run offline regressions, compile and synthetic service smoke.
- [x] Scan public files and Git history; leave a local reviewable patch.

See `docs/PUBLICATION_CHECKS.md` for scoped results and remaining limitations.
