# Public Pi service migration

Keep the existing Python retrieval CLI unchanged. Add `pi-harness/` containing
the session HTTP adapter, completion adapter, read-only patent-search extension
and Milvus/local-Markdown/TOS backend. Consume Pi 0.86.0 as a pinned package,
with upstream attribution, rather than copying the upstream monorepo.

The service defaults to isolated local state, loopback binding and read-only
tools. Writes are explicit opt-in and confined to workspace outputs. Shell
access is a separate opt-in and requires API authentication. Arbitrary project
extensions and account prompts are not discovered automatically.

Replace internal product prompts with neutral evidence-first instructions.
Remove production collection identities, audit totals, storage bucket names,
deployment accounts and host paths. No credentials, corpus, traces or old Git
history from the source snapshot enter this repository.

Metadata-only event logs retain the eleven-field contract. Both HTTP paths
must use the same parser, including nested `args.statement`. Raw SDK session
events remain authenticated local API data, not public artifacts.

Acceptance: inherited offline SDK/HTTP tests, new policy and logging tests,
Python retrieval and backend tests, TypeScript check/build, synthetic HTTP
smoke, tracked-tree and history secret scans. No live model, Milvus, storage or
paid API is contacted during acceptance. Do not push without a separate request.
