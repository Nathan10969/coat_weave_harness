# File tools and investigation contract

These are registered Python function tools, not shell commands. `ToolRegistry` pairs a JSON tool schema with an execution callback; `ToolCallingAgent` consumes streamed tool calls, executes them, appends results to the message history and continues.

## Discovery before investigation

The outer CLI embeds the query, searches the chunk index, aggregates hits by publication reference and reranks candidate titles plus representative chunks. With the supplied configuration, up to 10 publications enter reranking and up to 5 enter the workspace. The agent itself cannot call this outer retrieval pipeline to expand the candidates yet.

## Interfaces

| Tool | Parameters | Output and limit |
|---|---|---|
| `find` | `pattern`, `max_results` | Relative file paths; default glob `docs/**/*.md`; up to 500 results |
| `grep` | `pattern`, `path`, `regex`, `case_sensitive`, `file_glob`, `max_results` | `file:line:text`; default searches `docs`, case-insensitively with regex; up to 500 matches |
| `read` | `path`, `start_line`, `end_line` | Line-numbered text; inclusive range; default 1–200, maximum 400 lines per call |

`grep` scans file text beyond retrieved chunks, but matches one line at a time. Matching lines longer than 800 characters are truncated; a result-limit marker indicates incomplete output. `read` supplies the surrounding context and can be called repeatedly to inspect other ranges.

## Access boundary

Resolved paths must remain inside the workspace, and readable suffixes are `.md` and `.json`. Escape paths and symlinks resolving outside the root are rejected. The outer application makes a fresh per-question workspace.

The actual tool root is the manifest's parent workspace, not a per-file allowlist. Other allowed text files inside a reused workspace could be visible. Use fresh workspaces and keep unrelated files out. These checks are not container or OS-level isolation.

## Runtime budget

The supplied defaults are 24 investigation rounds, 80 attempted tool calls and 300,000 accumulated tool-output characters. A final no-tools answer request may follow budget exhaustion. Each individual output is capped at 60,000 characters. Request/tool retry counts are configurable; retries do not provide recovery from every possible mid-stream failure.

`require_tool_use` only rejects answers with zero attempted tool calls. It does not require every named tool, successful calls, manifest-first reading or sufficient evidence. Failed calls can still count as attempts.

## Evidence procedure: prompted, not automatically certified

The agent prompt asks for technical terms, synonyms, OCR word stems, context expansion, experiment/comparison checks and `[PUBLICATION_NUMBER, lines START-END]` citations. It asks the model not to answer from search snippets alone or infer visual facts from image paths.

No current validator checks claim support, citation accuracy, conditions/units completeness or exhaustive search. The model interprets distributed evidence; the code does not build a formulation–condition–result graph. The cited lines refer to cleaned Markdown, not original PDF page coordinates.

## Security caveats

Untrusted queries and document text can attempt prompt injection. Python regex execution has no dedicated timeout. Character/match limits are not CPU or memory limits. Model loading retains the original `trust_remote_code=True` behavior: only use trusted, reviewed model sources. See [SECURITY.md](../SECURITY.md).
