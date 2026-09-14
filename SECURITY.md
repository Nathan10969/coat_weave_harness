# Security and data handling

This repository is research software, not an audited production security boundary.

## Reporting

For vulnerabilities or exposed secrets, use GitHub's private vulnerability reporting if it is enabled for this repository. Otherwise contact the maintainer through their GitHub profile without posting exploit details, credentials or private documents in a public issue. General hardening suggestions can be discussed publicly with synthetic examples.

## Before running

- Keep real API keys in local `.env` files, never in Git or command-line examples.
- Review the configured provider and data policy: `ask` sends question and patent evidence text to that service and may incur charges.
- Only load trusted model files. The unchanged model loader enables remote model code; this is not safe for arbitrary model sources.
- Use fresh per-query workspaces with no unrelated documents or secrets.
- Treat retrieved text and queries as untrusted content; prompt injection has not been comprehensively mitigated or evaluated.
- Path containment and read-only registered tools are not OS/container isolation.
- Regex output limits do not enforce execution-time or CPU limits.

## Public data policy

The public tree intentionally excludes original patents, OCR output, parsed corpora, generated embeddings, model weights, run traces and real `.env` files. Only explicitly synthetic examples are included. `.gitignore` is a guardrail, not a guarantee: inspect staged files and scan for secrets before publishing.

The MIT license does not grant rights to third-party patents, datasets, models or service terms. No automated dependency-vulnerability audit or production-security acceptance is claimed by this initial publication.
