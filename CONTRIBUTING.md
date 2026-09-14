# Contributing

Start with an issue describing the research or engineering problem and a reproducible example. Keep discovery, evidence investigation and verification boundaries explicit.

## Offline development

From the repository root:

```bash
python -m pip install numpy PyYAML python-dotenv
python -m pip install --no-deps -e .
python -m unittest discover -s patent_coating/tests -v
```

The lightweight environment does not install model runtimes. Use Linux or a Windows environment with symlink-creation permission for the external-symlink test; report setup errors rather than silently skipping security checks.

## Pull requests

- Keep changes scoped and preserve Python import compatibility.
- Add deterministic tests for behavior changes, using synthetic data and fake clients.
- Do not call paid APIs, download models or write to production in automated tests.
- Never commit credentials, raw corpora, generated indexes, model weights or private traces.
- Include the commands run and their actual outcomes; label untested paths.
- Do not turn a provider response or passing unit test into an extraction-accuracy claim.

The existing Python package is `patent_coating`; the public project and distribution are CoatWeave Harness and `coat-weave-harness`.
