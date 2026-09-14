# Offline full-text tool walkthrough

This example is hand-authored and entirely synthetic. It is not a real patent, generated model transcript, benchmark or scientific result. No API, embedding model or GPU is required.

The small document deliberately puts its formulation, method and result in separate sections. The point is to inspect the **whole candidate text**, not to answer from an isolated match.

After the [lightweight installation](../README.md), start Python from the repository root:

```python
from pathlib import Path
from patent_coating.search.agentic_retrieval import SandboxFileTools

tools = SandboxFileTools(Path("examples/synthetic_workspace"))
print(tools.file_read("manifest.json"))
print(tools.file_find("docs/**/*.md"))
print(tools.file_grep("Sample A|Demo-UV|gloss", regex=True))
print(tools.file_read("docs/SYNTHETIC-A.md", start_line=5, end_line=22))
```

`grep` supplies locations; `read` exposes the formulation, invented protocol and fictional result with line numbers. A real agent would choose and execute these calls itself. This manual walkthrough demonstrates only the deterministic tools, not autonomous behavior or answer quality.

Files: [manifest](synthetic_workspace/manifest.json) and [synthetic full text](synthetic_workspace/docs/SYNTHETIC-A.md).
