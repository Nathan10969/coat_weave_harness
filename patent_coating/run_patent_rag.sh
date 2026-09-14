#!/usr/bin/env bash
set -euo pipefail

patent_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patent_conda_env="${PATENT_RAG_CONDA_ENV:-langchain}"

cd "${patent_project_dir}/.."
if [[ "${CONDA_DEFAULT_ENV:-}" == "${patent_conda_env}" ]]; then
  exec python -m patent_coating.patent_rag_app "$@"
fi
exec conda run --no-capture-output -n "${patent_conda_env}" \
  python -m patent_coating.patent_rag_app "$@"
