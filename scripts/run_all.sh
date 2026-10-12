#!/usr/bin/env bash
# Re-run every study end to end: executes notebooks, regenerates figures, results JSON and README.
set -euo pipefail
cd "$(dirname "$0")/.."
for f in research/*.py; do
  b=$(basename "$f" .py)
  jupytext --to ipynb -o "notebooks/$b.ipynb" "$f" >/dev/null
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 "notebooks/$b.ipynb"
done
python scripts/build_readme.py
