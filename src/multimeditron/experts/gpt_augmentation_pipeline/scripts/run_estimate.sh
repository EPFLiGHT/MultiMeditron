#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${TASK_TYPE:=skin}"
: "${NB_SAMPLES:=500}"   # pick a default sample size
export TASK_TYPE NB_SAMPLES

# 1) Build small-sample batches
python make_batches.py

# 2) Submit only the first part
export ESTIMATE_ONLY=true
python submit_batches.py

# 3) Collect + estimate
python collect_all.py
python estimate_price.py
