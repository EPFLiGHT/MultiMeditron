#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Optional: choose task type (skin | ophthalmology)
: "${TASK_TYPE:=skin}"
export TASK_TYPE

# Make sure we are NOT in estimate-only mode
export ESTIMATE_ONLY=false
unset NB_SAMPLES  # full dataset

# 1) Build full batches (writes to config.BATCHES_DIR unless BATCHES_DIR is set)
python make_batches.py

# 2) Submit all parts
python submit_batches.py

# 3) Collect everything (poll until ready)
python collect_all.py
