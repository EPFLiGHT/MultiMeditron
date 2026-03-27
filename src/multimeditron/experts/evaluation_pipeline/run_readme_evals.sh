#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/lightscratch/users/cljordan/multimeditron"
EVAL_DIR="$ROOT_DIR/src/multimeditron/experts/evaluation_pipeline"
DEFAULT_MODEL="$ROOT_DIR/src/multimeditron/experts/models/generalist_expert_v1"

export PYTHONPATH="$EVAL_DIR:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-14}"
LINE_NUMBER="${LINE_NUMBER:-300}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/src/multimeditron/experts/eval_results/${MODEL_NAME}_readme_${TIMESTAMP}}"

RUN_BASIC="${RUN_BASIC:-1}"
RUN_HARD_NEG="${RUN_HARD_NEG:-1}"
RUN_SKIN_TONE="${RUN_SKIN_TONE:-1}"
RUN_QUALITATIVE="${RUN_QUALITATIVE:-1}"
RUN_LEXICAL="${RUN_LEXICAL:-1}"

mkdir -p "$RESULTS_ROOT"
SUMMARY_FILE="$RESULTS_ROOT/summary.txt"

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$SUMMARY_FILE"
}

run_and_capture() {
  local name="$1"
  shift
  local logfile="$RESULTS_ROOT/${name}.log"

  log "Running $name"
  if "$@" >"$logfile" 2>&1; then
    log "$name: OK (log: $logfile)"
  else
    log "$name: FAILED (log: $logfile)"
    return 1
  fi
}

{
  echo "model_path=$MODEL_PATH"
  echo "model_name=$MODEL_NAME"
  echo "device=$DEVICE"
  echo "seed=$SEED"
  echo "line_number=$LINE_NUMBER"
  echo "results_root=$RESULTS_ROOT"
  echo
} > "$SUMMARY_FILE"

if [[ ! -d "$MODEL_PATH" ]]; then
  log "Model path does not exist: $MODEL_PATH"
  exit 1
fi

if [[ "$RUN_BASIC" == "1" ]]; then
  if [[ -n "${EVAL_DATASETS:-}" ]]; then
    IFS=':' read -r -a DATASET_ARRAY <<< "$EVAL_DATASETS"
    run_and_capture \
      "base_sim_benchmark" \
      "$PYTHON_BIN" "$EVAL_DIR/base_sim_benchmark.py" \
      --model "$MODEL_NAME" "$MODEL_PATH" \
      --eval-datasets "${DATASET_ARRAY[@]}" \
      --log-dir "$RESULTS_ROOT/base_sim_logs" \
      --line-number "$LINE_NUMBER" \
      --seed "$SEED" \
      --device "$DEVICE"
  else
    log "base_sim_benchmark: SKIPPED (set EVAL_DATASETS as colon-separated JSONL paths)"
  fi
else
  log "base_sim_benchmark: SKIPPED (RUN_BASIC=$RUN_BASIC)"
fi

if [[ "$RUN_HARD_NEG" == "1" ]]; then
  if [[ -n "${HARD_EVAL_DATASETS:-}" ]]; then
    run_and_capture \
      "hard_negatives" \
      env \
      MODEL_PATH="$MODEL_PATH" \
      MODEL_NAME="$MODEL_NAME" \
      HARD_EVAL_DATASETS="$HARD_EVAL_DATASETS" \
      HARD_RESULTS_TXT="$RESULTS_ROOT/hard_negatives_results.txt" \
      HARD_REF_MODEL_NAME="${HARD_REF_MODEL_NAME:-openai/clip-vit-base-patch32}" \
      HARD_TOPK="${HARD_TOPK:-3}" \
      SEED="$SEED" \
      "$PYTHON_BIN" - <<'PY'
import os
import hard_negatives_evaluation as mod

mod.EVAL_DATASETS = [p for p in os.environ["HARD_EVAL_DATASETS"].split(":") if p]
mod.CLIP_CONFIGS = [(os.environ["MODEL_NAME"], os.environ["MODEL_PATH"])]
mod.RESULTS_TXT = os.environ["HARD_RESULTS_TXT"]
mod.REF_MODEL_NAME = os.environ.get("HARD_REF_MODEL_NAME", mod.REF_MODEL_NAME)
mod.HARD_TOPK = int(os.environ.get("HARD_TOPK", mod.HARD_TOPK))
mod.SEED = int(os.environ.get("SEED", mod.SEED))
mod.main()
PY
  else
    log "hard_negatives: SKIPPED (set HARD_EVAL_DATASETS as colon-separated JSONL paths)"
  fi
else
  log "hard_negatives: SKIPPED (RUN_HARD_NEG=$RUN_HARD_NEG)"
fi

if [[ "$RUN_SKIN_TONE" == "1" ]]; then
  if [[ -n "${SCIN_EVAL_DATASETS:-}" && -n "${SCIN_MANIFEST_DATASETS:-}" ]]; then
    run_and_capture \
      "skin_tone_stratified" \
      env \
      MODEL_PATH="$MODEL_PATH" \
      MODEL_NAME="$MODEL_NAME" \
      SCIN_EVAL_DATASETS="$SCIN_EVAL_DATASETS" \
      SCIN_MANIFEST_DATASETS="$SCIN_MANIFEST_DATASETS" \
      SCIN_RESULTS_TXT="$RESULTS_ROOT/scin_skin_tone_results.txt" \
      SCIN_REF_MODEL_NAME="${SCIN_REF_MODEL_NAME:-openai/clip-vit-base-patch32}" \
      SCIN_HARD_TOPK="${SCIN_HARD_TOPK:-3}" \
      SEED="$SEED" \
      "$PYTHON_BIN" - <<'PY'
import os
import hard_benchmark_scin_tone_stratified as mod

mod.EVAL_DATASETS = [p for p in os.environ["SCIN_EVAL_DATASETS"].split(":") if p]
mod.MANIFEST_DATASETS = [p for p in os.environ["SCIN_MANIFEST_DATASETS"].split(":") if p]
mod.CLIP_CONFIGS = [(os.environ["MODEL_NAME"], os.environ["MODEL_PATH"])]
mod.RESULTS_TXT = os.environ["SCIN_RESULTS_TXT"]
mod.REF_MODEL_NAME = os.environ.get("SCIN_REF_MODEL_NAME", mod.REF_MODEL_NAME)
mod.HARD_TOPK = int(os.environ.get("SCIN_HARD_TOPK", mod.HARD_TOPK))
mod.SEED = int(os.environ.get("SEED", mod.SEED))
mod.main()
PY
  else
    log "skin_tone_stratified: SKIPPED (set SCIN_EVAL_DATASETS and SCIN_MANIFEST_DATASETS)"
  fi
else
  log "skin_tone_stratified: SKIPPED (RUN_SKIN_TONE=$RUN_SKIN_TONE)"
fi

if [[ "$RUN_QUALITATIVE" == "1" ]]; then
  if [[ -n "${QUAL_EVAL_DATASET:-}" ]]; then
    run_and_capture \
      "qualitative_retrieval" \
      env \
      MODEL_PATH="$MODEL_PATH" \
      QUAL_EVAL_DATASET="$QUAL_EVAL_DATASET" \
      QUAL_K="${QUAL_K:-3}" \
      QUAL_OUT_PATH="$RESULTS_ROOT/retrieval_viz.png" \
      QUAL_PREFERRED_LABEL="${QUAL_PREFERRED_LABEL:-}" \
      "$PYTHON_BIN" - <<'PY'
import os
from display_most_sim import visualize_retrieval

preferred = os.environ.get("QUAL_PREFERRED_LABEL", "").strip()
preferred_query_labels = {preferred} if preferred else None
visualize_retrieval(
    model_name_or_path=os.environ["MODEL_PATH"],
    eval_dataset=os.environ["QUAL_EVAL_DATASET"],
    k=int(os.environ.get("QUAL_K", "3")),
    preferred_query_labels=preferred_query_labels,
    out_path=os.environ["QUAL_OUT_PATH"],
)
PY
  else
    log "qualitative_retrieval: SKIPPED (set QUAL_EVAL_DATASET)"
  fi
else
  log "qualitative_retrieval: SKIPPED (RUN_QUALITATIVE=$RUN_QUALITATIVE)"
fi

if [[ "$RUN_LEXICAL" == "1" ]]; then
  if [[ -n "${LEXICAL_DATASET:-${QUAL_EVAL_DATASET:-}}" ]]; then
    run_and_capture \
      "lexical_overlap" \
      env \
      LEXICAL_DATASET="${LEXICAL_DATASET:-${QUAL_EVAL_DATASET:-}}" \
      LEXICAL_LINE_NUMBER="${LEXICAL_LINE_NUMBER:-1000}" \
      "$PYTHON_BIN" - <<'PY'
import json
import os
import random
import re
from pathlib import Path
import numpy as np

random.seed(14)

def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def jaccard(a: str, b: str) -> float:
    ta = set(norm(a).split())
    tb = set(norm(b).split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))

path = Path(os.environ["LEXICAL_DATASET"])
line_number = int(os.environ.get("LEXICAL_LINE_NUMBER", "1000"))
with path.open("r", encoding="utf-8") as f:
    lines = f.readlines()[:line_number]

N = len(lines)
if N < 4:
    raise ValueError(f"Need at least 4 rows, got {N}")

sims = []
for i in range(N):
    pos = json.loads(lines[i])["text"]
    candidates = list(range(N))
    candidates.remove(i)
    a, b, c = random.sample(candidates, 3)
    for j in (a, b, c):
        neg = json.loads(lines[j])["text"]
        sims.append(jaccard(pos, neg))

arr = np.array(sims)
print("dataset:", path)
print("pairs:", len(arr))
print("mean:", float(arr.mean()))
print("median:", float(np.median(arr)))
print("p95:", float(np.quantile(arr, 0.95)))
print("max:", float(arr.max()))
PY
  else
    log "lexical_overlap: SKIPPED (set LEXICAL_DATASET or QUAL_EVAL_DATASET)"
  fi
else
  log "lexical_overlap: SKIPPED (RUN_LEXICAL=$RUN_LEXICAL)"
fi

log "Done. Results stored in $RESULTS_ROOT"
