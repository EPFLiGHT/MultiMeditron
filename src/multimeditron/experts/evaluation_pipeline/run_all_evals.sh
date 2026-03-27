#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/lightscratch/users/cljordan/multimeditron"
EVAL_DIR="$ROOT_DIR/src/multimeditron/experts/evaluation_pipeline"
DEFAULT_MODEL="$ROOT_DIR/src/multimeditron/experts/models/generalist_expert_v1"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
DEVICE="${DEVICE:-cuda:0}"
LINE_NUMBER="${LINE_NUMBER:-300}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/src/multimeditron/experts/eval_results/${MODEL_NAME}_${TIMESTAMP}}"

RUN_BASE_SIM="${RUN_BASE_SIM:-1}"
RUN_ULTRASOUND="${RUN_ULTRASOUND:-1}"
RUN_XRAY="${RUN_XRAY:-1}"

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
  echo "results_root=$RESULTS_ROOT"
  echo
} > "$SUMMARY_FILE"

if [[ ! -d "$MODEL_PATH" ]]; then
  log "Model path does not exist: $MODEL_PATH"
  exit 1
fi

if [[ "$RUN_BASE_SIM" == "1" ]]; then
  if [[ -n "${EVAL_DATASETS:-}" ]]; then
    IFS=':' read -r -a DATASET_ARRAY <<< "$EVAL_DATASETS"
    run_and_capture \
      "base_sim_benchmark" \
      "$PYTHON_BIN" "$EVAL_DIR/base_sim_benchmark.py" \
      --model "$MODEL_NAME" "$MODEL_PATH" \
      --eval-datasets "${DATASET_ARRAY[@]}" \
      --log-dir "$RESULTS_ROOT/base_sim_logs" \
      --line-number "$LINE_NUMBER" \
      --device "$DEVICE"
  else
    log "base_sim_benchmark: SKIPPED (set EVAL_DATASETS as a colon-separated list of JSONL files)"
  fi
else
  log "base_sim_benchmark: SKIPPED (RUN_BASE_SIM=$RUN_BASE_SIM)"
fi

if [[ "$RUN_ULTRASOUND" == "1" ]]; then
  run_and_capture \
    "ultrasound_benchmark" \
    "$PYTHON_BIN" "$EVAL_DIR/ultrasound_new_benchmark.py" "$MODEL_PATH"
else
  log "ultrasound_benchmark: SKIPPED (RUN_ULTRASOUND=$RUN_ULTRASOUND)"
fi

if [[ "$RUN_XRAY" == "1" ]]; then
  if "$PYTHON_BIN" -c 'import kagglehub' >/dev/null 2>&1; then
    run_and_capture \
      "xray_benchmark" \
      "$PYTHON_BIN" "$EVAL_DIR/xray_eval.py" "$MODEL_PATH" false
  else
    log "xray_benchmark: SKIPPED (python package kagglehub is not installed)"
  fi
else
  log "xray_benchmark: SKIPPED (RUN_XRAY=$RUN_XRAY)"
fi

log "Done. Results stored in $RESULTS_ROOT"
