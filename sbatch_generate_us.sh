#!/bin/bash
#SBATCH --job-name generate-us-desc
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 72
#SBATCH --time 02:00:00
#SBATCH -A a127

# Generate structured ultrasound descriptions using Qwen3-VL-8B-Instruct.
#
# Usage:
#   # Generate 50 examples for PDF review (default):
#   sbatch sbatch_generate_us.sh BUSI
#   sbatch sbatch_generate_us.sh ct2
#   sbatch sbatch_generate_us.sh DDTI
#
#   # Generate all samples:
#   sbatch --time 08:00:00 sbatch_generate_us.sh BUSI --all
#
# Output lands in: /iopsstor/scratch/cscs/surech/multimeditron/generated_data/
#   context_examples/{DATASET}.jsonl   <- input for pdf_gen.py
#   output/{DATASET}_expert.jsonl      <- expert training format
#   output/{DATASET}_llm.jsonl         <- LLM training format

DATASET=${1:?"Usage: sbatch sbatch_generate_us.sh <DATASET> [--all]"}
EXTRA_ARGS="${@:2}"

REPO_DIR=/users/surech/meditron/MultiMeditron
OUTPUT_DIR=/iopsstor/scratch/cscs/surech/multimeditron/generated_data

# Source secrets
if [ -f "$REPO_DIR/.env" ]; then
    set -a; source "$REPO_DIR/.env"; set +a
fi

export HF_HOME=/capstor/store/cscs/swissai/a127/meditron/hf_cache
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN not set"}
export STORAGE_ROOT=/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow
export PYTHONPATH=$REPO_DIR/src:$PYTHONPATH

mkdir -p "$OUTPUT_DIR"

echo "START TIME: $(date)"
echo "DATASET: $DATASET"
echo "OUTPUT:  $OUTPUT_DIR"

srun \
    --ntasks=1 \
    --environment /users/surech/.edf/multimeditron.toml \
    --export=ALL \
    bash -c "
        cd $REPO_DIR
        python3 scripts/generate_us_descriptions.py \
            --dataset $DATASET \
            --output_dir $OUTPUT_DIR \
            $EXTRA_ARGS
    "

echo "END TIME: $(date)"
