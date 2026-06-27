#!/bin/bash
#SBATCH --job-name test-7experts
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 00:30:00
#SBATCH -A a127

export HF_HOME=/capstor/store/cscs/swissai/a127/meditron/hf_cache
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN is not set. Set it in your environment or ~/.bashrc"}
export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:/users/surech/meditron/MultiMeditron/third-party/lmms-eval:$PYTHONPATH

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_NET_GDR_LEVEL=0

WORKDIR=/users/surech/meditron/MultiMeditron
CONFIG_DIR=$WORKDIR/cookbook/sft/moe/attn/pep

echo "=========================================="
echo "TEST: 7-expert architecture (stage1 → stage2 → eval)"
echo "START TIME: $(date)"
echo "=========================================="
set -eo pipefail
set -x

SRUN="srun --cpus-per-task $SLURM_CPUS_PER_TASK --jobid $SLURM_JOB_ID --environment /users/surech/.edf/multimeditron.toml --export=ALL,NCCL_NET_GDR_LEVEL=0"

# ── Stage 1: Alignment (5 steps, single GPU, no DeepSpeed) → saves checkpoint-5 ──
echo ""
echo ">>> STAGE 1: Alignment test (5 steps)"
$SRUN torchrun --nproc_per_node 1 -m multimeditron train --config $CONFIG_DIR/test_stage1.yaml
echo ">>> STAGE 1 DONE: $(date)"

# Verify checkpoint was saved
ls -la $WORKDIR/models/freeze/attn_pep/test-7experts-alignment/checkpoint-5/ || { echo "ERROR: Stage 1 checkpoint not found"; exit 1; }

# ── Stage 2: End-to-End (5 steps, single GPU, no DeepSpeed) → saves checkpoint-5 ──
echo ""
echo ">>> STAGE 2: End-to-End test (5 steps)"
$SRUN torchrun --nproc_per_node 1 -m multimeditron train --config $CONFIG_DIR/test_stage2.yaml
echo ">>> STAGE 2 DONE: $(date)"

# Verify checkpoint was saved
ls -la $WORKDIR/models/unfreeze/attn_pep/test-7experts-end2end/checkpoint-5/ || { echo "ERROR: Stage 2 checkpoint not found"; exit 1; }

# ── Stage 3: Evaluation (GMAI benchmark, single GPU) ──
echo ""
echo ">>> STAGE 3: Evaluation test (GMAI only, batch_size=1)"
CHECKPOINT=$WORKDIR/models/unfreeze/attn_pep/test-7experts-end2end/checkpoint-5
$SRUN python3 -m accelerate.commands.launch \
    --num_processes 1 \
    -m lmms_eval \
    --model multimeditron \
    --model_args pretrained="$CHECKPOINT",tokenizer_type="llama",device_map="auto" \
    --tasks gmai \
    --batch_size 1 \
    --limit 10
echo ">>> STAGE 3 DONE: $(date)"

echo ""
echo "=========================================="
echo "ALL TESTS PASSED"
echo "END TIME: $(date)"
echo "=========================================="
