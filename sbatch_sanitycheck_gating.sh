#!/bin/bash
#SBATCH --job-name sanity-gating
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 00:30:00
#SBATCH -A a127

# Gating network sanity check — 4 nodes (16 GPUs), ~10 min
# 5 epochs, 5000 samples/class, full 7-class dataset
#
# Usage:
#   sbatch sbatch_sanitycheck_gating.sh

export HF_HOME=/iopsstor/scratch/cscs/surech/hf
export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:/users/surech/meditron/MultiMeditron/third-party/lmms-eval:${PYTHONPATH:-}

export WANDB_DIR=/capstor/store/cscs/swissai/a127/homes/surech/wandb
export WANDB_MODE=online

REPO_DIR=/users/surech/meditron/MultiMeditron
if [ -f "$REPO_DIR/.env" ]; then
  set -a; source "$REPO_DIR/.env"; set +a
fi

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET_GDR_LEVEL=0

OUTPUT_DIR=/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/sanitycheck/gating
CONFIG=/users/surech/meditron/MultiMeditron/config/gating_sanitycheck.yaml

echo "START TIME: $(date)"
echo "NODES:      $SLURM_NNODES"
echo "CONFIG:     $CONFIG"
echo "OUTPUT_DIR: $OUTPUT_DIR"
set -eo pipefail
set -x

GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6200

LAUNCHER="
  torchrun \
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $SLURM_NNODES \
  --node_rank \$SLURM_PROCID \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  --rdzv_backend c10d \
  --max_restarts 0 \
  --tee 3 \
  "

CMD="$LAUNCHER /users/surech/meditron/MultiMeditron/scripts/train_gating.py \
  --config $CONFIG \
  --output_dir $OUTPUT_DIR \
  --wandb \
  --wandb_project multimeditron-gating \
  --wandb_run_name sanity-7exp-gating"

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  --environment /users/surech/.edf/multimeditron.toml \
  --export=ALL,NCCL_NET_GDR_LEVEL=0 \
  "

srun $SRUN_ARGS bash -c "$CMD"

echo "END TIME: $(date)"
