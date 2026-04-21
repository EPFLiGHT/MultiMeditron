#!/bin/bash
#SBATCH --job-name gating-7class-debug
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --partition debug
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 00:29:59
#SBATCH -A a127

# Usage:
#   sbatch sbatch_train_gating_debug.sh
#
# Quick DDP debug run for gating network training (2 nodes, 8 GPUs).
# WandB is disabled; max 500 samples/class; 2 epochs only.

export HF_HOME=/iopsstor/scratch/cscs/surech/hf
export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:/users/surech/meditron/MultiMeditron/third-party/lmms-eval:${PYTHONPATH:-}

# WandB disabled for debug
export WANDB_MODE=disabled

# NCCL / distributed settings
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET_GDR_LEVEL=0

OUTPUT_DIR=/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/gating/7class_debug
CONFIG=/users/surech/meditron/MultiMeditron/config/gating_7class.yaml

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
  --num_epochs 2 \
  --batch_size 64 \
  --max_samples_per_class 500"

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  --environment /users/surech/.edf/multimeditron.toml \
  --export=ALL,NCCL_NET_GDR_LEVEL=0 \
  "

srun $SRUN_ARGS bash -c "$CMD"

echo "END TIME: $(date)"
