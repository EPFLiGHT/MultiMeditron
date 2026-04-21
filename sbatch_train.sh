#!/bin/bash
#SBATCH --job-name multimeditron-train
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 11:59:59
#SBATCH -A a127

# Usage:
#   sbatch sbatch_train.sh <config_path>
#
# Examples:
#   Stage 1:  sbatch sbatch_train.sh cookbook/sft/moe/attn/pep/stage1_alignment.yaml
#   Stage 2:  sbatch sbatch_train.sh cookbook/sft/moe/attn/pep/stage2_end2end.yaml
#   Debug:    sbatch --partition=debug --nodes=4 --time=00:30:00 sbatch_train.sh cookbook/sft/moe/attn/pep/stage1_alignment_debug.yaml

CONFIG=${1:?"Usage: sbatch [--partition=debug --nodes=N --time=HH:MM:SS] sbatch_train.sh <config_path>"}
CONFIG=$(realpath "$CONFIG")

export WANDB_DIR=/capstor/store/cscs/swissai/a127/homes/surech/wandb
export WANDB_MODE=offline
export HF_HOME=/capstor/store/cscs/swissai/a127/meditron/hf_cache
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN is not set. Set it in your environment or ~/.bashrc"}

# NCCL / distributed settings
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET_GDR_LEVEL=0

# Load user's source tree so edits to src/ take effect inside the container
export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:$PYTHONPATH

echo "START TIME: $(date)"
echo "NODES:  $SLURM_NNODES"
echo "CONFIG: $CONFIG"
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

CMD="$LAUNCHER -m multimeditron train --config $CONFIG"

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  --environment /users/surech/.edf/multimeditron.toml \
  --export=ALL,NCCL_NET_GDR_LEVEL=0 \
  "

GPU_LOG_DIR=/users/surech/meditron/reports/gpu-util-${SLURM_JOB_ID}
mkdir -p "$GPU_LOG_DIR"
echo "GPU utilization log (this node): $GPU_LOG_DIR/node-0.log"
nvidia-smi dmon -s u -d 5 > "$GPU_LOG_DIR/node-0.log" 2>&1 &
GPU_MONITOR_PID=$!

srun $SRUN_ARGS bash -c "$CMD"

kill $GPU_MONITOR_PID 2>/dev/null || true
echo "END TIME: $(date)"
