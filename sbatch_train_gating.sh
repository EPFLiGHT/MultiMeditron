#!/bin/bash
#SBATCH --job-name gating-7class
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 02:00:00
#SBATCH -A a127

# Usage:
#   sbatch sbatch_train_gating.sh
#
# Production DDP training for gating network (4 nodes, 16 GPUs).
# 20 epochs, 10K samples/class, WandB logging to "multimeditron-gating".

export HF_HOME=/iopsstor/scratch/cscs/surech/hf
export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:/users/surech/meditron/MultiMeditron/third-party/lmms-eval:${PYTHONPATH:-}

export WANDB_DIR=/capstor/store/cscs/swissai/a127/homes/surech/wandb
export WANDB_MODE=online

# Source .env for secrets (WANDB_API_KEY, etc.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a; source "$SCRIPT_DIR/.env"; set +a
fi

# NCCL / distributed settings
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET_GDR_LEVEL=0

OUTPUT_DIR=/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/gating/7class
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
  --num_epochs 20 \
  --batch_size 64 \
  --max_samples_per_class 5000 \
  --num_workers 2 \
  --wandb \
  --wandb_project multimeditron-gating"

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  --environment /users/surech/.edf/multimeditron.toml \
  --export=ALL,NCCL_NET_GDR_LEVEL=0 \
  "

GPU_LOG_DIR=/users/surech/meditron/reports/gpu-util-${SLURM_JOB_ID}
mkdir -p "$GPU_LOG_DIR"
nvidia-smi dmon -s u -d 5 > "$GPU_LOG_DIR/node-0.log" 2>&1 &
GPU_MONITOR_PID=$!

srun $SRUN_ARGS bash -c "$CMD"

kill $GPU_MONITOR_PID 2>/dev/null || true
echo "END TIME: $(date)"
