#!/bin/bash
#SBATCH --job-name multimeditron-stage1
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 11:59:59
#SBATCH -A a127

export WANDB_DIR=/capstor/store/cscs/swissai/a127/homes/surech/wandb
export WANDB_MODE=offline
export HF_HOME=/capstor/store/cscs/swissai/a127/meditron/hf_cache
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN is not set. Set it in your environment or ~/.bashrc"}

# Load user's source tree first so edits to src/ take effect inside the container
# (the container bakes its own copy at /workspace/MultiMeditron; /users is bind-mounted)
export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:$PYTHONPATH

CONFIG=/users/surech/meditron/MultiMeditron/cookbook/sft/moe/attn/pep/stage1_alignment.yaml

echo "START TIME: $(date)"
set -eo pipefail
set -x

GPUS_PER_NODE=4
echo "NODES: $SLURM_NNODES"

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

echo $CMD

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  --environment /users/surech/.edf/multimeditron.toml \
  "

srun $SRUN_ARGS bash -c "$CMD"
echo "END TIME: $(date)"
