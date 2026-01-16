#!/bin/bash
#SBATCH --job-name multimeditron-tutorial
#SBATCH --output /users/$USER/reports/multimeditron/R-%x.%j.out
#SBATCH --error /users/$USER/reports/multimeditron/R-%x.%j.err
#SBATCH --nodes $NNODES         # number of Nodes
#SBATCH --ntasks-per-node 1     # number of MP tasks. IMPORTANT: torchrun represents just 1 Slurm task
#SBATCH --gres gpu:4        # Number of GPUs
#SBATCH --cpus-per-task 288     # number of CPUs per task.
#SBATCH --time 11:59:59       # maximum execution time (DD-HH:MM:SS)
#SBATCH -A <ACCOUNT>

export WANDB_DIR=$WANDB_DIR
export WANDB_API_KEY=$WANDB_API_KEY

export WANDB_MODE=$WANDB_MODE
export HF_TOKEN=$HF_TOKEN
export HF_HOME=$HF_HOME
export CONFIG=$1

echo "START TIME: $(date)"
set -eo pipefail
set -x

######################
### Set environment ###
######################

GPUS_PER_NODE=4
echo "NODES: $SLURM_NNODES"

######################
#### Set network #####
######################
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6200
######################
# note that we don't want to interpolate `\$SLURM_PROCID` till `srun` since otherwise all nodes will get
# 0 and the launcher will hang
#
# same goes for `\$(hostname -s|tr -dc '0-9')` - we want it to interpolate at `srun` time

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

export CMD="$LAUNCHER -m multimeditron train --config $CONFIG"

echo $CMD

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  --environment ~/.edf/multimodal.toml
  "
# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$CMD"
echo "END TIME: $(date)"
