#!/bin/bash
#SBATCH --job-name multimeditron-eval
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 12:00:00
#SBATCH -A a127

# Usage:
#   sbatch sbatch_eval.sh <checkpoint> [tokenizer] [tasks] [limit] [log_samples]
#
# Examples:
#   Full eval:   sbatch sbatch_eval.sh /path/to/checkpoint llama gmai,slake,path_vqa
#   Quick test:  sbatch --partition=debug --nodes=2 --time=00:29:59 sbatch_eval.sh /path/to/checkpoint llama gmai 20
#   With sample logging: sbatch sbatch_eval.sh /path/to/checkpoint llama path_vqa "" true

CHECKPOINT=${1:?"Usage: sbatch sbatch_eval.sh <checkpoint_path> [tokenizer] [tasks] [limit] [log_samples]"}
TOKENIZER=${2:-llama}
TASKS=${3:-gmai,slake,path_vqa}
LIMIT=${4:-}
LOG_SAMPLES=${5:-}

export HF_HOME=/iopsstor/scratch/cscs/surech/hf
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN is not set"}

export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:/users/surech/meditron/MultiMeditron/third-party/lmms-eval:${PYTHONPATH:-}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29400

echo "START TIME:  $(date)"
echo "CHECKPOINT:  $CHECKPOINT"
echo "TOKENIZER:   $TOKENIZER"
echo "TASKS:       $TASKS"
echo "LIMIT:       ${LIMIT:-all}"
echo "NODES:       $SLURM_NNODES"
echo "MASTER_ADDR: $MASTER_ADDR"
set -eo pipefail
set -x

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

LOG_SAMPLES_ARG=""
[ -n "$LOG_SAMPLES" ] && LOG_SAMPLES_ARG="--log_samples --log_samples_suffix model_outputs"

srun \
  --nodes "$SLURM_NNODES" \
  --ntasks "$SLURM_NNODES" \
  --ntasks-per-node 1 \
  --cpus-per-task "$SLURM_CPUS_PER_TASK" \
  --jobid "$SLURM_JOB_ID" \
  --wait 60 \
  --environment /users/surech/.edf/multimeditron.toml \
  --export=ALL \
  bash -c "
    python3 -m accelerate.commands.launch \
      --num_processes $SLURM_NNODES \
      --num_machines $SLURM_NNODES \
      --machine_rank \$SLURM_NODEID \
      --main_process_ip $MASTER_ADDR \
      --main_process_port $MASTER_PORT \
      -m lmms_eval \
      --model multimeditron \
      --model_args pretrained=$CHECKPOINT,tokenizer_type=$TOKENIZER,device_map=auto \
      --tasks $TASKS \
      --batch_size 1 \
      --include_path /users/surech/meditron/MultiMeditron/third-party/lmms-eval/lmms_eval/tasks \
      --output_path /users/surech/meditron/reports/lmms_eval_results \
      $LIMIT_ARG \
      $LOG_SAMPLES_ARG
  "
