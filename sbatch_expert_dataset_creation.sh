#!/bin/bash
#SBATCH --job-name=expert-ds-export
#SBATCH --output=/users/haaissa/reports/data/expert-export-%x.%j.out
#SBATCH --error=/users/haaissa/reports/data/expert-export-%x.%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -A a127

echo "START TIME: $(date)"

set -eo pipefail
set -x

# required storage root for your dataset subsets
export STORAGE_ROOT=$STORE/meditron/multimediset/arrow
# optional: disable wandb use for this job
export WANDB_MODE=offline

# run the export script INSIDE the container environment
SRUN_ARGS=" \
  --ntasks ${SLURM_NTASKS} \
  --gpus-per-task 0 \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --environment multimeditron \
  "

srun $SRUN_ARGS python /users/haaissa/meditron/MultiMeditron/expert_dataset_creation.py --split train

echo "END TIME: $(date)"
