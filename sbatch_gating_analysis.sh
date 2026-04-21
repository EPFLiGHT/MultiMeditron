#!/bin/bash
#SBATCH --job-name gating-routing-analysis
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --time 00:29:00
#SBATCH --partition debug
#SBATCH -A a127
# No GPUs requested — ResNet50 inference is fast on CPU for ~1000 images total

export HF_HOME=/iopsstor/scratch/cscs/surech/hf
export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:${PYTHONPATH:-}

echo "START: $(date)"
echo "NODES: $SLURM_NNODES"

srun \
  --nodes 1 \
  --ntasks 1 \
  --ntasks-per-node 1 \
  --cpus-per-task 16 \
  --jobid "$SLURM_JOB_ID" \
  --environment /users/surech/.edf/multimeditron.toml \
  --export=ALL \
  python3 /users/surech/meditron/MultiMeditron/scripts/gating_routing_analysis.py

echo "END: $(date)"
