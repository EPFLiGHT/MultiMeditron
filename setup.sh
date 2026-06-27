#!/usr/bin/env bash
# Optional convenience environment for INTERACTIVE sessions (e.g. an srun shell
# inside the container). Source it with:  source setup.sh
#
# Batch jobs do NOT need this: the SLURM launchers (sbatch_train.sh, sbatch_eval.sh)
# and the EDF (docker/multimeditron.toml) already set the canonical cluster
# environment, including the Slingshot/libfabric NCCL configuration. This file
# only sets a few generally-useful, non-conflicting defaults.

# Faster HuggingFace downloads; quieter tokenizers.
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

# Keep the HF cache on scratch (the home directory has a small quota — downloads
# fail mid-run otherwise). Override for your own user as needed.
export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/$USER/hf}"

# WandB online by default; set to "offline" on compute nodes (no outbound network)
# and run `wandb sync <run_dir>` afterwards.
export WANDB_MODE="${WANDB_MODE:-online}"

# Reduce CUDA fragmentation for long training runs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Required on Clariden GH200 — without it, multi-node NCCL collectives hang.
# (Already set in the EDF for containerized runs; repeated here for bare srun.)
export NCCL_NET_GDR_LEVEL=0
