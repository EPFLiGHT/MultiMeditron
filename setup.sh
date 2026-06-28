export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="online"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_TIMEOUT=900  # seconds
export NCCL_IB_HCA=mlx5_bond_
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=$RUNAI_NUM_OF_GPUS

# --- HuggingFace cache (optional: redirect to fast scratch instead of ~/.cache/huggingface) ---
export HF_HOME="/lightscratch/users/$USER/multimeditron/cache"

# --- Benchmark evaluation paths ---
# Shared cluster path — no change needed
export MULTIMEDISET_ROOT="/lightscratch/datasets/MultiMediset/general_purpose"

# Update these two paths to point to your own downloaded datasets (see evaluation_pipeline/README.md)
export MRI_DATASET_ROOT="/lightscratch/users/cljordan/datasets/brain_tumor_mri/images"  # example — replace with your path
export XRAY_DATA_ROOT="/lightscratch/users/cljordan/datasets/nih_chest_xrays"           # example — replace with your path
export XRAY_KAGGLE_DATA_ROOT="/lightscratch/users/cljordan/datasets/nih_chest_xrays"    # example — replace with your path

# --- Optional: cap number of examples per benchmark split ---
# export CT_MAX_TRAIN_EXAMPLES=5000
# export CT_MAX_TEST_EXAMPLES=1000
# export SKIN_INTEGRATED_MAX_TRAIN_EXAMPLES=5000
# export SKIN_INTEGRATED_MAX_TEST_EXAMPLES=1000
# export OPHTH_MAX_TRAIN_EXAMPLES=5000
# export OPHTH_MAX_TEST_EXAMPLES=1000
# export ULTRASOUND_MAX_TRAIN_EXAMPLES=5000
# export ULTRASOUND_MAX_TEST_EXAMPLES=1000
# export MRI_MAX_TRAIN_EXAMPLES=5712
# export MRI_MAX_TEST_EXAMPLES=1311
# export XRAY_MAX_TRAIN_EXAMPLES=5000
# export XRAY_MAX_TEST_EXAMPLES=3000
# export HISTO_MAX_TRAIN_EXAMPLES=10000
# export HISTO_MAX_TEST_EXAMPLES=5000

