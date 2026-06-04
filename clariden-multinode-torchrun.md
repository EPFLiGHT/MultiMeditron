    # Multi-node torchrun on Clariden (GH200) — Working Setup

    Based on a production-tested setup running up to 128 nodes on the same cluster (account `a127`, GH200 ARM64).

    ---

    ## Answers to your questions

    ### 1. `--ntasks-per-node` pattern
    Use **`--ntasks-per-node 1` + `--gres gpu:4`** (not `--ntasks-per-node=4 --gpus-per-task=1`).  
    One SLURM task per node — torchrun handles the 4 GPU processes internally.

    ### 2. `--node_rank`: use `$SLURM_PROCID`, not `$SLURM_NODEID`
    `$SLURM_NODEID` can be `0` on all nodes in some SLURM configurations on Clariden, causing all workers to think they are the master → rendezvous timeout.  
    **Always use `$SLURM_PROCID`** — it is guaranteed to be unique per task.

    ### 3. Port
    Avoid `29500` — it is commonly occupied. Use something like `6200`.

    ### 4. Wrap `torchrun` in `srun`
    Without `srun`, only the head node launches the process; worker nodes do nothing.  
    The pattern is:
    ```bash
    srun --cpus-per-task=288 --jobid=$SLURM_JOB_ID bash -c "torchrun ..."
    ```

    ### 5. Required NCCL settings on GH200
    ```bash
    export NCCL_NET_GDR_LEVEL=0          # critical — without this NCCL hangs on Clariden
    export NCCL_TIMEOUT=1800
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ```

    ---

    ## Corrected `sbatch_train.sh`

    ```bash
    #!/bin/bash
    #SBATCH --job-name=afriquellm-qwen3
    #SBATCH --account=a127
    #SBATCH --nodes=32
    #SBATCH --ntasks-per-node=1
    #SBATCH --gres=gpu:4
    #SBATCH --cpus-per-task=288
    #SBATCH --time=12:00:00
    #SBATCH --partition=normal
    #SBATCH --output=/users/gsahakyan/logs/train_%j.log
    #SBATCH --error=/users/gsahakyan/logs/train_%j.err
    #SBATCH --environment=/users/gsahakyan/.edf/afriquellm.toml

    export DISABLE_VERSION_CHECK=1
    export HF_HOME=/capstor/store/cscs/swissai/a127/homes/gsahakyan/hf
    export PYTHONPATH=$PYTHONPATH:/capstor/store/cscs/swissai/a127/homes/gsahakyan/venv-gh200/lib/python3.12/site-packages

    # Critical NCCL settings for GH200 / Clariden
    export NCCL_NET_GDR_LEVEL=0
    export NCCL_TIMEOUT=1800
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TRITON_CACHE_DIR=/capstor/store/cscs/swissai/a127/homes/gsahakyan/.triton

    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_PORT=6200

    LAUNCHER="torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=\$SLURM_PROCID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --max_restarts=0"

    CMD="$LAUNCHER \
    /capstor/store/cscs/swissai/a127/homes/gsahakyan/venv-gh200/lib/python3.12/site-packages/llamafactory/launcher.py \
    /users/gsahakyan/afriquellm-training/training/train_cpt.yaml"

    srun --cpus-per-task=288 --jobid=$SLURM_JOB_ID bash -c "$CMD"
    ```

    ---

    ## Summary of changes vs your original script

    | Issue | Your script | Fix |
    |-------|-------------|-----|
    | Rendezvous timeout | `--node_rank=$SLURM_NODEID` | `--node_rank=\$SLURM_PROCID` |
    | Worker nodes idle | `srun torchrun ...` directly | `srun bash -c "torchrun ..."` |
    | Port conflict | `29500` | `6200` |
    | NCCL hangs | Missing `NCCL_NET_GDR_LEVEL` | `export NCCL_NET_GDR_LEVEL=0` |
    | Memory fragmentation OOM | Missing | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

    The two most likely culprits for the rendezvous timeout are **`$SLURM_PROCID`** and the **`srun bash -c`** wrapper.
