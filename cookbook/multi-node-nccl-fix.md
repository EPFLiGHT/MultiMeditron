# Multi-node PyTorch Training on Clariden (CSCS) — NCCL Fix Guide

**Symptom:** NCCL hangs or crashes when running multi-node jobs on GH200 nodes with Cray Slingshot fabric.

---

## Step 1 — Create your EDF container spec

Create `~/.edf/<yourjob>.toml`:

```toml
image = "your-docker-image:tag"
mounts = ["/capstor", "/iopsstor", "/users"]
writable = true

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"

[env]
NCCL_NET = "AWS Libfabric"
NCCL_CROSS_NIC = "1"
NCCL_NET_GDR_LEVEL = "0"          # ← CRITICAL: must be "0", NOT "PHB"
FI_CXI_DISABLE_HOST_REGISTER = "1"
FI_MR_CACHE_MONITOR = "userfaultfd"
FI_CXI_DEFAULT_CQ_SIZE = "131072"
FI_CXI_DEFAULT_TX_SIZE = "32768"
FI_CXI_RX_MATCH_MODE = "software"
FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD = "16777216"
FI_CXI_COMPAT = "0"
```

> **Why `NCCL_NET_GDR_LEVEL=0`?**
> GPU Direct RDMA (GDR) is unstable on GH200 + Slingshot. Setting it to `0` disables GDR
> entirely and forces NCCL to use host memory for transfers, which is reliable.
> The CSCS template default `PHB` enables GDR up to the PCIe Host Bridge — this crashes.

---

## Step 2 — sbatch script

```bash
#!/bin/bash
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1    # one srun task per node; torchrun handles GPU forks
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH -A <your_account>

# NCCL settings — set here AND in the EDF for belt-and-suspenders
export NCCL_DEBUG=INFO                      # verbose NCCL logs for debugging
export NCCL_TIMEOUT=1800
export TORCH_NCCL_AVOID_RECORD_STREAMS=1   # avoids stream recording bugs in PyTorch
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # fast error detection across ranks
export NCCL_NET_GDR_LEVEL=0

GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6200

LAUNCHER="torchrun \
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $SLURM_NNODES \
  --node_rank \$SLURM_PROCID \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  --rdzv_backend c10d \
  --max_restarts 0 \
  --tee 3"

srun \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  --environment ~/.edf/<yourjob>.toml \
  --export=ALL,NCCL_NET_GDR_LEVEL=0 \      # ← pass explicitly to srun too
  bash -c "$LAUNCHER -m your.training.module"
```

> **Why pass `NCCL_NET_GDR_LEVEL=0` to both `export` and the EDF?**
> `--export=ALL` on `srun` can override env vars set in the EDF in some container
> runtimes. Passing it explicitly in `--export` ensures it always wins.

---

## Step 3 — Verify NCCL found the right network

With `NCCL_DEBUG=INFO`, check the job logs for:

```
NCCL INFO NET/OFI Selected provider is cxi, fabric is cxi (found 4 nics)
NCCL INFO Using network AWS Libfabric
```

If you see `Using network Socket` instead → the OFI plugin didn't load. Check the
`[annotations]` block in your `.toml`.

---

## What each piece does

| Setting | Where | Purpose |
|---|---|---|
| `NCCL_NET_GDR_LEVEL=0` | EDF + sbatch | Disables GPU Direct RDMA — the main crash fix |
| `NCCL_NET=AWS Libfabric` | EDF | Forces Cray Slingshot (CXI) instead of InfiniBand |
| `aws_ofi_nccl.enabled=true` | EDF annotation | Loads the OFI NCCL plugin inside the container |
| `TORCH_NCCL_AVOID_RECORD_STREAMS=1` | sbatch | Avoids NCCL stream recording bugs in PyTorch |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | sbatch | Fast cross-rank error detection |
| `--rdzv_backend c10d` | torchrun | Reliable rendezvous on SLURM (vs `etcd`) |
| `--ntasks-per-node 1` | sbatch | One `srun` process per node; `torchrun` forks GPUs |
