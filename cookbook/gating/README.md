# MultiMeditron MoE Training Guide

This guide documents our full training pipeline for MultiMeditron's Mixture-of-Experts (MoE) architecture — covering gating network training, alignment, end-to-end fine-tuning, evaluation, and the pitfalls we ran into along the way.

> **Audience**: Anyone with access to the CSCS Clariden cluster (account `a127`).
> Based on our experience adding Ophthalmology + Dermatology experts to the 5-expert baseline (March 2026).

---

## Table of Contents

1. [Overview & Architecture](#-overview--architecture)
2. [Step 1 — Train the Gating Network](#-step-1--train-the-gating-network)
3. [Step 2 — Stage 1: Alignment Training](#-step-2--stage-1-alignment-training)
4. [Step 3 — Stage 2: End-to-End Fine-Tuning](#-step-3--stage-2-end-to-end-fine-tuning)
5. [Step 4 — Evaluation](#-step-4--evaluation)
6. [Cluster Reference (CSCS Clariden)](#-cluster-reference-cscs-clariden)
7. [Troubleshooting & Roadblocks](#-troubleshooting--roadblocks)

---

## 🏗️ Overview & Architecture

Our architecture uses a **Mixture-of-Experts (MoE)** vision encoder. Each input image is routed by a gating network to one or more domain-specific CLIP models, whose embeddings are fused via cross-attention before being projected into the LLM's token space.

```
Input image (224×224)
      │
  Gating Network (ResNet50)
      │
  ┌───┴────────────────────────────────┐
  │ Expert CLIP 1 (CT)                 │
  │ Expert CLIP 2 (MRI)               │
  │ Expert CLIP 3 (Ultrasound)        │
  │ Expert CLIP 4 (Xray)              │
  │ Expert CLIP 5 (General)           │
  │ Expert CLIP 6 (Ophthalmology)     │
  │ Expert CLIP 7 (Dermatology)       │
  └───┬────────────────────────────────┘
      │  top_k selected, softmax-weighted
      │
  Cross-Attention Fusion (PEP)
      │
  Linear Projection → LLM token space
      │
  LLaMA 3.1 8B (Meditron3)
```

**Our training pipeline has three phases** (assuming CLIP experts already exist):
1. Train/retrain the gating network to route images to the correct expert(s)
2. Stage 1 alignment training (frozen LLM, train projector + cross-attention)
3. Stage 2 end-to-end training (unfrozen LLM, all parameters)

---

## 🧭 Step 1 — Train the Gating Network

The gating network is a **ResNet50 classification backbone** with a replaced FC head that routes each input image to the most relevant expert(s). We retrain it every time we add or remove an expert.

### 📐 Architecture

We use a ResNet50 with a replaced fully-connected head:

```
Input image (224×224)
      │
  ResNet50 (frozen or thawed)
      │
  Linear(2048 → num_classes)
      │
  Softmax → per-expert weights   (used at inference in MoE)
  Top-K   → selected expert idx  (used to gate computation)
```

- **`num_classes`**: number of expert CLIP models (e.g. 7 for CT/MRI/Ultrasound/Xray/General/Ophthalmology/Skin)
- **`top_k`**: how many experts to activate per image (typically 1 for routing accuracy, 3 for richer fusion)
- Weights are softmax-normalized over all classes — the full softmax vector is used as fusion weights in cross-attention fusion (`cross_attn` mode), regardless of `top_k`

We store the model as a HuggingFace `PreTrainedModel` via `GatingNetwork` / `GatingNetworkConfig`.

---

### 🗂️ Dataset Preparation

Our training script expects an **ImageFolder** layout — one subdirectory per expert class:

```
data/
├── train/
│   ├── CT/            ← CT scans
│   ├── MRI/           ← MRI scans
│   ├── Ultrasound/    ← Ultrasound images
│   ├── Xray/          ← Chest X-rays
│   ├── General/       ← General images (LLaVA-Pretrain, etc.)
│   ├── Ophthalmology/ ← Fundus / OCT images
│   └── Skin/          ← Dermatology images
└── test/
    ├── CT/
    ├── ...            ← Same structure as train/
```

**Recommended dataset sources per class (7-expert setup):**

| Class | Dataset | ~Size |
|-------|---------|-------|
| CT | `ct2` | 25K |
| MRI | `PMC_VQA` (MRI subset) | 20K |
| Ultrasound | `BUSI` + `COVID_US` | 31K |
| Xray | `iu_xray` | 8K |
| General | `llava_pretrain` (sample) | 10K |
| Ophthalmology | `eye_dataset_converted` | 32K |
| Skin | `skin_dataset_converted` | 63K |

> **Tip:** We use `--max_samples_per_class` to cap each class and avoid imbalance. 10,000 per class is a good starting point.

---

### 🏋️ Training

We train a ResNet50 classification head on top of frozen ImageNet weights using `scripts/image_router_train.py`:

```bash
cd /users/surech/meditron/MultiMeditron

python3 scripts/image_router_train.py \
  --data_dir      data/gating/          \  # ImageFolder root
  --resnet_size   50                    \  # 18 | 34 | 50
  --num_epochs    20                    \
  --lr            1e-4                  \
  --batch_size    64                    \
  --max_samples_per_class 10000         \
  --output_dir    output/gating/
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--resnet_size` | `50` | ResNet variant (50 recommended) |
| `--max_samples_per_class` | `1000` | Cap per class for balance |
| `--lr` | `1e-4` | Learning rate |
| `--batch_size` | `16` | Training batch size |
| `--num_epochs` | `20` | Max epochs (early stopping applies) |
| `--data_dir` | `data/images/` | Root with `train/` and `test/` subdirs |
| `--output_dir` | `output/` | Where to save `model_<timestamp>.pth` |

The backbone is frozen by default — only the final FC layers are trained. Early stopping triggers when accuracy plateaus above 90% and loss stops decreasing.

---

### 💾 Converting to HuggingFace Format

After training, we convert the raw `.pth` weights into the HuggingFace `GatingNetwork` format so it can be consumed by our MoE training configs:

```python
from multimeditron.model.modalities.moe.gating import GatingNetwork, GatingNetworkConfig
import torch

config = GatingNetworkConfig(
    num_classes=7,
    top_k=3,
    image_processor_path="openai/clip-vit-base-patch32",
    class_names=["CT", "MRI", "Ultrasound", "Xray", "General", "Ophthalmology", "Skin"],
)
model = GatingNetwork(config, resnet_path="output/gating/model_<timestamp>.pth")
model.save_pretrained("models/CLIP/MultiMeditron-Gating-7exp")
```

> The saved directory will contain `config.json` and `model.safetensors`, ready for use as `gating_path` in our MoE YAML configs.

---

### 🔗 Integrating into MoE Training Configs

We point `gating_path` in the alignment or end-to-end YAML to the trained checkpoint:

```yaml
modalities:
  - model_type: moe_meditron_clip_pep
    image_processor: /path/to/clip-vit-base-patch32
    hidden_size: 4096
    expert_clip_names:
      - /path/to/MedExpert-CT
      - /path/to/MedExpert-MRI
      - /path/to/MedExpert-Ultrasound
      - /path/to/MedExpert-Xray
      - /path/to/clip-vit-base-patch32   # General
      - /path/to/OphthalmologyExpert
      - /path/to/SkinExpert
    generalist_idx: -1                   # -1 = last entry (SkinExpert acts as fallback here; set to 4 for General)
    gating_path: models/CLIP/MultiMeditron-Gating-7exp
    fusion_method: cross_attn            # cross_attn | avg | cat
    top_k_experts: 3
```

> **`generalist_idx`**: index into `expert_clip_names` pointing to the general-purpose CLIP. Used as fallback when routing confidence is low.

---

### 🖥️ Running on CSCS (Clariden)

The training script is lightweight (~1 GPU, 30 min for 20 epochs at 10K samples/class). We typically run it interactively in a debug job:

```bash
srun --time=00:29:59 --partition=debug -A a127 \
     --gres=gpu:1 --cpus-per-task=32 \
     --environment=~/.edf/multimeditron.toml \
     --pty bash

# Inside the job:
cd /users/surech/meditron/MultiMeditron
python3 scripts/image_router_train.py \
  --data_dir /capstor/store/cscs/swissai/a127/meditron/gating_data/ \
  --resnet_size 50 \
  --num_epochs 20 \
  --batch_size 64 \
  --max_samples_per_class 10000 \
  --output_dir /iopsstor/scratch/cscs/surech/multimeditron/checkpoints/gating/
```

---

### 📊 Expected Results

On our 7-class setup, we observed the following:

| Metric | Target |
|--------|--------|
| Validation accuracy | > 90% |
| Epochs to convergence | 5–15 (with early stopping) |
| Training time (1 GPU, 70K images) | ~15–30 min |

Routing accuracy directly impacts MoE quality. We found that if gating is poor (< 80%), experts receive off-modality images and the model underperforms a single-expert baseline.

---

### 🔍 Debugging Routing Quality

We use the following snippet to inspect which expert a set of images is routed to:

```python
from multimeditron.model.modalities.moe.gating import GatingNetwork
from PIL import Image
import torch

model = GatingNetwork.from_pretrained("models/CLIP/MultiMeditron-Gating")
model.eval()

CLASS_NAMES = ["CT", "MRI", "Ultrasound", "Xray", "General", "Ophthalmology", "Skin"]

img = Image.open("test_image.png").convert("RGB")
pixel_values = model.preprocess_images([img])

with torch.no_grad():
    logits, topk_indices, weights = model(pixel_values)

print("Predicted class:", CLASS_NAMES[topk_indices[0, 0].item()])
print("Expert weights: ", {CLASS_NAMES[i]: f"{w:.3f}" for i, w in enumerate(weights[0])})
```

---

## 🎯 Step 2 — Stage 1: Alignment Training

Stage 1 trains the vision-to-LLM projector while keeping the LLM backbone **frozen**. This teaches the model to interpret the new expert embeddings without forgetting language capabilities.

### Config: `cookbook/sft/moe/attn/pep/stage1_alignment.yaml`

Key settings (with commentary):

```yaml
base_llm: /capstor/.../Meditron3-8B/snapshots/...    # Base LLM (frozen in Stage 1)
base_model: null                                      # null = start fresh (no prior checkpoint)
resume_from_checkpoint: false                         # Set to checkpoint path to resume
training_mode: ALIGNMENT                              # Freezes LLM, trains projector + cross-attn

modalities:
  - model_type: moe_meditron_clip_pep
    expert_clip_names:
      - ClosedMeditron/MedExpert-CT
      - ClosedMeditron/MedExpert-MRI
      - ClosedMeditron/MedExpert-Ultrasound
      - ClosedMeditron/MedExpert-Xray
      - ClosedMeditron/clip-vit-base-patch32           # General
      - /capstor/.../CLIP/OphthalmologyExpert          # ← New expert
      - /capstor/.../CLIP/SkinExpert                   # ← New expert
    gating_path: ClosedMeditron/MultiMeditron-Gating   # Retrained gating (from Step 2)
    fusion_method: cross_attn
    top_k_experts: 5                                   # Higher top_k during alignment for broader exposure

datasets:                                              # Alignment datasets (caption-style, shorter)
  - packed_path: .../llava_pretrain_cleaned
  - packed_path: .../pixmo_anything
  - packed_path: .../pixmo_cap
  - packed_path: .../medtrinity_conversations_1_formatted_alignment
  - packed_path: .../eye_dataset_converted             # ← New modality data
  - packed_path: .../skin_dataset_converted            # ← New modality data

training_args:
  output_dir: /iopsstor/scratch/cscs/surech/multimeditron/checkpoints/freeze/attn_pep/MultiMeditron-8B-attn-pep-alignment-7exp
  learning_rate: 1.0e-5
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4                       # Effective batch = 8 × 4 × num_gpus
  num_train_epochs: 3
  save_steps: 712                                      # ~1 save per epoch at 8 nodes
  deepspeed: config/deepspeed_fast.json
  lr_scheduler_type: cosine_with_min_lr
  lr_scheduler_kwargs:
    min_lr: 1.0e-6
  dataloader_num_workers: 16                           # Safe to be high in Stage 1 (smaller dataset)
  dataloader_prefetch_factor: 4
  gradient_checkpointing: true
  bf16: true
```

### Launch

```bash
export HF_TOKEN=<your-token>
sbatch --nodes=8 --time=11:59:59 sbatch_train.sh \
  cookbook/sft/moe/attn/pep/stage1_alignment.yaml
```

For our 7-expert setup with 8 nodes (32 GPUs), Stage 1 took ~4–6 hours for 3 epochs (~2,139 steps). Output: `checkpoint-2139`.

### What to look for

| Metric | Healthy Range |
|--------|---------------|
| Starting loss | 2.5–3.5 |
| Final loss | 1.5–2.0 |
| Training speed | ~7 s/step at 8 nodes |

---

## 🔥 Step 3 — Stage 2: End-to-End Fine-Tuning

Stage 2 unfreezes the entire model (LLM + projector + cross-attention) for full supervised fine-tuning on medical VQA and conversation data. This is the most compute-intensive phase.

### Config: `cookbook/sft/moe/attn/pep/stage2_end2end.yaml`

Key differences from Stage 1:

```yaml
base_model: /iopsstor/scratch/cscs/surech/multimeditron/checkpoints/freeze/attn_pep/MultiMeditron-8B-attn-pep-alignment-7exp/checkpoint-2139   # ← Stage 1 output
training_mode: END2END                                           # Unfreezes everything

modalities:
  - top_k_experts: 3          # Lower top_k for sharper routing during fine-tuning

datasets:                     # Richer instruction-tuning data (more datasets, longer sequences)
  - packed_path: .../BUSI
  - packed_path: .../COVID_US
  - packed_path: .../ct2
  - packed_path: .../iu_xray
  - packed_path: .../PMC_VQA_FULL
  - packed_path: .../llava_instruct
  - packed_path: .../medtrinity_conversations_1_formatted
  - packed_path: .../medtrinity_conversations_2_formatted
  - packed_path: .../image_mammoth
  - packed_path: .../eye_dataset_converted              # ← New modality data
  - packed_path: .../skin_dataset_converted             # ← New modality data

training_args:
  output_dir: /iopsstor/scratch/cscs/surech/multimeditron/checkpoints/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end-7exp
  learning_rate: 1.0e-5
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2                        # Smaller accumulation → more frequent updates
  num_train_epochs: 1
  save_steps: 50                                        # Save often (ZeRO-3 checkpoints are large)
  max_sequence_length: 4096
  truncation: true
  deepspeed: config/deepspeed_fast.json
  dataloader_num_workers: 2                             # ⚠️ Keep low — high values cause I/O storms
  dataloader_prefetch_factor: 2
```

### Launch

Stage 2 requires significantly more compute. We ran it on 128 nodes (512 GPUs):

```bash
export HF_TOKEN=<your-token>
sbatch --nodes=128 --time=11:59:59 sbatch_train.sh \
  cookbook/sft/moe/attn/pep/stage2_end2end.yaml
```

| Scale | Effective batch | Total steps (1 epoch) | Step time | Wall time |
|-------|----------------|-----------------------|-----------|-----------|
| 8 nodes (32 GPUs) | 8 × 2 × 32 = 512 | ~24,000 | ~3.5 s | >24h |
| 64 nodes (256 GPUs) | 8 × 2 × 256 = 4,096 | ~3,000 | ~30 s | ~25h |
| 128 nodes (512 GPUs) | 8 × 2 × 512 = 8,192 | ~1,544 | ~53 s | ~23h |

> At 128 nodes, the job will **not** complete in 12h. We had to split it across two runs using `resume_from_checkpoint`.

### Resuming from a checkpoint

Set `resume_from_checkpoint` in the YAML:

```yaml
resume_from_checkpoint: /iopsstor/scratch/cscs/surech/multimeditron/checkpoints/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end-7exp/checkpoint-800
```

**Critical**: The resume must use the **same number of nodes/GPUs** as the original run. ZeRO-3 shards are tied to the rank count — we hit a `ShardedTensor` error when we tried changing the node count mid-run.

### What to look for

| Metric | Healthy Range |
|--------|---------------|
| Starting loss | 1.0–1.3 |
| Final loss | 0.4–0.6 |
| Training speed (128 nodes) | ~53 s/step |

---

## 📊 Step 4 — Evaluation

We evaluate checkpoints using `lmms-eval` with the accelerate-based multi-node launcher. Our eval script `sbatch_eval.sh` handles all setup.

### Supported benchmarks

| Task ID | Benchmark | Type |
|---------|-----------|------|
| `gmai` | GMAI-MMBench | Medical VQA (multi-choice) |
| `slake` | SLAKE-VQA | Medical VQA (open + closed) |
| `path_vqa` | PathVQA | Pathology VQA |

### Launch

```bash
export HF_TOKEN=<your-token>

# Quick test (debug partition, 30 min, first 20 samples)
sbatch --partition=debug --nodes=2 --time=00:29:59 \
  sbatch_eval.sh \
  /iopsstor/scratch/cscs/surech/multimeditron/checkpoints/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end-7exp/checkpoint-800 \
  llama \
  gmai,slake,path_vqa \
  20

# Full eval (normal partition, 16 nodes, ~50 min)
sbatch --time=03:00:00 --nodes=16 \
  sbatch_eval.sh \
  /iopsstor/scratch/cscs/surech/multimeditron/checkpoints/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end-7exp/checkpoint-800 \
  llama \
  gmai,slake,path_vqa
```

### Arguments

```
sbatch [--nodes N] [--time HH:MM:SS] sbatch_eval.sh <checkpoint> [tokenizer] [tasks] [limit]
```

| Arg | Default | Description |
|-----|---------|-------------|
| `checkpoint` | required | Path to model checkpoint |
| `tokenizer` | `llama` | Tokenizer type |
| `tasks` | `gmai,slake,path_vqa` | Comma-separated task list |
| `limit` | all | Max samples per task (for quick tests) |

### Output

Results go to `/users/surech/meditron/reports/lmms_eval_results/<checkpoint_name>/`. Each task produces a JSON with per-metric scores. Logs go to `/users/surech/meditron/reports/R-multimeditron-eval.<jobid>.{out,err}`.

### Custom tasks

Our task definitions live in `third-party/lmms-eval/lmms_eval/tasks/`. To add a new benchmark, create a YAML task file following the lmms-eval convention.

---

## 🖥️ Cluster Reference (CSCS Clariden)

### Key paths

| Item | Path |
|------|------|
| Repo root | `/users/surech/meditron/MultiMeditron` |
| Stage 1 checkpoints | `/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/freeze/attn_pep/` |
| Stage 2 checkpoints | `/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/unfreeze/attn_pep/` |
| CLIP experts | `/capstor/store/cscs/swissai/a127/meditron/models/CLIP/` |
| Datasets (Arrow) | `/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/` |
| HF cache | `/capstor/store/cscs/swissai/a127/meditron/hf_cache` |
| WandB dir | `/capstor/store/cscs/swissai/a127/homes/surech/wandb` |
| Training logs | `/users/surech/meditron/reports/R-multimeditron-train.<jobid>.{out,err}` |
| Eval results | `/users/surech/meditron/reports/lm_eval_results/` |
| GPU utilization | `/users/surech/meditron/reports/gpu-util-<jobid>/` |

### Container environments

| EDF | Purpose |
|-----|---------|
| `~/.edf/multimeditron.toml` | Training and evaluation |

### Partition limits

| Partition | Max nodes | Max wall time |
|-----------|-----------|---------------|
| `debug` | 2 | 30 min |
| `normal` | 128 | 12 hours |

### DeepSpeed configs

| Config | ZeRO Stage | CPU Offload | Use case |
|--------|-----------|-------------|----------|
| `config/deepspeed_fast.json` | 3 | No | **Recommended** — faster training |
| `config/deepspeed.json` | 3 | Yes (optimizer) | When OOM with `deepspeed_fast.json` |

### Monitoring

```bash
# Check job queue
squeue --me

# Watch training loss in real time
tail -f /users/surech/meditron/reports/R-multimeditron-train.<jobid>.out | grep "'loss'"

# Check fairshare / priority
sshare -A a127 -u surech
sprio -j <jobid>

# GPU utilization (from the `nvidia-smi dmon` log)
tail -f /users/surech/meditron/reports/gpu-util-<jobid>/node-0.log
```

### WandB sync (offline → cloud)

Compute nodes run WandB in offline mode. To sync runs to the cloud, we submit a container debug job:

```bash
sbatch --partition=debug --time=00:10:00 --nodes=1 -A a127 \
  --gres=gpu:1 --cpus-per-task=32 \
  --environment=~/.edf/multimeditron.toml \
  --output=/users/surech/meditron/reports/R-wandb-sync.%j.out \
  --error=/users/surech/meditron/reports/R-wandb-sync.%j.err \
  --wrap="wandb login <your-api-key> && wandb sync /capstor/.../wandb/offline-run-*"
```

---

## 🚧 Troubleshooting & Roadblocks

These are issues we encountered during development. Documenting them here to save others the debugging time.

### ZeRO-3 checkpoint resume: `ShardedTensor` mismatch

**Symptom**: Crash on resume with `RuntimeError: The checkpoint was created with X processes but attempted to load with Y`.

**Cause**: ZeRO-3 shards model state across all ranks. We hit this when trying to resume a 128-node run with a different node count.

**Fix**: Always resume with the exact same `--nodes` value and GPU count as the original training run. If you must change scale, convert the checkpoint to a full (non-sharded) model first using DeepSpeed's `zero_to_fp32.py`.

---

### ZeRO-2 OOM

**Symptom**: `OutOfMemoryError: CUDA out of memory` when using `"stage": 2` in the DeepSpeed config.

**Cause**: ZeRO-2 keeps full model parameters and gradients on each GPU. With our 7 CLIP experts + LLaMA-8B + cross-attention layers + activations, this exceeds 96 GB per GH200 GPU.

**Fix**: We use ZeRO-3 (`config/deepspeed_fast.json`). ZeRO-3 partitions parameters across all ranks, fitting within 96 GB. The tradeoff is ~53 s/step at 128 nodes due to all-gather communication overhead.

---

### I/O bottleneck: high `dataloader_num_workers`

**Symptom**: Training speed degrades over time, GPUs show low utilization, data loading becomes the bottleneck. Potentially `OSError: [Errno 28] No space left on device` if `/tmp` fills up.

**Cause**: Our Arrow datasets are on shared Lustre (`/capstor/`). High `num_workers` (e.g. 16) across 512 ranks = 8,192 concurrent readers → swamps the parallel filesystem.

**Fix**: We set `dataloader_num_workers: 2` and `dataloader_prefetch_factor: 2` in Stage 2 configs. Stage 1 with fewer nodes (8) can tolerate higher values.

---

### `NODE_FAIL` / `CANCELLED+` by Slurm

**Symptom**: Job killed prematurely with `srun: error: Node nidXXXXXX has been marked DOWN`.

**Cause**: Hardware faults are routine on large-scale HPC runs (128 nodes = 512 GPUs). A single GPU memory error kills the entire job.

**Fix**: We save checkpoints frequently (`save_steps: 50` in Stage 2) and resume. There is no automatic fault tolerance with DeepSpeed + torchrun.

---

### `TIMEOUT` — training exceeds wall time

**Symptom**: Job reaches the Slurm time limit before completing all steps. `sacct` shows state `TIMEOUT`.

**Cause**: At 128 nodes / ZeRO-3, Stage 2 takes ~23h for 1 epoch (~1,544 steps at 53 s/step). The normal partition limit is 12h.

**Fix**: We split training across two jobs using `resume_from_checkpoint`. With `save_steps: 50`, we always have a recent checkpoint. Our first job ran steps 0–800, and the second resumed from `checkpoint-800` to complete the remaining ~744 steps.

---

### `ModuleNotFoundError: No module named 'decord'` in eval

**Symptom**: lmms-eval crashes at import time during eval.

**Cause**: Some lmms-eval files had a top-level `from decord import VideoReader, cpu`. The `decord` package is not available in our environment.

**Fix**: Already applied — we wrapped all `decord` imports in lazy `try/except` blocks in `lmms_eval/models/simple/vllm.py`, `lmms_eval/protocol.py`, and `lmms_eval/models/model_utils/load_video.py`. Make sure you're using the latest code from the `add-ophthalmology-and-dermatology-experts` branch.

---

### WandB won't sync from login node

**Symptom**: `wandb: command not found` or `pip: command not found` on the login node.

**Cause**: Login nodes on Clariden have a bare Alpine system with no pip and no conda. WandB is only available inside the training container.

**Fix**: Submit a lightweight container job to sync (see [WandB sync](#wandb-sync-offline--cloud) above).

---

### NCCL timeout / hang on multi-node

**Symptom**: Training hangs at the start or after a few steps with `NCCL WARN ... peer ... connection timeout`.

**Cause**: Default NCCL GDR (GPU Direct RDMA) settings don't work correctly on GH200 interconnect.

**Fix**: Ensure `NCCL_NET_GDR_LEVEL=0` is set. Our `sbatch_train.sh` script exports this. If using a custom launch script, add:
```bash
export NCCL_NET_GDR_LEVEL=0
```
**Warning**: The template EDF at `cookbook/assets/edf.toml` has `NCCL_NET_GDR_LEVEL = "PHB"` which is wrong. Use `~/.edf/multimeditron.toml` which has the correct value.
