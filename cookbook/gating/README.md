# MultiMeditron MoE Training Guide

This guide documents our full training pipeline for MultiMeditron's Mixture-of-Experts (MoE) architecture — covering gating network training, alignment, end-to-end fine-tuning, evaluation, and the pitfalls we ran into along the way.

> **Audience**: Anyone with access to the CSCS Clariden cluster (account `a127`).
> Based on our experience adding Ophthalmology + Dermatology experts to the 5-expert baseline (March 2026).

---

## ⚠️ Prerequisites

Before running any commands in this guide, make sure you have:

1. **Container environment**: All commands run inside the EDF container — **never** on the bare login node (which has no Python packages). Every `sbatch` script in this repo already includes `--environment=~/.edf/multimeditron.toml`. For interactive jobs, always pass it explicitly:
   ```bash
   srun --environment=~/.edf/multimeditron.toml --pty bash
   ```

2. **Environment variables**: Set these in your `~/.bashrc` or export them before every `sbatch` call:
   ```bash
   export HF_TOKEN=<your-huggingface-token>
   export HF_HOME=/iopsstor/scratch/cscs/surech/hf          # Avoids home-directory quota errors
   export WANDB_DIR=/capstor/store/cscs/swissai/a127/homes/surech/wandb

   # These are used inside the YAML configs (e.g. $STORAGE_ROOT/llava_pretrain_cleaned):
   export WORKING_DIR=/users/surech/meditron/MultiMeditron   # Repo root (used for deepspeed path)
   export STORAGE_ROOT=/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow  # Arrow datasets
   export MODEL_ROOT=/iopsstor/scratch/cscs/surech/multimeditron/checkpoints         # Checkpoint output
   ```
   > **Warning**: The default `HF_HOME` is `~/.cache/huggingface/`, which is on the 50 GB home quota. Model downloads will fill it instantly. Always point `HF_HOME` to scratch.

3. **Working directory**: Always `cd` to the repo root before launching:
   ```bash
   cd /users/surech/meditron/MultiMeditron
   ```
   The YAML configs use **environment variables** — `$WORKING_DIR`, `$STORAGE_ROOT`, and `$MODEL_ROOT` — for DeepSpeed, dataset, and checkpoint paths respectively. These **must** be exported before launching a job (see above).

4. **Checkpoint paths**: Always point to a specific checkpoint subdirectory (e.g. `.../checkpoint-666`), **not** the parent output directory. The parent directory does not contain model weights.

---

## Table of Contents

0. [Prerequisites](#%EF%B8%8F-prerequisites)
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
  └───┬────────────────────────────────┘
      │  top_k selected, softmax-weighted
      │
  Cross-Attention Fusion (PEP)
      │
  Linear Projection → LLM token space
      │
  LLaMA 3.1 8B (Meditron3)
```

> **Extending to more experts**: To add new modalities (e.g. Ophthalmology, Dermatology), add their CLIP models to `expert_clip_names` in the YAML config, retrain the gating network with the new classes, and re-run the pipeline.

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

- **`num_classes`**: number of expert CLIP models (5 in the baseline: CT/MRI/Ultrasound/Xray/General)
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
│   └── General/       ← General images (LLaVA-Pretrain, etc.)
└── test/
    ├── CT/
    ├── ...            ← Same structure as train/
```

> **Adding new modalities**: To extend to more experts (e.g. Ophthalmology, Skin), add new subdirectories to the ImageFolder layout and retrain.

**Recommended dataset sources per class (5-expert baseline):**

| Class | Dataset | ~Size |
|-------|---------|-------|
| CT | `ct2` | 25K |
| MRI | `PMC_VQA` (MRI subset) | 20K |
| Ultrasound | `BUSI` + `COVID_US` | 31K |
| Xray | `iu_xray` | 8K |
| General | `llava_pretrain` (sample) | 10K |

> **Tip:** We use `--max_samples_per_class` to cap each class and avoid imbalance. 10,000 per class is a good starting point.

---

### 🏋️ Training

We train a ResNet50 classification head on top of frozen ImageNet weights using `scripts/train_gating.py`. Unlike the old ImageFolder-based router, this script reads per-class Arrow datasets directly (via the `dataset_class_map` in the YAML config) and saves the result as a ready-to-use HuggingFace `GatingNetwork`:

```bash
cd /users/surech/meditron/MultiMeditron

# Multi-GPU via torchrun (config-driven):
torchrun --nproc_per_node=4 scripts/train_gating.py --config config/gating_7class.yaml

# Override any config value from the CLI:
torchrun --nproc_per_node=4 scripts/train_gating.py \
  --config config/gating_7class.yaml \
  --num_epochs 30 --lr 3e-4 --batch_size 64
```

**Key config keys / CLI overrides:**

| Key | Default | Description |
|----------|---------|-------------|
| `dataset_class_map` | *(required)* | Maps class index → list of Arrow dataset paths |
| `class_names` | 7 expert paths | Ordered expert checkpoints (aligns gating index ↔ expert) |
| `max_samples_per_class` | `0` (all) | Cap per class for balance |
| `lr` | `1e-4` | Learning rate |
| `batch_size` | `32` | Training batch size |
| `num_epochs` | `20` | Max epochs (early stopping applies) |
| `val_split` / `test_split` | `0.1` / `0.1` | Validation and held-out test fractions |
| `freeze_backbone` | `true` | Train only the FC head (use `--unfreeze_backbone` to train all) |
| `output_dir` | `models/CLIP/MultiMeditron-Gating-7class` | Where the checkpoint is saved |

The backbone is frozen by default — only the final FC layer is trained. A held-out `test_split` is carved off **before** training so the reported test accuracy is unbiased.

---

### 💾 HuggingFace Format

`train_gating.py` saves the checkpoint **directly** in HuggingFace `GatingNetwork`
format (`config.json` + `model.safetensors`, with `class_names` embedded) at
`output_dir` — no manual conversion step is needed. It is ready to use as
`gating_path` in the MoE YAML configs.

> If you have a legacy raw `.pth` ResNet50 state-dict to convert, wrap it once:
>
> ```python
> from multimeditron.model.modalities.moe.gating import GatingNetwork, GatingNetworkConfig
> config = GatingNetworkConfig(
>     num_classes=7, top_k=1,
>     image_processor_path="openai/clip-vit-base-patch32",
>     class_names=["CT", "General", "MRI", "Ultrasound", "Xray", "Ophthalmology", "Skin"],
> )
> GatingNetwork(config, resnet_path="model_<timestamp>.pth").save_pretrained("models/CLIP/MultiMeditron-Gating")
> ```

---

### 🔗 Integrating into MoE Training Configs

We point `gating_path` in the alignment or end-to-end YAML to the trained checkpoint:

```yaml
modalities:
  - model_type: moe_meditron_clip_pep
    image_processor: openai/clip-vit-base-patch32
    hidden_size: 4096
    expert_clip_names:
      - ClosedMeditron/MedExpert-CT
      - ClosedMeditron/MedExpert-MRI
      - ClosedMeditron/MedExpert-Ultrasound
      - ClosedMeditron/MedExpert-Xray
      - ClosedMeditron/clip-vit-base-patch32   # General
    generalist_idx: -1                   # -1 = last entry (General)
    gating_path: ClosedMeditron/MultiMeditron-Gating
    fusion_method: cross_attn            # cross_attn | avg | cat
    top_k_experts: 5
```

> **Note**: Expert CLIP model names are resolved from the HuggingFace cache (`$HF_HOME`). They were uploaded to the `ClosedMeditron` HF organization.

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
torchrun --nproc_per_node=1 scripts/train_gating.py --config config/gating_7class.yaml
```

Or submit non-interactively with the launcher: `sbatch sbatch_train_gating.sh`.

---

### 📊 Expected Results

On our 5-class setup, we observed the following:

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

CLASS_NAMES = ["CT", "MRI", "Ultrasound", "Xray", "General"]

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
base_llm: meta-llama/Llama-3.1-8B-Instruct           # Base LLM (frozen in Stage 1)
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
    gating_path: ClosedMeditron/MultiMeditron-Gating
    fusion_method: cross_attn
    top_k_experts: 5

datasets:                                              # Alignment datasets (caption-style, shorter)
  - packed_path: $STORAGE_ROOT/llava_pretrain_cleaned
  - packed_path: $STORAGE_ROOT/pixmo_anything
  - packed_path: $STORAGE_ROOT/pixmo_cap
  - packed_path: $STORAGE_ROOT/medtrinity_conversations_1_formatted_alignment

training_args:
  output_dir: $MODEL_ROOT/freeze/attn_pep/MultiMeditron-8B-attn-pep-alignment
  learning_rate: 1.0e-5
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  save_steps: 0.25                                     # ~1 save per 25% of epoch
  deepspeed: $WORKING_DIR/config/deepspeed.json        # ZeRO-3 with CPU offload
  lr_scheduler_type: cosine_with_min_lr
  lr_scheduler_kwargs:
    min_lr: 1.0e-6
  dataloader_num_workers: 16
  dataloader_prefetch_factor: 4
  gradient_checkpointing: true
  bf16: true
```

> **Note**: Expert model names like `ClosedMeditron/MedExpert-CT` are HuggingFace model IDs, resolved from `$HF_HOME`. Environment variables `$STORAGE_ROOT`, `$WORKING_DIR`, and `$MODEL_ROOT` must be set before launching (see [Prerequisites](#%EF%B8%8F-prerequisites)).

### Launch

```bash
cd /users/surech/meditron/MultiMeditron
export HF_TOKEN=<your-token>
export HF_HOME=/iopsstor/scratch/cscs/surech/hf

sbatch --nodes=8 --time=11:59:59 sbatch_train.sh
```

> **Note**: On the `master` branch, `sbatch_train.sh` has the config path **hardcoded** to `stage1_alignment.yaml`. To run a different config, edit the `CONFIG=` line in the script before submitting. The script handles container setup (`--environment`), NCCL settings, PYTHONPATH, and WandB config automatically.

For our 5-expert setup with 8 nodes (32 GPUs), Stage 1 took ~4–6 hours for 1 epoch. Output: `checkpoint-666`.

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
base_model: $MODEL_ROOT/freeze/attn_pep/MultiMeditron-8B-attn-pep-alignment/checkpoint-666   # ← Stage 1 output
training_mode: END2END                                           # Unfreezes everything

modalities:
  - top_k_experts: 5

datasets:                     # Richer instruction-tuning data (more datasets, longer sequences)
  - packed_path: $STORAGE_ROOT/BUSI
  - packed_path: $STORAGE_ROOT/COVID_US
  - packed_path: $STORAGE_ROOT/ct2
  - packed_path: $STORAGE_ROOT/iu_xray
  - packed_path: $STORAGE_ROOT/PMC_VQA_FULL
  - packed_path: $STORAGE_ROOT/llava_instruct
  - packed_path: $STORAGE_ROOT/medtrinity_conversations_1_formatted
  - packed_path: $STORAGE_ROOT/medtrinity_conversations_2_formatted
  - packed_path: $STORAGE_ROOT/image_mammoth

training_args:
  output_dir: $MODEL_ROOT/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end
  learning_rate: 1.0e-5
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  save_steps: 0.25                                      # Save 4× per epoch
  max_sequence_length: 4096
  truncation: true
  deepspeed: $WORKING_DIR/config/deepspeed.json         # ZeRO-3 with CPU offload
  dataloader_num_workers: 16
  dataloader_prefetch_factor: 4
```

### Launch

Stage 2 requires significantly more compute. We ran it on 128 nodes (512 GPUs):

```bash
cd /users/surech/meditron/MultiMeditron
export HF_TOKEN=<your-token>
export HF_HOME=/iopsstor/scratch/cscs/surech/hf

# Edit sbatch_train.sh: change CONFIG= to point to stage2_end2end.yaml
sbatch --nodes=128 --time=11:59:59 sbatch_train.sh
```

> **Important**: On the `master` branch, `sbatch_train.sh` has `CONFIG` hardcoded to `stage1_alignment.yaml`. Before launching Stage 2, edit the script to change the `CONFIG=` line to point to `cookbook/sft/moe/attn/pep/stage2_end2end.yaml`.

| Scale | Effective batch | Total steps (1 epoch) | Step time | Wall time |
|-------|----------------|-----------------------|-----------|-----------|
| 8 nodes (32 GPUs) | 2 × 8 × 32 = 512 | ~24,000 | ~3.5 s | >24h |
| 64 nodes (256 GPUs) | 2 × 8 × 256 = 4,096 | ~3,000 | ~30 s | ~25h |
| 128 nodes (512 GPUs) | 2 × 8 × 512 = 8,192 | ~1,544 | ~53 s | ~23h |

> At 128 nodes, the job will **not** complete in 12h. We had to split it across two runs using `resume_from_checkpoint`.

### Resuming from a checkpoint

Set `resume_from_checkpoint` in the YAML:

```yaml
resume_from_checkpoint: $MODEL_ROOT/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end/checkpoint-800
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

> **Note**: `sbatch_eval.sh` does not exist on the `master` branch. It was added on the `add-ophthalmology-and-dermatology-experts` feature branch. To run evaluation, either cherry-pick it from that branch or write your own wrapper using the pattern below.

```bash
cd /users/surech/meditron/MultiMeditron
export HF_TOKEN=<your-token>
export HF_HOME=/iopsstor/scratch/cscs/surech/hf

# Quick test (debug partition, 30 min, first 20 samples)
sbatch --partition=debug --nodes=2 --time=00:29:59 \
  sbatch_eval.sh \
  $MODEL_ROOT/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end/checkpoint-3063 \
  llama \
  gmai,slake,path_vqa \
  20

# Full eval (normal partition, 16 nodes, ~50 min)
sbatch --time=03:00:00 --nodes=16 \
  sbatch_eval.sh \
  $MODEL_ROOT/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end/checkpoint-3063 \
  llama \
  gmai,slake,path_vqa
```

> **Important**: The checkpoint path must point to a specific checkpoint subdirectory (e.g. `.../checkpoint-3063`), not the parent training output directory. The parent does not contain `model.safetensors`.

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

> **Note**: Always use `sbatch_eval.sh` (accelerate-based, `multimeditron.toml` container). Do **not** use vLLM-based eval — vLLM cannot load our custom `multimodal` model type and has been abandoned.

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
| Eval results | `/users/surech/meditron/reports/lmms_eval_results/` |
| GPU utilization | `/users/surech/meditron/reports/gpu-util-<jobid>/` |

### Container environments

| EDF | Purpose |
|-----|---------|
| `~/.edf/multimeditron.toml` | Training and evaluation |
> **Critical**: All `sbatch_*.sh` scripts use `--environment=~/.edf/multimeditron.toml` to run inside the container. If you write your own launch script, you **must** include this flag. The login node has a bare Alpine system with no Python packages — commands like `python3`, `pip`, or `wandb` will fail outside the container.

### Required environment variables

Set these before submitting any job:

| Variable | Value | Why |
|----------|-------|-----|
| `HF_TOKEN` | Your HuggingFace token | Required to download model weights |
| `HF_HOME` | `/iopsstor/scratch/cscs/surech/hf` | Default `~/.cache/huggingface/` fills the 50 GB home quota |
| `WANDB_DIR` | `/capstor/store/cscs/swissai/a127/homes/surech/wandb` | WandB offline run storage (set by `sbatch_train.sh`) |
| `WORKING_DIR` | `/users/surech/meditron/MultiMeditron` | Repo root — YAML configs use `$WORKING_DIR/config/deepspeed.json` |
| `STORAGE_ROOT` | `/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow` | Arrow dataset root — YAML configs use `$STORAGE_ROOT/<dataset>` |
| `MODEL_ROOT` | `/iopsstor/scratch/cscs/surech/multimeditron/checkpoints` | Checkpoint output — YAML configs use `$MODEL_ROOT/freeze/...` |
### Partition limits

| Partition | Max nodes | Max wall time |
|-----------|-----------|---------------|
| `debug` | 2 | 30 min |
| `normal` | 128 | 12 hours |

### DeepSpeed configs

| Config | ZeRO Stage | CPU Offload | Notes |
|--------|-----------|-------------|-------|
| `config/deepspeed.json` | 3 | Yes (optimizer) | **Only config on `master`** |

> The YAML configs reference DeepSpeed via `$WORKING_DIR/config/deepspeed.json`. Make sure `$WORKING_DIR` is set (see [Prerequisites](#%EF%B8%8F-prerequisites)).

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

**Fix**: We use ZeRO-3 (`config/deepspeed.json`). ZeRO-3 partitions parameters across all ranks, fitting within 96 GB. The tradeoff is ~53 s/step at 128 nodes due to all-gather communication overhead.

---

### I/O bottleneck: high `dataloader_num_workers`

**Symptom**: Training speed degrades over time, GPUs show low utilization, data loading becomes the bottleneck. Potentially `OSError: [Errno 28] No space left on device` if `/tmp` fills up.

**Cause**: Our Arrow datasets are on shared Lustre (`/capstor/`). High `num_workers` (e.g. 16) across 512 ranks = 8,192 concurrent readers → swamps the parallel filesystem.

**Fix**: We set `dataloader_num_workers: 2` and `dataloader_prefetch_factor: 2` in Stage 2 configs. Stage 1 with fewer nodes (8) can tolerate higher values.

---

### `NODE_FAIL` / `CANCELLED+` by Slurm

**Symptom**: Job killed prematurely with `srun: error: Node nidXXXXXX has been marked DOWN`.

**Cause**: Hardware faults are routine on large-scale HPC runs (128 nodes = 512 GPUs). A single GPU memory error kills the entire job.

**Fix**: We save checkpoints frequently (`save_steps: 0.25` saves 4× per epoch) and resume. There is no automatic fault tolerance with DeepSpeed + torchrun.

---

### `TIMEOUT` — training exceeds wall time

**Symptom**: Job reaches the Slurm time limit before completing all steps. `sacct` shows state `TIMEOUT`.

**Cause**: At 128 nodes / ZeRO-3, Stage 2 takes ~23h for 1 epoch (~1,544 steps at 53 s/step). The normal partition limit is 12h.

**Fix**: We split training across two jobs using `resume_from_checkpoint`. With `save_steps: 0.25`, we always have a recent checkpoint.

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

---

### Bare-host `ModuleNotFoundError` (`click`, `accelerate`, etc.)

**Symptom**: `ModuleNotFoundError: No module named 'click'` or `No module named 'accelerate'` when running training or eval.

**Cause**: The job ran on the bare login/compute node without the container. Clariden nodes have a minimal Alpine system with no Python packages installed.

**Fix**: Ensure `--environment=~/.edf/multimeditron.toml` is included in your `sbatch` or `srun` command. All `sbatch_*.sh` scripts in this repo already include it. If you write a custom script, add:
```bash
srun --environment=~/.edf/multimeditron.toml ...
```

---

### `HF_HOME` quota error on model download

**Symptom**: `OSError: [Errno 122] Disk quota exceeded` or `No space left on device` during model weight download.

**Cause**: The default HuggingFace cache dir is `~/.cache/huggingface/`, which lives on the home filesystem (50 GB quota). A single 8B model download exceeds this.

**Fix**: Set `HF_HOME` to a scratch path before launching:
```bash
export HF_HOME=/iopsstor/scratch/cscs/surech/hf
```
`sbatch_train.sh` already exports this, but you must set it for any manual `srun` or `sbatch` commands too.

---

### Checkpoint path: parent directory vs checkpoint subdirectory

**Symptom**: Model loading fails with `FileNotFoundError` or `OSError: ... is not a valid model directory` when pointing at a training output directory.

**Cause**: Training output directories (e.g. `.../MultiMeditron-8B-attn-pep-end2end/`) contain checkpoint subdirectories (`checkpoint-100/`, `checkpoint-666/`, etc.) — the parent itself does not contain `model.safetensors` or `config.json`.

**Fix**: Always specify the full path to a checkpoint subdirectory:
```bash
# ✅ Correct — points to a specific checkpoint
.../MultiMeditron-8B-attn-pep-end2end/checkpoint-3063

# ❌ Wrong — parent directory has no model files
.../MultiMeditron-8B-attn-pep-end2end/
```
