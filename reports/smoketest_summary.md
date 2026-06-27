# Smoke Test: NanoVLM vs MultiMeditron

**Branch**: [`haaissa/MultiMeditron@nanovlm-test`](https://github.com/haaissa/MultiMeditron/tree/nanovlm-test)  
**Goal**: Verify that MultiMeditron can replicate the nanoVLM architecture and match its efficiency, then compare the two approaches on Clariden.

---

## Architecture Comparison

| | NanoVLM (Baseline) | MultiMeditron (nanoVLM-like) |
|---|---|---|
| **Vision encoder** | SigLIP2-512px | SigLIP2-512px |
| **Projection** | Pixel Shuffle (÷4) | Pixel Shuffle (÷4) |
| **Image tokens** | **64** | **64** |
| **LLM** | SmolLM2-360M | SmolLM2-360M |

The smoke test goal is to **replicate nanoVLM exactly** inside the MultiMeditron framework and verify training parity (same token count, same throughput, same loss trajectory). Both models use identical backbones — the difference is the training framework (nanoVLM's native PyTorch loop vs MultiMeditron's HuggingFace Trainer + DeepSpeed).

Pixel Shuffle (factor 4) reduces 1024 SigLIP2-512 patches → **64 tokens**, matching nanoVLM's efficiency. This is implemented in MultiMeditron via `projection_type: pixel_shuffle` + `pixel_shuffle_factor: 4` in the modality config (added in the `nanovlm-test` branch).

---

## Model Sources

| Model | HuggingFace ID | Notes |
|---|---|---|
| SmolLM2-360M-Instruct | [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | LLM backbone |
| SigLIP2-224 | [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224) | Phase 1 & 2 vision encoder |
| SigLIP2-512 | [`google/siglip2-base-patch16-512`](https://huggingface.co/google/siglip2-base-patch16-512) | nanoVLM-v2 vision encoder |
| BiomedCLIP | [`michel-ducartier/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`](https://huggingface.co/michel-ducartier/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Medical CLIP variant |
| nanoVLM pretrained | [`lusxvr/nanoVLM-222M`](https://huggingface.co/lusxvr/nanoVLM-222M) | Reference checkpoint for inference comparison |

**Clariden cached locations** (no download needed):

| Model | Path |
|---|---|
| SmolLM2-360M-Instruct | `/iopsstor/scratch/cscs/haaissa/hf/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951` |
| BiomedCLIP | `/iopsstor/scratch/cscs/haaissa/hf/hub/models--michel-ducartier--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/0f4526a545c7ed3311e3107e239a2d6d8816a43f` |

Note: SigLIP2 is fetched automatically from HF at runtime (not cached locally). Set `HF_HOME` appropriately.

---

## Data Sources

| Dataset | Description | Location on Clariden |
|---|---|---|
| The Cauldron (expert split) | ~1.7M multimodal QA samples | `/iopsstor/scratch/cscs/haaissa/cauldron_data/expert_cauldron_formatted.jsonl` |
| The Cauldron (full) | Full dataset for phase 2 | `/iopsstor/scratch/cscs/haaissa/cauldron_data/cauldron_formatted.jsonl` |
| Images (Cauldron) | Raw image files | `/iopsstor/scratch/cscs/haaissa/cauldron_data/images/` |
| LLaVA Pretrain | Standard VLM alignment data | `/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/llava_pretrain_cleaned` |

**References**:
- [HuggingFaceM4/the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) — main training dataset
- [HuggingFaceM4/FineVision — CoSyn_400k_chart](https://huggingface.co/datasets/HuggingFaceM4/FineVision/viewer/CoSyn_400k_chart/train) — chart/document visual split
- [nanoVLM blog post](https://huggingface.co/blog/nanovlm) — paper and architecture details
- [nanoVLM GitHub](https://github.com/huggingface/nanoVLM) — official reference implementation

---

## MultiMeditron Configs

All configs live in: [`config/`](https://github.com/haaissa/MultiMeditron/tree/nanovlm-test/config) (nanovlm-test branch)

### Phase 1 — Projector-Only Alignment (`config/nanovlm_phase1.yaml`)

```yaml
base_llm: "HuggingFaceTB/SmolLM2-360M-Instruct"
token_size: 960
tokenizer_type: "qwen3"
attachment_token: "<|image|>"
seed: 42

# Phase 1: Expert Alignment (projector-only training)
training_mode: "ALIGNMENT"
base_model: null  # Start from random projector

truncation: true
max_sequence_length: 2048

# === nanoVLM backbone: SigLIP2-224 ===
modalities:
  - model_type: "meditron_clip"
    clip_name: "google/siglip2-base-patch16-224"
    hidden_size: 960

loaders:
  - loader_type: "fs-image"
    modality_type: "image"
    base_path: "/iopsstor/scratch/cscs/haaissa/cauldron_data/images"

training_args:
  output_dir: "/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-phase1-alignment"
  run_name: "nanovlm-phase1"
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_steps: 10000
  learning_rate: 1.0e-3
  weight_decay: 0.0
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  bf16: true
  logging_steps: 100
  save_steps: 500
  remove_unused_columns: false
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: true
  dataloader_num_workers: 16
  dataloader_prefetch_factor: 4
  ddp_find_unused_parameters: false
  report_to: "none"

datasets:
  - packed_path: "/iopsstor/scratch/cscs/haaissa/cauldron_data/expert_cauldron_formatted.jsonl"
```

### Phase 2 — End-to-End Fine-tuning (`config/nanovlm_phase2.yaml`)

```yaml
base_llm: "HuggingFaceTB/SmolLM2-360M-Instruct"
token_size: 960
tokenizer_type: "qwen3"
attachment_token: "<|image|>"
seed: 42

# Phase 2: End-to-End Instruction Fine-tuning
training_mode: "END2END"
resume_from_checkpoint: true
base_model: "/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-phase2-finetune/checkpoint-12000"

truncation: true
max_sequence_length: 2048

# === nanoVLM backbone: SigLIP2-224 ===
modalities:
  - model_type: "meditron_clip"
    clip_name: "google/siglip2-base-patch16-224"
    hidden_size: 960

loaders:
  - loader_type: "fs-image"
    modality_type: "image"
    base_path: "/iopsstor/scratch/cscs/haaissa/cauldron_data/images"

training_args:
  output_dir: "/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-phase2-finetune"
  run_name: "nanovlm-phase2"
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_steps: 40000
  learning_rate: 5.0e-5
  weight_decay: 0.0
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  bf16: true
  logging_steps: 100
  save_steps: 500
  remove_unused_columns: false
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: true
  dataloader_num_workers: 16
  dataloader_prefetch_factor: 4
  ddp_find_unused_parameters: false
  report_to: "none"

datasets:
  - packed_path: "/iopsstor/scratch/cscs/haaissa/cauldron_data/cauldron_formatted.jsonl"
```

### Single Phase — NanoVLM Parity (`config/nanovlm_v2.yaml`)

This is the fully faithful replication of the nanoVLM architecture: SigLIP2-512px + Pixel Shuffle (÷4) → 64 tokens.

```yaml
base_llm: "HuggingFaceTB/SmolLM2-360M-Instruct"
token_size: 960
tokenizer_type: "qwen3"
attachment_token: "<|image|>"
seed: 42

# Single Phase Training: End-to-End from Step 0 — matching nanoVLM exactly
training_mode: "FULL"
base_model: null  # Start fresh, not from the blind checkpoint

# Exact nanoVLM learning rates (from models/config.py):
#   lr_mp = 0.00512   (projector — 5x higher to escape static noise fast)
#   lr_vision_backbone = 5e-5
#   lr_language_backbone = 5e-5
custom_lr:
  vision: 5.0e-5      # Matches nanoVLM lr_vision_backbone
  projector: 0.00512  # Matches nanoVLM lr_mp exactly
  # LLM uses default learning_rate below (matches lr_language_backbone)

truncation: true
max_sequence_length: 4096  # Matches nanoVLM lm_max_length

# === nanoVLM backbone: SigLIP2-512px + Pixel Shuffle ===
modalities:
  - model_type: "meditron_clip"
    clip_name: "google/siglip2-base-patch16-512"
    projection_type: "pixel_shuffle"
    pixel_shuffle_factor: 4
    hidden_size: 960

loaders:
  - loader_type: "fs-image"
    modality_type: "image"
    base_path: "/iopsstor/scratch/cscs/haaissa/cauldron_data/images"

datasets:
  - packed_path: "/iopsstor/scratch/cscs/haaissa/cauldron_data/expert_cauldron_formatted.jsonl"
    type: "jsonl"
    weight: 1.0

training_args:
  output_dir: "/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-v2-full"
  run_name: "nanovlm-v2-full-v2"
  max_steps: 40000                      # Matches nanoVLM max_training_steps
  per_device_train_batch_size: 2        # Matches nanoVLM batch_size
  gradient_accumulation_steps: 8        # Matches nanoVLM gradient_accumulation_steps
  learning_rate: 5.0e-5                 # Matches nanoVLM lr_language_backbone
  max_grad_norm: 1.0                    # Matches nanoVLM max_grad_norm
  weight_decay: 0.0
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  logging_steps: 100                    # Matches nanoVLM stats_log_interval
  save_steps: 500
  bf16: true
  remove_unused_columns: false
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: true
  dataloader_num_workers: 8
  dataloader_prefetch_factor: 2
  ddp_find_unused_parameters: false
  report_to: "wandb"
```

---

## NanoVLM Reference Configs

These are the Qwen3-4B + BiomedCLIP configs from the `qwen_biomedclip` cookbook, used in the MultiMeditron pipeline for comparison.

### Stage 1 — Alignment (`cookbook/sft/single_clip/qwen_biomedclip/stage1_alignment.yaml`)

```yaml
base_llm: Qwen/Qwen3-4B-Instruct-2507
base_model: null
attachment_token: <|reserved_special_token_0|>
tokenizer_type: qwen3
token_size: 2560

loaders:
  - loader_type: raw-image
    modality_type: image

modalities:
  - model_type: meditron_biomedclip
    clip_name: michel-ducartier/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    hidden_size: 2560
    trust_remote_code: true

training_mode: ALIGNMENT

datasets:
  - packed_path: $STORAGE_ROOT/llava_pretrain_cleaned
  - packed_path: $STORAGE_ROOT/pixmo_anything
  - packed_path: $STORAGE_ROOT/pixmo_cap
  - packed_path: $STORAGE_ROOT/medtrinity_conversations_1_formatted_alignment/

training_args:
  output_dir: $MODEL_ROOT/freeze/single_clip/MultiMeditron-Qwen-4B-Alignment-BiomedCLIP/
  dataloader_num_workers: 16
  dataloader_prefetch_factor: 4
  remove_unused_columns: false
  ddp_find_unused_parameters: false
  learning_rate: 1.0e-4
  bf16: true
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: true
  save_strategy: epoch
  max_grad_norm: 1.0
  run_name: MultiMeditron-Qwen-4B-Alignment-Generalist-Delimiter
  deepspeed: $WORKING_DIR/config/deepspeed.json
  accelerator_config:
    dispatch_batches: false
  lr_scheduler_type: "cosine_with_min_lr"
  lr_scheduler_kwargs:
    min_lr: 3.0e-5
  report_to: wandb
  logging_steps: 1
  weight_decay: 0.01
```

### Stage 2 — End-to-End (`cookbook/sft/single_clip/qwen_biomedclip/stage2_end2end.yaml`)

```yaml
base_llm: Qwen/Qwen3-4B-Instruct-2507
base_model: $MODEL_ROOT/freeze/single_clip/MultiMeditron-Qwen-4B-Alignment-BiomedCLIP/checkpoint-333
attachment_token: <|reserved_special_token_0|>
tokenizer_type: qwen3
token_size: 2560
truncation: True
max_sequence_length: 4096

loaders:
  - loader_type: raw-image
    modality_type: image

modalities:
  - model_type: meditron_biomedclip
    clip_name: michel-ducartier/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    hidden_size: 2560
    trust_remote_code: true

training_mode: END2END

datasets:
  - packed_path: $STORAGE_ROOT/BUSI
  - packed_path: $STORAGE_ROOT/COVID_US
  - packed_path: $STORAGE_ROOT/ct2
  - packed_path: $STORAGE_ROOT/iu_xray
  - packed_path: $STORAGE_ROOT/PMC_VQA_FULL
  - packed_path: $STORAGE_ROOT/medtrinity_conversations_1_formatted
  - packed_path: $STORAGE_ROOT/medtrinity_conversations_2_formatted
  - packed_path: $STORAGE_ROOT/image_mammoth
  - packed_path: $STORAGE_ROOT/llava_instruct

training_args:
  output_dir: $MODEL_ROOT/unfreeze/single_clip/MultiMeditron-Qwen-4B-End2End-BiomedCLIP
  dataloader_num_workers: 16
  dataloader_prefetch_factor: 4
  remove_unused_columns: false
  ddp_find_unused_parameters: false
  learning_rate: 1.0e-5
  bf16: true
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: true
  save_strategy: steps
  save_steps: 0.25
  max_grad_norm: 1.0
  run_name: MultiMeditron-Qwen-4B-End2End-Generalist-Delimiter
  deepspeed: $WORKING_DIR/config/deepspeed.json
  accelerator_config:
    dispatch_batches: false
  lr_scheduler_type: "cosine_with_min_lr"
  lr_scheduler_kwargs:
    min_lr: 3.0e-6
  report_to: wandb
  logging_steps: 1
  weight_decay: 0.01
```

---

## Existing Checkpoints (Clariden)

| Run | Path | Steps | Notes |
|---|---|---|---|
| nanoVLM-v2 full (haaissa) | `/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-v2-full/checkpoint-3000` | 3000 | SigLIP2-512 + PixelShuffle, flat safetensors = plain DDP |
| MM Qwen4B Phase1 (haaissa) | `/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/freeze/single_clip/MultiMeditron-Qwen-4B-Alignment-BiomedCLIP/checkpoint-2662` | 2662 | ZeRO-3 format |

---

## How to Rerun

### Prerequisites

```bash
# Repo (nanovlm-test branch has the pixel_shuffle projector)
git clone https://github.com/haaissa/MultiMeditron.git
cd MultiMeditron
git checkout nanovlm-test
```

### Clariden (SLURM)

```bash
export HF_TOKEN=<your_token>

# NanoVLM parity run (SigLIP2-512 + PixelShuffle → 64 tokens)
sbatch --partition debug --nodes 1 --time 00:30:00 \
  sbatch_train.sh config/nanovlm_v2.yaml

# Two-phase approach (SigLIP2-224 → 196 tokens)
# Phase 1 first:
sbatch --partition normal --nodes 4 --time 04:00:00 \
  sbatch_train.sh config/nanovlm_phase1.yaml

# Then Phase 2 (update base_model path to phase1 checkpoint):
sbatch --partition normal --nodes 4 --time 12:00:00 \
  sbatch_train.sh config/nanovlm_phase2.yaml
```

Key env vars set by `sbatch_train.sh`:
- `HF_HOME=/capstor/store/cscs/swissai/a127/meditron/hf_cache` (use `HF_HOME=/iopsstor/scratch/cscs/haaissa/hf` to use haaissa's cache with SmolLM2 pre-downloaded)
- `WANDB_MODE=offline`
- Container: `~/.edf/multimeditron.toml`

### Local / Single GPU (smoke test — 20 steps)

See `cookbook/sft/single_clip/qwen_biomedclip/stage1_alignment_smoketest_zero2.yaml` for a minimal 20-step smoke test using SmolLM2-360M + BiomedCLIP + ZeRO-2:

```bash
export HF_TOKEN=<your_token>
sbatch --partition debug --nodes 1 --time 00:15:00 \
  sbatch_train.sh cookbook/sft/single_clip/qwen_biomedclip/stage1_alignment_smoketest_zero2.yaml
```

---

## Key Findings from the Smoke Test

1. **Pixel Shuffle is required for nanoVLM parity**: without it, MultiMeditron feeds 1024 tokens to SmolLM2 (vs nanoVLM's 64), making it ~16× slower per image. Always use `projection_type: pixel_shuffle` + `pixel_shuffle_factor: 4` with SigLIP2-512.
2. **Pixel Shuffle is now supported** in MultiMeditron via `projection_type: "pixel_shuffle"` + `pixel_shuffle_factor` in the modality config (added in the `nanovlm-test` branch).
3. **ZeRO-2 works** for single-node runs (see `config/deepspeed_zero2.json`); ZeRO-3 was causing crashes due to `deepspeed.zero.Init()` being called unconditionally even for ZeRO-2 configs (now fixed in `src/multimeditron/cli/train.py`).
4. **nanoVLM reference checkpoint**: [`lusxvr/nanoVLM-222M`](https://huggingface.co/lusxvr/nanoVLM-222M) (SigLIP-224 + SmolLM2-135M, ~6h on 1× H100, trained on 1.7M samples from the Cauldron).
