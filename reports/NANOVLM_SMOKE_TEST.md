# NanoVLM Smoke Test — Reference Document

> **Purpose**: This document captures everything needed to run and understand the NanoVLM
> smoke test without confusing it with MultiMeditron-8B. It records the precise configs,
> architecture differences (LLM/projector only — vision experts and gating are intentionally
> excluded from the comparison), dataset setup, and separate storage paths.

---

## 1. What Is the Smoke Test?

The goal is to check whether our LLM backbone (Meditron3-8B / LLaMA 3.1-8B) is underperforming
**relative to an optimally-trained reference model** trained under the same framework.

The reference is a minimal VLM — a NanoVLM-style model — built from a small, well-tested
LLM backbone (`SmolLM2-360M-Instruct`) and a lightweight vision encoder (`SigLIP2-base`),
trained end-to-end on a large general vision-language dataset (The Cauldron / FineVision).

If the smoke test LLM does not learn visual grounding at all, it can indicate a bug.
If it learns well but our 8B model does not, it points to an LLM-side training issue.

### Why MMSTAR?

[MMStar](https://github.com/MMStar-Benchmark/MMStar) is a clean, leakage-free VQA benchmark
used originally by nanoVLM to measure general vision-language understanding. It is a good
sanity check for whether the model has learned multimodal alignment at all.

nanoVLM-222M (SigLIP + SmolLM2-135M, ~6h on one H100, ~1.7M samples) achieves **35.3% on MMStar**.
That is the reference bar for a well-trained minimal VLM on this benchmark.

---

## 2. Config File Locations

All NanoVLM test configs live in the `nanovlm-test` branch of haaissa's fork:

```
haaissa/MultiMeditron @ nanovlm-test
└── config/
    ├── nanovlm_phase1.yaml   ← Stage 1: projector-only alignment
    ├── nanovlm_phase2.yaml   ← Stage 2: end-to-end fine-tuning
    └── nanovlm_v2.yaml       ← Single-phase (FULL) matching nanoVLM exactly
```

Branch URL: https://github.com/haaissa/MultiMeditron/tree/nanovlm-test/config

Data prep scripts:
```
scripts/
├── prepare_nanovlm_data.py   ← Simple streaming download + format (early version)
└── prepare_data.py           ← Production pipeline: bulk HF download → multi-proc processing
```

---

## 3. Precise Config — Each YAML File

### 3.1 `nanovlm_phase1.yaml` — Alignment (projector only)

| Parameter | Value |
|-----------|-------|
| `training_mode` | `ALIGNMENT` |
| `base_model` | `null` (random projector init) |
| Vision encoder | `google/siglip2-base-patch16-224` |
| `hidden_size` | 960 |
| `projection_type` | default (linear) |
| LLM | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| `tokenizer_type` | `qwen3` |
| `attachment_token` | `<\|image\|>` |
| `token_size` | 960 |
| `max_sequence_length` | 2048 |
| `max_steps` | 10 000 |
| `learning_rate` | 1.0e-3 |
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 8 → effective batch = 16 |
| `warmup_ratio` | 0.03 |
| `lr_scheduler_type` | cosine |
| `bf16` | true |
| Dataset (packed) | `/iopsstor/scratch/cscs/haaissa/cauldron_data/expert_cauldron_formatted.jsonl` |
| Images base path | `/iopsstor/scratch/cscs/haaissa/cauldron_data/images` |
| Output dir | `/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-phase1-alignment` |
| `report_to` | none |

---

### 3.2 `nanovlm_phase2.yaml` — End-to-End Fine-tuning

| Parameter | Value |
|-----------|-------|
| `training_mode` | `END2END` |
| `resume_from_checkpoint` | `true` (from `nanovlm-phase2-finetune/checkpoint-12000`) |
| Vision encoder | `google/siglip2-base-patch16-224` |
| `hidden_size` | 960 |
| LLM | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| `tokenizer_type` | `qwen3` |
| `attachment_token` | `<\|image\|>` |
| `token_size` | 960 |
| `max_sequence_length` | 2048 |
| `max_steps` | 40 000 |
| `learning_rate` | 5.0e-5 (single LR for all parameters) |
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 8 → effective batch = 16 |
| `warmup_ratio` | 0.03 |
| `lr_scheduler_type` | cosine |
| `bf16` | true |
| Dataset (packed) | `/iopsstor/scratch/cscs/haaissa/cauldron_data/cauldron_formatted.jsonl` |
| Images base path | `/iopsstor/scratch/cscs/haaissa/cauldron_data/images` |
| Output dir | `/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-phase2-finetune` |
| `report_to` | none |

---

### 3.3 `nanovlm_v2.yaml` — Single-Phase FULL (matches nanoVLM exactly)

This is the canonical "apple-to-apple" reproduction config.

| Parameter | Value | nanoVLM reference |
|-----------|-------|-------------------|
| `training_mode` | `FULL` | single-phase end-to-end |
| `base_model` | `null` (start fresh) | fresh |
| Vision encoder | `google/siglip2-base-patch16-512` | SigLIP2-512 |
| `projection_type` | `pixel_shuffle` | pixel shuffle |
| `pixel_shuffle_factor` | 4 | 4 |
| `hidden_size` | 960 | 960 |
| LLM | `HuggingFaceTB/SmolLM2-360M-Instruct` | SmolLM2-360M |
| `tokenizer_type` | `qwen3` | qwen3 |
| `attachment_token` | `<\|image\|>` | `<\|image\|>` |
| `token_size` | 960 | 960 |
| `max_sequence_length` | 4096 | `lm_max_length = 4096` |
| `max_steps` | 40 000 | `max_training_steps = 40000` |
| LR — vision | 5.0e-5 | `lr_vision_backbone = 5e-5` |
| LR — projector | 0.00512 | `lr_mp = 0.00512` |
| LR — LLM | 5.0e-5 | `lr_language_backbone = 5e-5` |
| `per_device_train_batch_size` | 2 | `batch_size = 2` |
| `gradient_accumulation_steps` | 8 | `gradient_accumulation_steps = 8` |
| `max_grad_norm` | 1.0 | `max_grad_norm = 1.0` |
| `warmup_ratio` | 0.03 | — |
| `lr_scheduler_type` | cosine | cosine |
| `logging_steps` | 100 | `stats_log_interval = 100` |
| `bf16` | true | mixed precision |
| Dataset (packed) | `/iopsstor/scratch/cscs/haaissa/cauldron_data/expert_cauldron_formatted.jsonl` | The Cauldron / FineVision |
| Images base path | `/iopsstor/scratch/cscs/haaissa/cauldron_data/images` | |
| Output dir | `/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-v2-full` | |
| `report_to` | `wandb` | wandb |
| `run_name` | `nanovlm-v2-full-v2` | — |

---

## 4. Architecture Difference: NanoVLM Test vs MultiMeditron-8B

> Vision experts and gating network are **excluded** from this comparison — they are
> orthogonal to the question being tested (LLM backbone quality).

| Component | NanoVLM test (smoke test) | MultiMeditron-8B (production) |
|-----------|--------------------------|-------------------------------|
| **LLM backbone** | `SmolLM2-360M-Instruct` (360 M params) | `Meditron3-8B` = LLaMA 3.1-8B fine-tuned on medical text (8 B params) |
| **LLM architecture** | LLaMA-style decoder, 24 layers, 960 hidden dim, 15 heads | LLaMA-3.1 decoder, 32 layers, 4096 hidden dim, 32 heads |
| **LLM tokenizer** | qwen3 tokenizer | LLaMA-3.1 tokenizer |
| **Attachment token** | `<\|image\|>` | `<\|reserved_special_token_0\|>` |
| **Vision encoder** | SigLIP2-base-patch16-224 (or 512 in v2) | 7× CLIP ViT-B/32 expert encoders *(excluded from diff)* |
| **Projection module** | Pixel shuffle (factor 4) → linear layer | 7 × MLP (768 → 4096) per-expert projectors (PEP) |
| **Fusion** | Direct token concatenation (image tokens prepended to text tokens) | Cross-Attention fusion (ATTN-PEP): Generalist patches = Query, specialist patches weighted by gating = Key/Value |
| **Gating** | None — single encoder | ResNet-50 7-class classifier *(excluded from diff)* |
| **Total model size** | ~480 M (SigLIP-B + SmolLM2-360M + projector) | ~8.4 B (Meditron3-8B + 7×CLIP + projectors + fusion) |
| **Training phases** | Phase 1: alignment (projector only, 10k steps) + Phase 2: end-to-end (40k steps); **or** single FULL (v2, 40k steps) | Stage 1: alignment (projector only) + Stage 2: end-to-end |
| **Training data** | HuggingFaceM4/FineVision_concat_shuffled_2 (~1.7M general VQA samples) | MultiMediset (medical imaging, domain-specific) |
| **Domain** | General vision-language (natural images, charts, OCR, …) | Medical imaging (CT, MRI, ultrasound, X-ray, pathology, ophthalmology, dermatology) |
| **Benchmark** | MMSTAR (general VQA) | GMAI, SLAKE, PathVQA (medical VQA) |

### Key LLM difference (the core of the smoke test)

The smoke test uses a **22× smaller LLM** (`SmolLM2-360M`) to verify that:
1. The MultiMeditron training pipeline produces a functional VLM when the LLM is small and well-pretrained.
2. If the 360M model learns alignment but our 8B model does not, the problem is in the 8B training setup (learning rate, data, projector initialization).
3. If neither learns alignment, the problem is in the pipeline/code itself.

---

## 5. Dataset — The Cauldron / FineVision

| Item | Details |
|------|---------|
| **HF repo used** | `HuggingFaceM4/FineVision_concat_shuffled_2` |
| **Underlying dataset** | HuggingFaceM4/FineVision (superset of *The Cauldron*) |
| **Cauldron viewer** | https://huggingface.co/datasets/HuggingFaceM4/FineVision/viewer/CoSyn_400k_chart/train |
| **Size** | ~1.7 M image-question-answer pairs |
| **Content** | 50+ general VQA sub-datasets: charts, OCR, science diagrams, natural images, screen understanding, etc. |
| **Format after preprocessing** | JSONL (`cauldron_formatted.jsonl` for LLM; `expert_cauldron_formatted.jsonl` for vision) |
| **Preprocessing script** | `scripts/prepare_data.py` (bulk HF-CLI download → multi-process Arrow mapping) |
| **Quality filtering** | Turns filtered by `relevance_rating ≥ 1`, `image_correspondence_rating ≥ 1`, `visual_dependency_rating ≥ 1`, `formatting_rating ≥ 1` |

### Cluster data paths (haaissa's scratch — do NOT mix with MultiMeditron paths)

```
/iopsstor/scratch/cscs/haaissa/
├── cauldron_data/
│   ├── images/                          ← extracted JPEG images (~image_0.jpg … image_N.jpg)
│   ├── expert_cauldron_formatted.jsonl  ← modalities + conversations (used for vision/alignment)
│   └── cauldron_formatted.jsonl         ← modalities + text blob (used for LLM fine-tuning)
└── hf/
    └── FineVision_local/                ← raw parquet files downloaded by huggingface-cli
```

> **Important**: All NanoVLM data lives under `/iopsstor/scratch/cscs/haaissa/`.
> MultiMeditron-8B data lives under `/iopsstor/scratch/cscs/surech/`.
> **Do not mix these paths** in config files.

---

## 6. Checkpoint Paths

| Experiment | Path |
|-----------|------|
| Phase 1 (alignment) | `/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-phase1-alignment/` |
| Phase 2 (end-to-end) | `/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-phase2-finetune/` |
| V2 single-phase | `/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-v2-full/` |

MultiMeditron-8B checkpoints remain under `/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/`.

---

## 7. Evaluation

### Benchmark

| Benchmark | Used by | Description |
|-----------|---------|-------------|
| **MMSTAR** | NanoVLM smoke test | 1 500 leakage-free general VQA questions across 6 categories. Reference: nanoVLM-222M scores 35.3%. |
| **GMAI** | MultiMeditron-8B | 4 550 clinical VQA questions across 10 medical imaging modalities. |
| **SLAKE** | MultiMeditron-8B | 642 medical VQA questions (radiology, binary + open-ended). |
| **PathVQA** | MultiMeditron-8B | ~6 700 histopathology VQA questions (yes/no + open-ended). |

### MMSTAR reference bar

| Model | MMSTAR |
|-------|--------|
| nanoVLM-222M (SigLIP-B + SmolLM2-135M, 1.7M samples, 6h on 1×H100) | 35.3% |
| nanoVLM-450M (published, newer pipeline) | see [lusxvr/nanoVLM-450M](https://huggingface.co/lusxvr/nanoVLM-450M) |
| Our smoke test (SmolLM2-360M, SigLIP2-base) | TBD |

---

## 8. Code Changes in the `nanovlm-test` Branch

The branch is **4 commits ahead of EPFLiGHT/MultiMeditron:master**:

| Commit | Change |
|--------|--------|
| `dfaf726` | `src/multimeditron/model/` + `train/` — add pixel shuffle projector support (`projection_type: "pixel_shuffle"`, `pixel_shuffle_factor`) |
| `7617d1c` | `scripts/prepare_data.py` + `src/multimeditron/cli/` + `dataset/` — production data preprocessing pipeline for Cauldron/FineVision |
| `1670367` | `config/nanovlm_phase1.yaml`, `nanovlm_phase2.yaml`, `nanovlm_v2.yaml` — NanoVLM training configs |
| (earlier) | `scripts/prepare_nanovlm_data.py` — earlier simpler streaming version of data prep |

---

## 9. References

- **NanoVLM repo**: https://github.com/huggingface/nanoVLM
- **NanoVLM blog post**: https://huggingface.co/blog/nanovlm
- **Published nanoVLM-450M**: https://huggingface.co/lusxvr/nanoVLM-450M
- **FineVision dataset (Cauldron superset)**: https://huggingface.co/datasets/HuggingFaceM4/FineVision/viewer/CoSyn_400k_chart/train
- **FineVision_concat_shuffled_2** (used for training): `HuggingFaceM4/FineVision_concat_shuffled_2`
- **MMSTAR benchmark**: https://github.com/MMStar-Benchmark/MMStar
- **Test branch**: https://github.com/haaissa/MultiMeditron/tree/nanovlm-test
