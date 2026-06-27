# MultiMeditron — Contribution Report

**Author:** Surechen Rajendram (surech, ShinUrech)
**Period covered:** February 2026 – May 2026
**Repository:** EPFLiGHT/MultiMeditron — branch `add-ophthalmology-and-dermatology-experts` (now `Multimeditron-debug`)
**Cluster:** Clariden CSCS, GH200 ARM64, account `a127`

---

## Overview

This report documents all engineering, experimentation, and analysis contributions made to the MultiMeditron project. The work spans four main areas:

1. Extending the model from 5 to 7 expert encoders (Ophthalmology and Dermatology)
2. Building and validating a 7-class gating network training pipeline
3. Running and analysing full-scale evaluations
4. Implementing sequence packing with a Flash Attention 2 correctness fix and associated infrastructure

---

## 1. Extension to 7-Expert Architecture

**Dates:** March 3 – March 23, 2026
**Commits:** `1470922`, `a96eb03`, `f51751f`, `1f8a576`, `bb7c65e`, `840a631`, `2402eaa`, `f7abc14`

### Context

The original MultiMeditron used 5 CLIP expert encoders: CT, MRI, Ultrasound, X-ray, and a Generalist. Ophthalmology and Dermatology experts had been trained separately but were not yet integrated into the end-to-end training pipeline.

### Changes Made

**Training configs (`cookbook/sft/moe/attn/pep/`):**
- `stage1_alignment.yaml`: Added Ophthalmology and Dermatology expert paths; updated CSCS-specific storage paths for the base LLM and CLIP weights.
- `stage2_end2end.yaml`: Extended expert list from 5 to 7; updated learning rate, gradient accumulation, and ZeRO config paths for the 7-expert run on 128 nodes.

**Model bug fixes (`src/multimeditron/`):**
- `model/model.py`: Fixed `AutoConfig.from_pretrained` call — the parameter is `dtype=`, not `torch_dtype=`. The incorrect keyword silently fell back to fp32 during model initialisation, causing incorrect mixed-precision behaviour. Also fixed `AutoConfig` and `AutoModel` registration for the `multimodal` model type.
- `cli/train.py`: Fixed CUDA index-out-of-bounds in expert selection when the expert count changed. Extracted model build into `_build_model()` helper. Added safe checkpoint resume with existence check.

**Cluster infrastructure:**
- `sbatch_train.sh`: Created the main SLURM launcher for the `multimeditron.toml` EDF container; hardcoded `NCCL_NET_GDR_LEVEL=0` and Slingshot NCCL environment; added `PYTHONPATH` and `HF_HOME` exports.
- `config/deepspeed_fast.json`: Production ZeRO-3 DeepSpeed config used for all Stage 1 and Stage 2 runs.
- `sbatch_train_stage2.sh`: Dedicated Stage 2 launcher with fixed 128-node count (required by ZeRO-3 shard count).

**Pipeline validation:**
- `cookbook/sft/moe/attn/pep/test_stage1.yaml`, `test_stage2.yaml`: Debug configs running 2 steps on 2 nodes to confirm the full pipeline (data loading → model forward → loss → checkpoint) executes without errors.
- `sbatch_debug.sh`, `sbatch_test_pipeline.sh`: Launchers for pipeline validation jobs.

---

## 2. Gating Network Training Pipeline (7-Class)

**Dates:** March 21 – April 16, 2026
**Commits:** `2ceb21a`, `e4d4edf`, `c818b35`, `e3405b9`, `fad9641`, `4353451`

### Context

The gating network is a lightweight classifier built on top of the frozen CLIP ViT-B/32 backbone. It maps image embeddings to a probability distribution over experts. Expanding from 5 to 7 classes required retraining the gating network from scratch with new datasets for Ophthalmology and Dermatology.

### Changes Made

**Training script (`scripts/train_gating.py`):** Full rewrite from a 2-class placeholder to a production 7-class training pipeline (729 lines). Key features:
- Loads per-modality Arrow datasets from configurable paths; applies class-proportional sampling to handle the large imbalance between modality dataset sizes.
- Backbone: frozen CLIP ViT-B/32 (`openai/clip-vit-base-patch32`). Trainable head: single linear layer over the 512-dim CLS embedding.
- Optimizer: AdamW with cosine LR schedule and warmup. WandB logging for all train/val metrics.
- Best checkpoint saved based on validation accuracy.

**3-way train/val/test split (commit `e3405b9`):** Added a held-out test set carved off before training begins. This prevents the validation set from influencing model selection and provides an unbiased accuracy estimate. The test fraction is configurable via `test_split` in `config/gating_7class.yaml` (default 0.1). After training completes, the best checkpoint is evaluated on the test set and results written to `<output_dir>/test_results.json`.

**Configuration (`config/gating_7class.yaml`):** 104-line YAML config specifying dataset paths for all 7 modalities, training hyperparameters, output directory, and test_split.

**SLURM launchers:**
- `sbatch_train_gating.sh`: Standard training job (81 lines) with WandB offline sync pattern.
- `sbatch_train_gating_debug.sh`: Short debug job for quickly verifying the pipeline.

**Test script (`scripts/test_gating.py`):** Loads a trained gating checkpoint and evaluates routing accuracy per modality on 200 images each, printing a per-expert confusion table.

**Docstrings (`src/multimeditron/model/attention.py`, `model/modalities/moe/gating.py`):** Added Google-style docstrings to `CrossAttention` and `GatingNetwork` classes documenting all parameters, inputs, and outputs.

### Results

The trained 7-expert gating network was validated on 5 of 7 modalities (MRI and Generalist could not be tested — no standalone MRI-only Arrow dataset was found on the cluster):

| Dataset | Predicted Expert | Top-1 % |
|---|---|---|
| CT (ct2) | CT | 100% |
| X-ray (iu_xray) | X-ray | 100% |
| Ultrasound (BUSI) | Ultrasound | 99% |
| Eye (eye_dataset) | Ophthalmology | 100% |
| Skin (skin_dataset) | Skin | 99% |

By comparison, the previous 5-expert gating network had a critical routing bug: it sent 96.5% of CT images to the Ultrasound expert (0% to CT). The retraining completely resolved this.

---

## 3. Multi-Node Evaluation Pipeline and Benchmark Analysis

**Dates:** March 12 – April 16, 2026
**Commits:** `86b363d`, `c818d35`, `16d6ad1`, `cff5daf`, `04ec7ce`, `8206a2a`, `4353451`

### Changes Made

**Eval infrastructure:**
- `sbatch_eval.sh`: Multi-node evaluation launcher using `accelerate`-based lmms-eval with the `multimeditron.toml` container. Supports configurable node count, checkpoint path, and benchmark list. Proven working; vLLM-based eval was explored and abandoned (vLLM cannot load the custom `multimodal` model type).
- `sbatch_eval_vllm.sh`: Added initially, then deleted after confirming vLLM cannot handle custom model types.
- Confirmed eval timing: 16 nodes ≈ 50 min, 4 nodes ≈ 3.5 h for the three standard benchmarks (GMAI 4,550 samples, SLAKE 642 samples, PathVQA ~6,700 samples).

**lmms-eval submodule (`third-party/lmms-eval`):**
- Bumped submodule to include a `decord` import fix (eval container does not have `decord` installed — the fix lazy-imports it inside `try/except`).
- Added per-modality subtask definitions for GMAI and SLAKE: `gmai_ophthalmology`, `gmai_dermatology`, `slake_mri`, `slake_xray`, `slake_ct`.

**Dataset audit (`cookbook/DATA_AUDIT.md`):** Systematic audit of all 7 modality training datasets. Identified duplicate image files across splits and across modalities. Duplicate fraction varied from <1% (CT) to ~8% (Dermatology). Recommended deduplication strategy per dataset.

**Evaluation analysis (`reports/EVAL_ANALYSIS.md`, `scripts/compare_modality_results.py`):** Full quantitative comparison of 5-expert checkpoint-3063 vs 7-expert checkpoint-800 across GMAI, SLAKE, and PathVQA (see Section 3.1).

**Gating routing analysis (`scripts/gating_routing_analysis.py`, `scripts/pathvqa_routing_analysis.py`):**
- `gating_routing_analysis.py`: Loads both 5-expert and 7-expert gating checkpoints; runs 200 images per modality; prints top-1 routing accuracy and average gating weight per expert.
- `pathvqa_routing_analysis.py`: Specifically investigates the PathVQA binary question regression — runs 500 PathVQA images through both gating checkpoints and prints a full routing distribution table.
- `sbatch_gating_analysis.sh`: SLURM launcher for the routing analysis job.

### 3.1 Evaluation Results Summary

Full tables are in `reports/EVAL_ANALYSIS.md`. Key findings:

**Top-level benchmarks (5-expert ckpt-3063 vs 7-expert ckpt-800):**

| Benchmark | 5-expert | 7-expert | Δ |
|---|---|---|---|
| GMAI | 29.6% | 31.1% | +1.5% ✅ |
| SLAKE overall | 29.6% | 30.6% | +1.0% ✅ |
| SLAKE yes/no | 51.1% | 51.1% | 0.0% |
| PathVQA overall | 30.1% | 24.4% | −5.7% ❌ |
| PathVQA yes/no | 58.6% | 47.1% | −11.5% ❌ |

**Per-modality highlights:**
- Dermatology (GMAI): +8.7% from 5-expert to 7-expert — the dedicated SkinExpert provides clear benefit.
- MRI (SLAKE yes/no): +8.8% — improved routing after CT images no longer pollute the Ultrasound path.
- X-ray (SLAKE yes/no): −14.2% — regression under investigation; likely a small-sample artefact (≈99 binary questions).
- Ophthalmology (GMAI): −3.5% — the new expert may be more specialised on slit-lamp images while the GMAI subtask includes OCT and fundus.

**PathVQA binary regression root cause:** The 7-expert model predicts "No" 89.3% of the time on PathVQA binary questions (true positive rate for "Yes" = 10.8%). This is caused by a routing failure: the 7-expert gating sends 52.2% of histopathology images to the Skin expert and 37.2% to MRI, compared to the 5-expert model which sent 98.2% to the Generalist. Both Skin and MRI experts are out-of-distribution for histopathology and default to "No" when uncertain. There is no dedicated histopathology expert in either model.

---

## 4. GPU Utilisation Analysis and Training Infrastructure Improvements

**Dates:** March 17 – May 22, 2026
**Commits:** `7e33259`, `8206a2a` (GPU_UTILIZATION_ANALYSIS.md), `4ea5dea`, `c3d0c33`

### 4.1 GPU Utilisation Profiling

Instrumented all training jobs with embedded `nvidia-smi dmon` calls inside `sbatch_train.sh` to collect per-GPU SM% and memory bandwidth % every 5 seconds. Results for 128-node (512 GPU) Stage 2 job (job 1709145):

| Metric | 128 nodes (512 GPUs) | 2 nodes (8 GPUs) |
|---|---|---|
| SM% | 99.1% | 77.3% |
| Memory bandwidth% | 1.0% | 6.6% |
| Samples/GPU-hour | 1,115 | — |
| sec/step | 51.7 s | — |

**MFU (Model FLOP Utilisation):** 1.5% at 128 nodes. Well-optimised large-scale training typically achieves 35–55% MFU. The SM=99% / MEM BW=1% signature identifies the root cause: ZeRO-3 at 512 ranks dominates runtime with NCCL allgather/reduce-scatter collectives, which run on the SM but do not access HBM. The NCCL network I/O serialises all 512 ranks, and at 51.7 seconds/step, the GPU is waiting on the network the vast majority of the time.

Theoretical speedup from eliminating ZeRO-3 overhead: ~25× (from 51.7 s to ~2.0 s/step assuming 40% MFU). `reports/GPU_UTILIZATION_ANALYSIS.md` documents this analysis in full.

**NCCL multi-node fix (`cookbook/multi-node-nccl-fix.md`):** Documented the `NCCL_NET_GDR_LEVEL=0` workaround required on Clariden GH200 nodes to prevent GPUDirect RDMA failures on Slingshot. Without this, distributed training hangs during the first NCCL collective. Added this environment variable to `sbatch_train.sh` via the EDF container.

### 4.2 Sequence Packing Implementation

**Dates:** May 22, 2026
**Commits:** `30303e6`, `4ea5dea`
**Lines of code:** ~918 insertions

Sequence packing addresses a primary source of wasted compute: in standard training, each batch is padded to the length of the longest sequence in the batch. With highly variable-length medical VQA data, padding fractions above 60% are common.

#### Data collator (`src/multimeditron/model/data_loader.py`)

Added `_pack_sequences()`: a greedy first-fit bin-packing algorithm that assigns variable-length sequences to fixed-length bins (size = `max_seq_len`). For each bin, the collator produces a standard `input_ids` tensor padded to `max_seq_len` and a `cu_seqlens` tensor encoding the cumulative sequence lengths of the real sub-sequences within the bin. The format is `[0, s₀, s₀+s₁, ..., Σsᵢ, max_seq_len]` where the final entry marks the padding boundary.

The collator is activated by `pack_sequences: true` in the training YAML config. When disabled, the original padding collator is used unchanged.

#### Flash Attention 2 correctness fix (`src/multimeditron/train/trainer.py`)

Flash Attention 2 in HuggingFace Transformers derives `cu_seqlens` from `attention_mask.sum(-1)`, treating each batch element as a single causal sequence. When applied to packed bins, this causes cross-sample attention leakage: tokens from later sub-sequences in a bin can attend to tokens from earlier sub-sequences.

The fix monkey-patches `_get_unpad_data` in `transformers.modeling_flash_attention_utils` with a replacement that:
1. Checks a thread-local context object `_PACKING_CONTEXT` for pre-computed `cu_seqlens`.
2. If present (packed mode), extracts sub-sequence lengths from the stored `cu_seqlens` list and constructs new boundaries for `flash_attn_varlen_func` that respect individual sequence boundaries within each bin.
3. If absent (non-packed mode), falls back to the standard HuggingFace behaviour.

The thread-local design ensures the patch is active only during the forward pass inside `compute_loss`, where it is set via a `try/finally` block. This is safe under DeepSpeed and HuggingFace `Trainer`.

A secondary bug was also fixed: `torch.cumsum` upcasts `int32` input to `int64`, but `flash_attn_varlen_func` requires `int32` for its `cu_seqlens_q` argument. Added explicit `dtype=torch.int32` to the `cumsum` call to prevent the silent type promotion.

#### Startup verification

At training startup, `trainer.py` prints the locations in the module graph where `_get_unpad_data` was patched:
```
[PACKING PATCH] locations=['transformers.modeling_flash_attention_utils._get_unpad_data', ...]
```
This confirms the patch is active before any training step runs.

#### Unit tests (`tests/test_packing.py`)

368-line test suite covering:
- Bin-packing correctness: sequences assigned correctly, no sequence split across bins, bin sizes do not exceed `max_seq_len`.
- `cu_seqlens` format: correct dtype (int32), correct values, padding sentinel present.
- Round-trip: packed batch can be unpacked back to original sequences losslessly.
- Edge cases: single very long sequence, all sequences identical length, one sequence per bin.

#### Ancillary infrastructure

- `config/deepspeed_zero{1,2,3}.json`: Separate DeepSpeed configs for each ZeRO optimisation stage, enabling controlled comparison runs.
- `sbatch_test_packing.sh`: 4-node SLURM job that runs a short packed training run and confirms `[PACKING PATCH]` appears in the log.
- `src/multimeditron/profiling.py`: Utility functions for measuring training throughput (samples/sec, tokens/sec, MFU).
- `src/multimeditron/model/projectors/pixel_shuffle.py`: Pixel-shuffle projector variant for future use.

#### Known issue: training collapse in job 2340502

A packed training run (job 2340502, ZeRO-2, 4 nodes, 16 GPUs) collapsed from step 3 onward:
- Steps 1–2: `loss ≈ 15.5`, `grad_norm ≈ 310`.
- Steps 3–200: `loss = 0.0` exactly, `grad_norm = 1.4142135` (= √2).

The constant √2 gradient norm is a suspicious mathematical constant suggesting the gradient signal has vanished and only the weight decay term remains. Root cause is under investigation. Most likely candidate: all labels masked to `-100` (e.g. due to a bug in the packing collator's label alignment), producing a zero-loss batch with no true gradient. The WandB summary metric `train_loss = 0.152` is misleading — it is the arithmetic mean of the 200 steps, dominated by two high-loss warm-up steps.

The unpacked baseline run (job 2340681) was submitted in parallel to provide a reference loss curve for comparison. The packed vs. unpacked MFU comparison will be completed once the collapse is resolved.

### 4.3 Training Config Coverage

**Commit:** `c3d0c33` (43 files changed, 1,675 insertions)

Added a complete matrix of training configs across ZeRO stages and node counts:

- **Smoketest configs** (1–2 nodes, 10 steps): Validate that the entire pipeline (data loading, model forward, loss, backward, checkpoint) runs without error. One config per ZeRO stage (1/2/3) and per gating variant (attn/avg/cat × pep/shared).
- **Sanitycheck configs** (4 nodes, 200 steps): Short runs to check loss decreases, confirm GPU memory budget, and compare different ZeRO stages head-to-head.
- **`stage2_sanitycheck_zero2_packed.yaml`**: Paired sanitycheck specifically for validating the packed collator against an unpacked ZeRO-2 baseline — same hyperparameters, only `pack_sequences` differs.
- **`stage2_end2end_zero2.yaml`**: Production-scale Stage 2 config using ZeRO-2 instead of ZeRO-3, intended as the next full training run once the packing collapse is resolved.

---

## 5. Documentation and Analysis Reports

**Commits:** `7b1b715`, `1238551`, `8206a2a`, `4d783f6`, `d45cad3`

### Internal reports

| File | Contents |
|---|---|
| `reports/GPU_UTILIZATION_ANALYSIS.md` | SM% vs MEM BW% for all training jobs; MFU computation; ZeRO-3 root cause analysis; recommendation to move to ZeRO-2 |
| `reports/EVAL_ANALYSIS.md` | Full 5-expert vs 7-expert benchmark tables; per-modality breakdown; PathVQA regression investigation with confusion matrix and routing distribution |
| `reports/possible_optimizations.md` | Ranked list of optimisations by expected impact: ZeRO-2, sequence packing, gradient checkpointing, dataloader prefetching |
| `reports/smoketest_summary.md` | Config-level summary of all smoketest and sanitycheck runs with their results |
| `reports/architecture_diagram.md` | Textual architecture diagram of the full MultiMeditron model (input pipeline → CLIP encoders → gating → cross-attention → LLM) |
| `cookbook/DATA_AUDIT.md` | Dataset-level duplicate analysis across all 7 modalities |
| `reports/journal.md` | Running chronological log of experiments, observations, and decisions |

### Code documentation

- `docs/source/guides/moe.rst`: 424-line Sphinx guide to the MoE architecture — covers expert configs, gating network, cross-attention mechanism, PEP vs shared variants.
- `docs/source/guides/evaluation.rst`, `configuration.rst`, `training.rst`: Extended with new content covering the 7-expert setup and cluster-specific instructions.
- `cookbook/REGISTRY.md`: Model path registry listing all published checkpoints with their locations on capstor/iopsstor.
- `cookbook/gating/README.md`: Complete rewrite for the 7-class gating setup — dataset preparation, training commands, checkpoint paths, eval instructions.

### Utility scripts

| Script | Purpose |
|---|---|
| `scripts/bench_dataloader.py` | Benchmark DataLoader throughput (samples/sec, per-batch latency p50/p95/p99) isolated from GPU compute, to determine whether training is I/O-bound |
| `scripts/check_packing_attention.py` | Verify that the FA2 packing patch produces identical attention outputs to a reference unpacked run |
| `scripts/parse_smoketest_results.py` | Parse SLURM log files from smoketest/sanitycheck runs and print a summary table |
| `scripts/debug_mcq_context.py` | Debug MCQ (multiple-choice) context formatting — useful for diagnosing prompt construction issues |
| `scripts/generate_us_descriptions.py` | Generate textual descriptions for ultrasound images using a captioning model |
| `scripts/compare_modality_results.py` | Load two lmms-eval result directories and print a side-by-side per-modality comparison table |
| `scripts/gating_routing_analysis.py` | Measure routing accuracy for both gating checkpoints across all available modality datasets |
| `scripts/pathvqa_routing_analysis.py` | PathVQA-specific routing analysis comparing 5-expert vs 7-expert gating distributions |

---

## 6. Bug Fixes and Infrastructure Fixes

| Commit | Bug | Fix |
|---|---|---|
| `053c9e9` | `AutoConfig.from_pretrained(..., torch_dtype=dtype)` silently ignored, model loaded in fp32 | Changed to `dtype=dtype` |
| `fb39d14` | CUDA index-out-of-bounds when expert count changed from 5 to 7 | Gating/expert size mismatch handled with zero-filled weight tensor padding |
| `f51751f` | DeepSpeed init failure when `resume_from_checkpoint` points to non-existent path | Added existence check before passing path to Trainer |
| `64edcf3` | Config path passed to container resolved against the baked-in image copy rather than the user's source tree | Added `realpath` call in `sbatch_train.sh` before path is forwarded |
| `4ea5dea` | Lustre NFS serves stale `.pyc` bytecache to compute nodes — patch in `trainer.py` was silently not being applied on GH200 nodes | Added `PYTHONDONTWRITEBYTECODE=1` to `sbatch_train.sh` |
| `8d6c069` | `sbatch_eval_vllm.sh` committed by mistake | Deleted |

---

## 7. Summary Statistics

| Category | Quantity |
|---|---|
| Commits (user-authored, on this branch) | ~35 |
| Lines added | ~12,000+ |
| New Python files | ~15 |
| New training config YAML files | ~45 |
| New SLURM launcher scripts | ~10 |
| Reports and documentation files | ~12 |

---

## 8. Ultrasound Dataset Enrichment Pipeline

**Dates:** May 23 – June 4, 2026

### Context

The ultrasound training datasets (BUSI, ct2, DDTI, CovidUS) originally contained short VQA-style labels ("benign tumor", "malignant", yes/no answers). The goal was to replace these with rich 7-section clinical descriptions generated by a VLM, in two output formats matching the project schema:

- **Expert format**: plain `text` + image reference — for pretraining (image–text alignment).
- **LLM format**: `conversations` (user prompt + assistant response) + embedded image bytes — for instruction fine-tuning.

The 7-section structure was specified by the team lead:
1. Visible organs and structures
2. Features of each organ/structure
3. Additional findings
4. Gray scale and Doppler features
5. Dynamic features
6. Image quality and limitations
7. Impression/conclusion

### VLM Inference Pipeline (`scripts/generate_us_descriptions.py`)

Built a SLURM-based pipeline running **Qwen3-VL-8B-Instruct** on the GH200 GPUs. For each image the script:
1. Loads the source Arrow dataset (e.g. `BUSI_expert`) and extracts the image bytes and any pre-existing clinical context.
2. Constructs a structured prompt asking Qwen3 to produce a 7-section description.
3. Decodes the output and writes three JSONL files per dataset:
   - `context_examples/{DS}.jsonl` — input used for PDF review (context + generated text + source index)
   - `output/{DS}_expert.jsonl` — Expert format
   - `output/{DS}_llm.jsonl` — LLM format (image bytes base64-encoded in JSONL)

**Usage:**
```bash
# 50-sample preview:
sbatch sbatch_generate_us.sh BUSI

# Full generation:
sbatch --time 08:00:00 sbatch_generate_us.sh BUSI --all
```

**Datasets and sizes:**

| Dataset | Domain | Samples |
|---|---|---|
| BUSI | Breast ultrasound | 624 |
| ct2 | Kidney ultrasound/CT | 3,282 |
| DDTI | Thyroid ultrasound | 278 |
| CovidUS | Lung ultrasound | 19,192 |
| **Total** | | **~23,376** |

### Bug Fixes

#### Critical: Wrong model architecture class
**Symptom:** Generated descriptions were complete gibberish:
```
\ \ \.\n\n.\n\n.\n\nar ifif,.\n\n \s.\n\n increasingly,,, all...
```
**Root cause:** `Qwen3-VL-8B-Instruct` uses architecture type `qwen3_vl`, but the script was loading it as `Qwen2_5_VLForConditionalGeneration` (type `qwen2_5_vl`). Dozens of attention bias weights and visual MLP weights were randomly initialized — the mismatch warning was visible in stderr but the model loaded without error. The output was pure noise.

**Fix:** Replaced the explicit class with `AutoModelForVision2Seq`, which auto-resolves the correct class from the model config:
```python
# Before (wrong):
from transformers import Qwen2_5_VLForConditionalGeneration as VLModel

# After (correct):
from transformers import AutoModelForVision2Seq as VLModel
```

#### Qwen3 `<think>` token stripping
Qwen3 models emit internal chain-of-thought wrapped in `<think>...</think>` tags before producing the final answer. These tags must be stripped from the output before saving:
```python
import re
text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
```

### PDF Review Tooling

#### `scripts/pdf_gen.py`
Renders a visual review PDF from the `context_examples/*.jsonl` files. Each page shows the source image alongside the generated 7-section description. Supports splitting across multiple reviewers via `--num_people N` (each reviewer gets a separate PDF containing a disjoint subset of samples).

**ReportLab XML escape fix:** Qwen3 descriptions contain `**bold**`, `<`, `>`, `&` which crashed ReportLab's XML paragraph parser. Fixed by escaping all text before passing to `Paragraph()`:
```python
from xml.sax.saxutils import escape as xml_escape
desc_safe = xml_escape(desc).replace('\n', '<br/>')
```

**Usage:**
```bash
export STORAGE_ROOT=/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow
python scripts/pdf_gen.py \
    --context_dir /path/to/generated_data/context_examples \
    --output_prefix /path/to/review_person \
    --num_people 1
```

#### `scripts/review_training_datasets.py`
New script to generate a PDF review of all 11 original 7-expert training datasets (20 samples each). Used to document what the model was trained on as a baseline reference before enrichment.

**Datasets covered:** BUSI, COVID_US, ct2, iu_xray, PMC_VQA_FULL, llava_instruct, medtrinity_conversations ×2, image_mammoth, eye_dataset_converted, skin_dataset_converted.

**Output:** `training_data_review.pdf` (220 pages).

### Generated Outputs (on scratch)

```
/iopsstor/scratch/cscs/surech/multimeditron/generated_data/
  context_examples/    BUSI.jsonl  ct2.jsonl  DDTI.jsonl          (50 rows each)
  output/              BUSI_expert.jsonl    BUSI_llm.jsonl
                       ct2_expert.jsonl     ct2_llm.jsonl
                       DDTI_expert.jsonl    DDTI_llm.jsonl
  review_person_person1.pdf      ← 50-sample preview (BUSI + ct2 + DDTI), May 29
  training_data_review.pdf       ← 11 original training datasets, May 27
```

### Pending

- **PDF review approval** by team lead before committing to full ~23K generation run.
- **Full generation jobs** (after approval):
  ```bash
  sbatch --time 08:00:00 sbatch_generate_us.sh BUSI --all
  sbatch --time 08:00:00 sbatch_generate_us.sh ct2 --all
  sbatch --time 08:00:00 sbatch_generate_us.sh DDTI --all
  sbatch --time 08:00:00 sbatch_generate_us.sh CovidUS --all
  ```
- **JSONL → Arrow conversion** — `_llm.jsonl` files contain base64-encoded bytes; a conversion script (`scripts/convert_jsonl_to_arrow.py`) is needed to write them back as proper Arrow datasets to replace or augment the existing capstor datasets.
- **Metadata audit** — BUSI, ct2, DDTI source datasets have not been checked for accompanying clinical metadata (labels, notes, masks) that should be incorporated into the generated context.
- **DSPy / MMirage** — originally requested by team lead as tools for structured prompt management and image preprocessing; not yet integrated.

---

## 9. Summary Statistics (updated June 2026)

| Category | Quantity |
|---|---|
| Commits (user-authored, on this branch) | ~40 |
| Lines added | ~13,000+ |
| New Python files | ~18 |
| New training config YAML files | ~45 |
| New SLURM launcher scripts | ~12 |
| Reports and documentation files | ~13 |
| Ultrasound samples enriched (50-sample preview) | 150 |
| Ultrasound samples to enrich (full run) | ~23,376 |
| Benchmarks evaluated | GMAI (4,550), SLAKE (642), PathVQA (6,719) |
| Full training runs completed (Stage 2, 128 nodes) | 3 |
| Gating network training runs | 2 (5-class retrain, 7-class new) |

---

## 8. Outstanding Items

The following items were initiated but are not yet complete:

1. **Training collapse diagnosis**: Job 2340502 (packed, ZeRO-2) collapses to `loss=0.0` from step 3. Root cause not yet identified. Suspected: label masking bug in packing collator.
2. **Measured MFU with packing**: Theoretical analysis suggests ~2–5× improvement over ZeRO-3 baseline. No empirical measurement yet — blocked by the collapse above.
3. **DataLoader profiling**: `scripts/bench_dataloader.py` was written but never run. `num_workers` and `prefetch_factor` values in configs (16 and 4 respectively) are defaults, not empirically tuned.
4. **Switch production stage 1 and 2 configs to ZeRO-2**: `stage1_alignment.yaml` and `stage2_end2end.yaml` still reference `config/deepspeed_fast.json` (ZeRO-3). A ZeRO-2 variant `stage2_end2end_zero2.yaml` was created; the canonical configs were not updated.
5. **Histopathology/PathVQA routing fix**: The 7-expert model's "No" bias on PathVQA binary questions is a known regression with a clear root cause (Skin/MRI gating for histopathology). Proposed fixes are documented in `reports/EVAL_ANALYSIS.md` Section 5.
