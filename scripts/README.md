# `scripts/` — utilities and training helpers

This directory holds standalone utilities for data prep, gating-network training,
routing analysis, and evaluation post-processing.

> **Production training/eval** is driven by the `multimeditron` CLI and the SLURM
> launchers at the repo root (`sbatch_train.sh`, `sbatch_eval.sh`) — see
> `cookbook/README.md`. The scripts here are supporting tools.

---

## Gating network training

Train the ResNet50 gating network that routes images to the correct expert.

```bash
# Multi-GPU via torchrun (config-driven):
torchrun --nproc_per_node=4 scripts/train_gating.py --config config/gating_7class.yaml

# Or submit on CSCS (debug partition, ~30 min):
sbatch sbatch_train_gating.sh
```

`train_gating.py` loads per-class Arrow datasets (via the `dataset_class_map` in
the YAML config), trains a classification head on a frozen ImageNet ResNet50, and
saves the result directly as a HuggingFace `GatingNetwork` (`config.json` +
`model.safetensors` with `class_names`) — ready to use as `gating_path` in the MoE
configs. Any config value can be overridden from the CLI (e.g. `--lr 3e-4
--num_epochs 30`). A held-out `test_split` (default 0.1) is carved off before
training for an unbiased accuracy estimate. See `cookbook/gating/README.md` for the
full guide.

> Replaces the former `image_router_train.py` (ImageFolder-based, manual
> `.pth`→HF conversion), which has been removed.

`test_gating.py` loads a trained checkpoint and prints its routing distribution on
eye/skin images — a quick sanity check.

---

## Expert (CLIP) training

The canonical CLIP-expert trainer lives in the package and is invoked via the CLI:

```bash
multimeditron train-expert scripts/config_us.yaml
```

(`scripts/train_clip.py`, a duplicate of `src/multimeditron/experts/train_clip.py`,
has been removed.) The config selects the vision/text models and the dataset
mixture (`dataset_configs` with per-dataset `weight`s). Keep
`vision_model_name: openai/clip-vit-base-patch32`; the trainer is specialised for it.

Two older domain-specific trainers remain for reference:

- `expert_model_train.py` — generic CLIP fine-tuner (HuggingFace dataset URL input).
- `biomed_train.py` — BiomedCLIP fine-tuner that reads a `.jsonl` of `{text, modalities}`
  examples (BiomedCLIP has different I/O than CLIP, so it needs its own loader):
  `python3 biomed_train.py --data_url chexpert/chexpert.jsonl --output_dir chexpert_test --num_epochs 20`

---

## Gating / routing analysis

These three scripts share `gating_utils.py` (Arrow image loading, the ResNet
preprocessing transform, the expert-label map, and the gating inference loop):

| Script | Purpose |
|---|---|
| `gating_routing_analysis.py` | Compare 5-expert vs 7-expert routing on 5 modality-pure held-out datasets (top-1 % + avg softmax weight per expert). Use to verify routing after a retrain or to diagnose CT/US confusion. |
| `pathvqa_routing_analysis.py` | PathVQA-specific: which expert do histopathology images route to under each gating network? |
| `test_gating.py` | Quick routing sanity check on eye/skin images. |

```bash
# No GPU required (ResNet50 runs on CPU); or submit via:
sbatch sbatch_gating_analysis.sh
python3 scripts/gating_routing_analysis.py
```

---

## Evaluation post-processing

`compare_modality_results.py` — side-by-side per-modality GMAI accuracy table for two
lmms-eval result directories (e.g. 5-expert ckpt-3063 vs 7-expert ckpt-800), including
ophthalmology/dermatology subtasks and sample counts.

```bash
python3 scripts/compare_modality_results.py        # auto-discovers result dirs
# Flags: --results-root, --model-a, --model-b
```

---

## Data preparation

| Script | Purpose |
|---|---|
| `prep_image_datasets.py` | Download datasets from the MultiMediset HF repo (jsonl + parquet/zip archives) into a local folder layout. Edit the `dataset_folders` / `path_datasets` dicts at the top before running. |
| `convert_image_datasets.py` | Reformat raw datasets (e.g. eye/skin) into the MultiMeditron training schema (`conversations` + `modalities` with image bytes). |

Pipeline: `prep_image_datasets.py` (download) → `convert_image_datasets.py` (reformat).

---

## Profiling / debugging

| Script | Purpose |
|---|---|
| `bench_dataloader.py` | Benchmark DataLoader throughput (samples/s, p50/p95/p99 latency) to check for I/O bottlenecks. |
| `check_packing_attention.py` | Verify the FA2 sequence-packing patch produces the same attention output as an unpacked reference. |
| `parse_smoketest_results.py` | Parse SLURM logs from smoketest/sanitycheck runs into a summary table. |
| `debug_mcq_context.py` | Inspect MCQ prompt construction and compare model generations (modes: text / generate / compare). |

---

## Ultrasound dataset enrichment

| Script | Purpose |
|---|---|
| `generate_us_descriptions.py` | Run Qwen3-VL to generate 7-section clinical descriptions for ultrasound datasets (BUSI, ct2, DDTI, CovidUS). |
| `pdf_gen.py` | Render a review PDF (source image + generated description) for human review, splittable across reviewers. |
| `review_training_datasets.py` | Render a review PDF of the original 7-expert training datasets (baseline reference). |
