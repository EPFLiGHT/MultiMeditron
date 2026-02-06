# Dataset Augmentation with OpenAI Batch API
# Author: Nazlican Turan

This directory contains a **reproducible, batch-based pipeline** for augmenting multimodal expert datasets (e.g. dermatology, ophthalmology) using the OpenAI Chat Completions API.

The pipeline is designed for:
- large-scale dataset augmentation,
- deterministic execution,
- cost estimation and tracking,
- separation of data processing and model interaction.

No datasets or API outputs are committed to the repository.

---

## Overview

The pipeline takes an existing multimodal dataset (JSONL manifest with image paths and text),
builds structured GPT prompts, submits them via the OpenAI **Batch API**, and collects the
generated outputs together with exact token usage and cost.

All steps are CLI-driven and configurable via environment variables.

---

## Directory Structure

```
.
├── config.py              # Central configuration (paths, model, pricing)
├── utils.py               # Dataset loading and image base64 encoding
├── make_batches.py        # Build Batch API request files (JSONL)
├── submit_batches.py      # Submit batch request files to OpenAI
├── collect_all.py         # Retrieve batch outputs, merge results, compute actual cost
├── estimate_price.py      # Cost projection from a small sample
├── scripts/
│   ├── run_all.sh        # Full dataset pipeline
│   └── run_estimate.sh    # Cost estimation pipeline
└── README.md
```

---

## Pipeline Steps

### 1. Dataset Preparation (external)

Raw datasets are first converted into a **standard multimediset format**:

```json
{
  "text": "...",
  "modalities": [
    { "type": "image", "value": "relative/path/to/image.jpg" }
  ]
}
```

This preprocessing step is dataset-specific and not included in this pipeline. The pipeline assumes the datasets already exist.

### 2. Build Batch Requests

```bash
# Defaults: TASK_TYPE=skin, NB_SAMPLES unset (all samples), BATCHES_DIR from config.py
python make_batches.py
```

* Loads dataset entries via `utils.py`
* Base64-encodes images
* Builds structured multimodal prompts
* Splits requests into size-limited JSONL files (`part_*.jsonl`)
* Output is written to `BATCHES_DIR`

### 3. (Optional) Estimate Cost

Use the estimation script to run a small sample and project total costs:
```bash
# Default runs 500 samples on skin task
./scripts/run_estimate.sh

# Customize task type and sample size
TASK_TYPE=ophthalmology NB_SAMPLES=1000 ./scripts/run_estimate.sh
```

This pipeline:
1. Builds batches for a small sample (`NB_SAMPLES`, default 500)
2. Submits only the first batch part (`ESTIMATE_ONLY=true`)
3. Collects outputs and computes estimate via `estimate_price.py`

No assumptions are made about token counts.

### 4. Submit Full Batches

Use the full pipeline script to process the entire dataset:
```bash
# Default runs full skin dataset
./scripts/run_full.sh

# Customize task type
TASK_TYPE=ophthalmology ./scripts/run_full.sh
```

This pipeline:
1. Builds batches for the full dataset (no `NB_SAMPLES` limit)
2. Submits all batch parts to OpenAI
3. Collects all outputs and computes actual cost

Alternatively, run steps manually:
```bash
# Set task type (default: skin)
export TASK_TYPE=skin

# 1) Build full batches
python make_batches.py

# 2) Submit all parts
python submit_batches.py

# 3) Collect everything (polls until ready)
python collect_all.py
```

---

## Environment Variables

* `TASK_TYPE` - Task type: `skin` or `ophthalmology` (default: `skin`)
* `NB_SAMPLES` - Number of samples to process (unset = all samples)
* `ESTIMATE_ONLY` - Submit only first batch part when `true` (default: `false`)
* `OPENAI_API_KEY` - Your OpenAI API key (required)
* `OPENAI_MODEL` - Model to use (set in `config.py`)
* `BATCHES_DIR` - Directory for batch files (set in `config.py`)
* `OUTPUT_DIR` - Directory for outputs (set in `config.py`)

## Configuration

All configuration is centralized in `config.py` and controlled via environment variables:

* `OPENAI_API_KEY`
* `OPENAI_MODEL`
* dataset paths
* batch/output directories
* pricing per million tokens

---

## Notes

* The pipeline currently uses the Chat Completions API via the Batch endpoint.
* Prompts are intentionally verbose and domain-specific to maximize annotation quality.
* Generated outputs are intended for dataset curation and annotation, not clinical use.
* `submit_batches.py` is safe to re-run and will not re-submit already completed batches.
---

## Disclaimer

This pipeline is intended for research and dataset development purposes only.