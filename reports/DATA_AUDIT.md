# MultiMeditron Training Data Audit

> **Audience:** MultiMeditron contributors and reviewers evaluating training data quality.
> **Date:** 2 April 2026
> **Branch:** `add-ophthalmology-and-dermatology-experts`
> **Scope:** All datasets used in Stage 1 (alignment) and Stage 2 (end-to-end) training for the 7-expert ATTN-PEP model.

---

## Table of Contents

- [1 — Executive Summary](#1--executive-summary)
- [2 — Dataset Inventory](#2--dataset-inventory)
- [3 — Internal Duplication per Dataset](#3--internal-duplication-per-dataset)
- [4 — Cross-Dataset Overlaps](#4--cross-dataset-overlaps)
- [5 — Cross-Stage Overlaps](#5--cross-stage-overlaps)
- [6 — Training Pipeline Properties](#6--training-pipeline-properties)
- [7 — Benchmark Contamination Check](#7--benchmark-contamination-check)
- [8 — Methodology](#8--methodology)
- [9 — Recommendations](#9--recommendations)

---

## 1 — Executive Summary

We audited all 16 training datasets for internal duplicates, cross-dataset overlaps, cross-stage leakage, and benchmark contamination. Key findings:

- **ct2 (Kidney Ultrasound)** has a **44.9% duplication rate** — nearly half the rows are redundant. This is the only dataset with a severe problem.
- **image_mammoth** has a **3.5% duplication rate** (sampled) with extreme outliers: one Raven's Progressive Matrices conversation appears up to **57 times**.
- **MedTrinity 1 & 2** each have ~0.7–0.9% internal duplication, plus 0.47% cross-split overlap. Root cause: GPT-4V generating identical captions for similar medical images.
- **All other datasets** are clean (<0.1% duplication).
- **No cross-dataset overlaps** exist outside the MedTrinity family (except 6 trivial llava_instruct ∩ image_mammoth matches).
- **No validation loop** exists in the training pipeline — 100% of data is used for training.
- **No benchmark contamination** detected against GMAI, SLAKE, or PathVQA.

---

## 2 — Dataset Inventory

### Stage 1 — Alignment

| Dataset | Arrow Name | Domain | Samples | Source |
|---|---|---|---|---|
| LLaVA Pretrain | `llava_pretrain_cleaned` | Generalist | 558,000 | [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) |
| Pixmo Anything | `pixmo_anything` | Generalist | 101,000 | [Pixmo](https://huggingface.co/datasets/allenai/pixmo-anything) |
| Pixmo Cap | `pixmo_cap` | Generalist | 717,000 | [Pixmo](https://huggingface.co/datasets/allenai/pixmo-cap) |
| MedTrinity Alignment | `medtrinity_conversations_1_formatted_alignment` | Medical | 100,000 | MedTrinity-25M |
| Eye Dataset | `eye_dataset_converted` | Ophthalmology | 32,535 | Custom |
| Skin Dataset | `skin_dataset_converted` | Dermatology | 63,417 | Custom |

**Stage 1 total: ~1,572,000 rows**

### Stage 2 — End-to-End

| Dataset | Arrow Name | Domain | Samples | Source |
|---|---|---|---|---|
| MedTrinity 1 | `medtrinity_conversations_1_formatted` | Medical | 5,032,522 | MedTrinity-25M |
| MedTrinity 2 | `medtrinity_conversations_2_formatted` | Medical | 5,032,523 | MedTrinity-25M |
| LLaVA Instruct | `llava_instruct` | Generalist | 624,000 | [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |
| Image MAmmoTH | `image_mammoth` | Generalist | 1,639,212 | [MAmmoTH](https://github.com/TIGER-AI-Lab/MAmmoTH) |
| BUSI | `BUSI` | Ultrasound | 773 | [Kaggle](https://www.kaggle.com/datasets/subhajournal/busi-breast-ultrasound-images-dataset) |
| COVID-US | `COVID_US` | Ultrasound | 30,101 | [GitHub](https://github.com/nrc-cnrc/COVID-US) |
| ct2 (Kidney) | `ct2` | Ultrasound | 6,461 | [Kaggle](https://www.kaggle.com/datasets/siatsyx/ct2usforkidneyseg) |
| IU X-Ray | `iu_xray` | Radiology | 2,964 | [Kaggle](https://www.kaggle.com/datasets/areebbinnadeem/iu-xray) |
| PMC-VQA Full | `PMC_VQA_FULL` | Medical VQA | 176,948 | [PMC-VQA](https://huggingface.co/datasets/RadGenome/PMC-VQA) |
| Eye Dataset | `eye_dataset_converted` | Ophthalmology | 32,535 | Custom |
| Skin Dataset | `skin_dataset_converted` | Dermatology | 63,417 | Custom |

**Stage 2 total: ~12,641,456 rows**

> **Note:** `eye_dataset_converted` and `skin_dataset_converted` appear in **both** stages. This is intentional — these expert-specific datasets are used for alignment and then reinforced during end-to-end training.

---

## 3 — Internal Duplication per Dataset

### 📊 Summary Table

| Dataset | Rows | Scan | Unique | Wasted Rows | Dup Rate | Max Freq | Severity |
|---|---|---|---|---|---|---|---|
| BUSI | 773 | exhaustive | 773 | 0 | **0.00%** | 1x | ✅ Clean |
| COVID_US | 30,101 | exhaustive | 30,101 | 0 | **0.00%** | 1x | ✅ Clean |
| ct2 | 6,461 | exhaustive | 3,560 | 2,901 | **44.90%** | 11x | 🔴 Severe |
| iu_xray | 2,964 | exhaustive | 2,945 | 19 | **0.64%** | 2x | ✅ Clean |
| DDTI | 347 | exhaustive | 347 | 0 | **0.00%** | 1x | ✅ Clean |
| PMC_VQA_FULL | 176,948 | exhaustive | 176,948 | 0 | **0.00%** | 1x | ✅ Clean |
| eye_dataset_converted | 32,535 | exhaustive | 32,535 | 0 | **0.00%** | 1x | ✅ Clean |
| skin_dataset_converted | 63,417 | exhaustive | 63,396 | 21 | **0.03%** | 9x | ✅ Clean |
| MedTrinity Alignment | 100,000 | exhaustive | 99,763 | 237 | **0.24%** | 5x | ✅ Clean |
| MedTrinity 1 | 260,756† | sampled 30/579 | 258,880 | 1,876 | **0.72%** | 11x | 🟡 Low |
| MedTrinity 2 | 293,156† | sampled 30/515 | 290,424 | 2,732 | **0.93%** | 16x | 🟡 Low |
| llava_pretrain_cleaned | 94,069† | sampled 30/178 | 94,009 | 60 | **0.06%** | 6x | ✅ Clean |
| pixmo_anything | 12,576† | sampled 30/242 | 12,572 | 4 | **0.03%** | 2x | ✅ Clean |
| pixmo_cap | 28,944† | sampled 30/799 | 28,944 | 0 | **0.00%** | 1x | ✅ Clean |
| llava_instruct | 33,624† | sampled 30/557 | 33,624 | 0 | **0.00%** | 1x | ✅ Clean |
| image_mammoth | 10,465† | sampled 30/4700 | 10,096 | 369 | **3.53%** | 57x | 🟠 Moderate |

> † Rows scanned (not total dataset size). See [Methodology](#8--methodology) for sampling details.

### 🔴 ct2 — Detailed Breakdown

The ct2 (Kidney CT-to-Ultrasound) dataset has extreme duplication. Out of 6,461 rows, only 3,560 unique conversations exist:

| Appears N times | Unique conversations | Total rows |
|---|---|---|
| 1x | 2,475 | 2,475 |
| 2x | 357 | 714 |
| 3x | 250 | 750 |
| 4x | 187 | 748 |
| 5x | 120 | 600 |
| 6x | 86 | 516 |
| 7x | 46 | 322 |
| 8x | 21 | 168 |
| 9x | 13 | 117 |
| 10x | 4 | 40 |
| 11x | 1 | 11 |

**1,085 unique conversations** appear 2 or more times. The heavy tail (5x–11x) accounts for 1,774 rows (27.5% of the dataset). This likely stems from identical question templates applied to very similar kidney images.

### 🟠 image_mammoth — Detailed Breakdown

The worst offenders are **Raven's Progressive Matrices** questions — identical prompts with different answer letters:

| Appears N times | Unique conversations | Total rows |
|---|---|---|
| 1x | 10,002 | 10,002 |
| 2x | 54 | 108 |
| 3x | 22 | 66 |
| 4x | 6 | 24 |
| 5x | 3 | 15 |
| 6x | 2 | 12 |
| 7x | 1 | 7 |
| 10x | 1 | 10 |
| 11x | 1 | 11 |
| 48x | 1 | 48 |
| 52x | 1 | 52 |
| 53x | 1 | 53 |
| 57x | 1 | 57 |

Top duplicated conversations (from a 30-shard sample):
- **[57x]** "Here is a Raven's Progressive Matrice..." → Answer: D
- **[53x]** Same prompt → Answer: A
- **[52x]** Same prompt → Answer: C
- **[48x]** Same prompt → Answer: B
- **[11x]** "Describe and compare the quality among the four images" (image quality comparison)

These are likely template-based questions where the same text is paired with different images but the answer is identical.

### 🟡 MedTrinity — Detailed Breakdown

#### MedTrinity Alignment (100K — exhaustive)

| Appears N times | Unique conversations | Total rows |
|---|---|---|
| 1x | 99,561 | 99,561 |
| 2x | 172 | 344 |
| 3x | 26 | 78 |
| 4x | 3 | 12 |
| 5x | 1 | 5 |

202 unique conversations are duplicated. Wasted: 237 rows.

#### MedTrinity 1 (sampled 30/579 shards)

| Appears N times | Unique conversations | Total rows |
|---|---|---|
| 1x | 257,485 | 257,485 |
| 2x | 1,102 | 2,204 |
| 3x | 185 | 555 |
| 4x | 56 | 224 |
| 5x | 37 | 185 |
| 6x | 9 | 54 |
| 7x | 3 | 21 |
| 8x | 1 | 8 |
| 9x | 1 | 9 |
| 11x | 1 | 11 |

1,395 unique conversations duplicated across sampled shards.

#### MedTrinity 2 (sampled 30/515 shards)

| Appears N times | Unique conversations | Total rows |
|---|---|---|
| 1x | 288,645 | 288,645 |
| 2x | 1,319 | 2,638 |
| 3x | 266 | 798 |
| 4x | 85 | 340 |
| 5x | 47 | 235 |
| 6x | 23 | 138 |
| 7x | 11 | 77 |
| 8x | 5 | 40 |
| 9x | 8 | 72 |
| 10x | 5 | 50 |
| 11x | 5 | 55 |
| 12x | 2 | 24 |
| 14x | 2 | 28 |
| 16x | 1 | 16 |

1,779 unique conversations duplicated. MT2 has a heavier tail than MT1.

**Root cause**: MedTrinity-25M was auto-generated from PubMed images using GPT-4V. Multiple prompt templates were applied per image. Similar medical images from the same source (e.g., adjacent CT slices) produce identical captions, creating text-level duplicates even when the images differ slightly.

Top duplicated MedTrinity conversations:
- **[16x]** "The image is a transverse CT scan of the abdomen, displaying key abdominal organs including the liver, spleen, kidneys..."
- **[14x]** Same abdomen CT description, different prompt template
- **[11x]** "The image is a CT scan of the chest, displaying the lungs, heart, and other thoracic structures."

---

## 4 — Cross-Dataset Overlaps

### Within Stage 2

| Dataset A | Dataset B | Shared Conversations | % of A | % of B |
|---|---|---|---|---|
| MedTrinity 1 (sample) | MedTrinity 2 (sample) | 1,208 | 0.47% | 0.42% |
| llava_instruct (sample) | image_mammoth (sample) | 6 | 0.02% | 0.06% |

> **All other Stage 2 pairwise comparisons: 0 overlap.**

The MedTrinity cross-split overlap (0.47%) suggests the original 10M dataset was split into two shards (`conversations_1` / `conversations_2`) without perfect deduplication. Extrapolated to the full dataset, ~47K conversations may appear in both splits.

The 6 shared conversations between llava_instruct and image_mammoth are negligible.

### Within Stage 1

**No overlaps detected** between any Stage 1 dataset pairs.

---

## 5 — Cross-Stage Overlaps

| Stage 1 Dataset | Stage 2 Dataset | Shared | % of Stage 1 | % of Stage 2 | Type |
|---|---|---|---|---|---|
| MedTrinity Alignment | MedTrinity 1 (sample) | 5,772 | 5.79% | 2.23% | **Expected** — alignment is a subset of MT1 |
| MedTrinity Alignment | MedTrinity 2 (sample) | 671 | 0.67% | 0.23% | **Unexpected** — some alignment data in MT2 |
| eye_dataset_converted | eye_dataset_converted | 32,535 | 100% | 100% | **Intentional** — same dataset in both stages |
| skin_dataset_converted | skin_dataset_converted | 63,396 | 100% | 100% | **Intentional** — same dataset in both stages |

> **All other cross-stage pairs: 0 overlap.**

The alignment → MT1 overlap is expected: `medtrinity_conversations_1_formatted_alignment` is a 100K subset of `medtrinity_conversations_1_formatted` (note the shared `_1_` in the name). The small alignment → MT2 leak (671 samples, 0.67%) is an artifact of the imperfect MT1/MT2 split.

---

## 6 — Training Pipeline Properties

### No Validation Loop

The training pipeline (`src/multimeditron/cli/train.py`) instantiates `MultimodalTrainer` with only `train_dataset=dataset` — no `eval_dataset` is passed. The training configs contain no `eval_strategy`, `eval_steps`, or `do_eval` parameters. **100% of data is used for training**; only train loss is logged to WandB.

### No Deduplication

The dataset loading code in `src/multimeditron/cli/train.py` uses:

```python
for ds_config in config["datasets"]:
    dataset = load_from_disk(ds_config['packed_path'])
    packed_datasets.append(dataset)
ds = concatenate_datasets(packed_datasets).shuffle(seed=config.get("seed", 0))
```

Datasets are simply concatenated and shuffled. **No deduplication step exists.**

---

## 7 — Benchmark Contamination Check

We verified that evaluation benchmarks do not overlap with training data:

| Benchmark | Samples | Source | In Training? | Verdict |
|---|---|---|---|---|
| [GMAI-MMBench](https://huggingface.co/datasets/louis-mart/GMAI-MMBench-val) | 4,550 | Independently curated from 284 medical sources | No | ✅ No contamination |
| [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) | 1,061 (en) | Own curated CT/X-Ray/MRI images | No | ✅ No contamination |
| [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | 6,719 | Textbook pathology images | No | ✅ No contamination |

**MedTrinity indirect risk**: MedTrinity sources PubMed images, and GMAI draws from 284 datasets that may include PubMed papers. However, the task formats are entirely different (free-text captioning vs. MCQ) and no text-level overlap is plausible.

---

## 8 — Methodology

### 8.1 — Data Location

All training datasets are stored as HuggingFace Arrow datasets at:

```
/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/<dataset_name>/
```

Each dataset consists of multiple Arrow shard files (`data-NNNNN-of-MMMMM.arrow`) saved via HuggingFace `save_to_disk()`. The schema is uniform:

```
conversations: List[{role: str, content: str}]
modalities:    List[{type: str, value: {bytes: binary, path: null}}]
```

Dataset registration and sample counts are documented in `cookbook/REGISTRY.md`.

### 8.2 — Deduplication Detection Approach

Each conversation was hashed to detect text-level duplicates:

```python
import pyarrow.ipc as ipc
import hashlib
from collections import Counter

def hash_conversation(conv_list):
    """Concatenate all turns into a single string and MD5 hash it."""
    text = "||".join(f"{c['role']}:{c['content']}" for c in conv_list)
    return hashlib.md5(text.encode()).hexdigest()
```

**What this captures:**
- Exact text-level duplicates: conversations where user question + assistant response are byte-identical.
- Conversations with the same text but different images are counted as duplicates (since only text is hashed).

**What this does NOT capture:**
- Near-duplicates (paraphrased or slightly different responses to the same image).
- Image-level duplicates with different captions.

### 8.3 — Scanning Strategy

**Small/medium datasets** (≤30 shards or ≤200K total rows): exhaustive scan of all shards.

**Large datasets** (>30 shards): random sampling of 30 shards using `random.sample()` with fixed seeds for reproducibility.

| Dataset | Total Shards | Sampled Shards | Seed | Rows Scanned | Coverage |
|---|---|---|---|---|---|
| MedTrinity Alignment | 12 | 12 (all) | — | 100,000 | 100% |
| MedTrinity 1 | 579 | 30 | 42 | 260,756 | 5.2% |
| MedTrinity 2 | 515 | 30 | 43 | 293,156 | 5.8% |
| llava_pretrain_cleaned | 178 | 30 | 44 | 94,069 | 16.9% |
| pixmo_anything | 242 | 30 | 44 | 12,576 | 12.4% |
| pixmo_cap | 799 | 30 | 44 | 28,944 | 3.8% |
| llava_instruct | 557 | 30 | 44 | 33,624 | 5.4% |
| image_mammoth | 4,700 | 30 | 44 | 10,465 | 0.6% |
| BUSI | 1 | 1 (all) | — | 773 | 100% |
| COVID_US | 4 | 4 (all) | — | 30,101 | 100% |
| ct2 | 1 | 1 (all) | — | 6,461 | 100% |
| iu_xray | 2 | 2 (all) | — | 2,964 | 100% |
| DDTI | 1 | 1 (all) | — | 347 | 100% |
| PMC_VQA_FULL | 34 | 34 (all) | — | 176,948 | 100% |
| eye_dataset_converted | 30 | 30 (all) | — | 32,535 | 100% |
| skin_dataset_converted | 25 | 25 (all) | — | 63,417 | 100% |

> **Sampling limitation:** For sampled datasets, reported duplication rates are lower bounds. Duplicates that fall across un-sampled shards are missed. The true rate could be higher, especially for MedTrinity where duplicates are spread across many shards.

### 8.4 — Arrow File Reading

HuggingFace `save_to_disk()` writes Arrow files in **IPC stream** format (not IPC file format). Reading uses:

```python
import pyarrow.ipc as ipc

def read_shard(ds_name, shard_name):
    path = os.path.join(base, ds_name, shard_name)
    with open(path, 'rb') as f:
        reader = ipc.open_stream(f)
        return reader.read_all()
```

### 8.5 — Cross-Dataset Overlap Detection

For each dataset, we built a set of all unique conversation hashes (using the same MD5 approach). For small datasets, hashes were built from the full data; for large datasets, from the 30-shard sample. We then computed pairwise set intersections:

```python
overlap = hashes_A & hashes_B
pct_of_A = 100 * len(overlap) / len(hashes_A)
pct_of_B = 100 * len(overlap) / len(hashes_B)
```

This was done for all pairs within Stage 1 (15 pairs), within Stage 2 (55 pairs), and across stages (66 pairs, excluding same-dataset intentional overlaps).

### 8.6 — Analysis Script

The full analysis was run on the Clariden login node using Python 3.11 with `pyarrow 23.0.1`. Conversation hashes were computed on-the-fly (no intermediate files). Each small-dataset scan completed in seconds; MedTrinity sampling took ~2 minutes per split; cross-dataset overlap computation completed in ~10 minutes. All commands ran against capstor at `/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/`.

### 8.7 — Reproducibility

To reproduce any result, run on a Clariden login node:

```python
import pyarrow.ipc as ipc
import hashlib
import os
import random
from collections import Counter

BASE = "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow"

def hash_conversation(conv_list):
    text = "||".join(f"{c['role']}:{c['content']}" for c in conv_list)
    return hashlib.md5(text.encode()).hexdigest()

def analyze_dataset(ds_name, max_shards=None, seed=42):
    path = os.path.join(BASE, ds_name)
    files = sorted(f for f in os.listdir(path)
                   if f.startswith("data-") and f.endswith(".arrow"))
    if max_shards and len(files) > max_shards:
        random.seed(seed)
        files = [files[i] for i in sorted(random.sample(range(len(files)), max_shards))]

    counts = Counter()
    for f in files:
        with open(os.path.join(path, f), 'rb') as fh:
            table = ipc.open_stream(fh).read_all()
        for j in range(len(table)):
            counts[hash_conversation(table.column("conversations")[j].as_py())] += 1

    total = sum(counts.values())
    unique = len(counts)
    freq_dist = Counter(counts.values())
    print(f"{ds_name}: {total} rows, {unique} unique, "
          f"{total - unique} wasted ({100*(total-unique)/total:.2f}%)")
    for n in sorted(freq_dist):
        print(f"  {n}x: {freq_dist[n]} conversations")

# Example: reproduce ct2 analysis
analyze_dataset("ct2")

# Example: reproduce MedTrinity 1 sampling
analyze_dataset("medtrinity_conversations_1_formatted", max_shards=30, seed=42)
```

---

## 9 — Recommendations

### 🔴 High Priority — ct2 Deduplication

The ct2 dataset should be deduplicated before the next training run. With 44.9% redundancy, the model sees certain kidney ultrasound conversations up to 11 times per epoch. This could bias the model toward specific kidney descriptions.

**Suggested fix:** Deduplicate by conversation hash before saving to Arrow:

```python
from datasets import load_from_disk

ds = load_from_disk("/capstor/.../ct2")
seen = set()
unique_indices = []
for i, example in enumerate(ds):
    h = hash_conversation(example["conversations"])
    if h not in seen:
        seen.add(h)
        unique_indices.append(i)
ds_deduped = ds.select(unique_indices)
ds_deduped.save_to_disk("/capstor/.../ct2_deduped")
```

This would reduce ct2 from 6,461 → 3,560 rows.

### 🟠 Medium Priority — image_mammoth Investigation

The 57x-repeated Raven's Matrices conversations and similar template-based duplicates may reduce effective dataset diversity. Consider either:
- Deduplicating image_mammoth (would reduce by ~3.5%, extrapolated ~57K rows out of 1.64M)
- Capping maximum frequency per conversation to e.g. 5x if some repetition is desired

### 🟡 Low Priority — MedTrinity Cleanup

At ~0.7-0.9% duplication (extrapolated ~36K + ~47K internal, plus ~47K cross-split), the impact on a 10M-sample dataset is marginal. A full deduplication pass would require reading all 1,094 shards, which is feasible but time-intensive.

### ✅ No Action Needed

All other datasets (BUSI, COVID_US, DDTI, PMC_VQA_FULL, eye_dataset, skin_dataset, llava_pretrain, pixmo_anything, pixmo_cap, llava_instruct) are clean and need no intervention.
