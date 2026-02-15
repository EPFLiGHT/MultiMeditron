# CLIP Model Evaluation Pipeline

Toolkit for evaluating CLIP-style vision-language models on medical imaging tasks using multiple evaluation protocols: basic image-text alignment, hard negative retrieval, skin tone stratified analysis, qualitative visualization and classification evaluation.

## Overview

This pipeline provides several evaluation methods:
- **4-way forced choice retrieval** (Recall@1, 25% chance baseline)
- **Hard negative evaluation** (visually similar but semantically different negatives)
- **Skin tone stratified analysis** (Fitzpatrick scale-based fairness evaluation)
- **Qualitative retrieval visualization** (nearest neighbor inspection)
- **Diagnostic probes** (text/image tower sanity checks)

## Files

- `base_clip_evaluation.py` - Basic 4-way retrieval with random negatives
- `base_sim_benchmark.py` - Image-text alignment evaluation with tower diagnostics
- `hard_negatives_evaluation.py` - Hard negative retrieval protocol
- `hard_benchmark_scin_tone_stratified.py` - Skin tone fairness evaluation
- `display_most_sim.py` - Qualitative nearest neighbor visualization
- `check_negative_overlap.py` - Lexical overlap analysis of negatives
- `load_from_clip.py` - Model loading utilities (supports vanilla CLIP, BiomedCLIP, custom models)
- `Benchmark.py` - Basic abstract class for CLIP model evaluation benchmark
- `mlp_eval.py` - Classification evaluation with a multi layer perceptron
- `ultrasound_new_benchmark.py` Benchmark assessing anatomical region classification in ultrasound imaging
- `xray_eval.py' Benchmark assessing diseases classification in X-ray imaging

## Installation

```bash
pip install -r requirements_experts.txt
```
Requires Python 3.8+

## Data Format

JSONL files with:
```json
{
  "text": "Moderate diabetic retinopathy without macular edema",
  "modalities": [{"type": "image", "value": "images/example.jpg"}]
}
```

## Quick Start

### Basic Evaluation

```bash
python base_clip_evaluation.py \
  --dataset /path/to/eval.jsonl \
  --num-samples 300 \
  --seed 14
```

### Hard Negative Evaluation

```python
# Edit CLIP_CONFIGS in hard_negatives_evaluation.py:
CLIP_CONFIGS = [
    ("my_model", "/path/to/model"),
]

python hard_negatives_evaluation.py
```

Output: `skin_clip_hard_benchmark.txt` with Recall@1 scores

### Skin Tone Stratified Evaluation

```python
# Edit CLIP_CONFIGS in hard_benchmark_scin_tone_stratified.py
python hard_benchmark_scin_tone_stratified.py
```

Output: Recall@1 broken down by Fitzpatrick skin type groups (light/medium/dark)

### Qualitative Visualization

```python
from display_most_sim import visualize_retrieval

visualize_retrieval(
    model_name_or_path="/path/to/model",
    eval_dataset="/path/to/eval.jsonl",
    k=3,  # top-3 neighbors
    preferred_query_labels={"diabetic retinopathy"},
    out_path="retrieval_viz.png"
)
```
### Anatomical benchmark

``` 
python ultrasound_new_benchmark.py <<Path_of_evaluated_model.py>>
```

### X-ray benchmark

``` 
python xray_eval.py <<Path_of_evaluated_model.py>>
```

## Evaluation Protocols

### 1. Random Negative (Baseline)

**Script:** `base_clip_evaluation.py` or `base_sim_benchmark.py`

For each image:
1. Select 1 correct caption + 3 random negatives
2. Compute cosine similarity between image and all 4 texts
3. Prediction is correct if ground-truth ranks highest

**Chance baseline:** 25%

### 2. Hard Negative (Challenging)

**Script:** `hard_negatives_evaluation.py`

For each image:
1. Use reference CLIP model to find visually similar images
2. Select 3 negatives with different disease labels from top-K nearest neighbors
3. Fixed protocol built once, reused for all models

**Key features:**
- Enforces different disease classes (no trivial duplicates)
- Visually confusable negatives
- Fair cross-model comparison

**Configuration:**
```python
REF_MODEL_NAME = "openai/clip-vit-base-patch32"  # Reference for building protocol
HARD_TOPK = 3  # Pool size for negative sampling
TIE_POLICY = "count_incorrect"  # How to handle ties
```

### 3. Skin Tone Stratified

**Script:** `hard_benchmark_scin_tone_stratified.py`

Same as hard negative protocol, but:
- Extracts Fitzpatrick skin type from captions or manifest
- Groups into light (FST 1-2), medium (FST 3), dark (FST 4-6)
- Reports separate Recall@1 for each group

**Metadata extraction:**
- FST from text: "Fitzpatrick skin type: FST4"
- Disease proxy: "Dermatologist differential (weighted): acne (0.85)"

### 4. Anatomical benchmark

**Script:** `ultrasound_new_benchmark.py`

This script implements an anatomical classification benchmark for ultrasound images using embeddings from a CLIP vision–text encoder. 
- It aggregates multiple public ultrasound datasets into a unified 4-class problem (breast, abdomen, thyroid, other)
- encodes all images into fixed embeddings, and stores them for reuse. 
- An MLP classifier is then trained on these embeddings to evaluate how well the encoder separates different anatomical regions.

### 5. X-ray benchmark

**Script:** `mlp_eval.py`

This script defines a chest X-ray benchmark to evaluate a CLIP vision–text encoder using multi-label disease classification. 
- It encodes X-ray images into embeddings, builds multi-hot labels for 15 radiological findings from a CSV file, and splits the data into train/test sets. 
- An MLP is then trained on these embeddings to measure how well the encoder captures radiological findings.

## Model Loading

The `load_from_clip.py` module supports:

### Vanilla CLIP
```python
from load_from_clip import load_model
model = load_model("openai/clip-vit-base-patch32")
```

### BiomedCLIP
```python
model = load_model("biomedclip")
```

### Fine-tuned Models
```python
# Automatically finds latest checkpoint
model = load_model("/path/to/training/output_dir")
```

### Custom Models
```python
# Specific checkpoint
model = load_model("/path/to/checkpoint-1000")
```

## Configuration

### Key Parameters

```python
# Hard negative evaluation
SEED = 14                      # Random seed
IMG_BS = 32                    # Image batch size
TXT_BS = 64                    # Text batch size
TXT_MAX_LEN = 128              # Max text length
HARD_TOPK = 3                  # Negative pool size
TIE_EPS = 1e-7                 # Similarity epsilon for tie detection
TIE_POLICY = "count_incorrect" # Tie handling: "count_incorrect", "skip", "argmax"
```

### Tie Policies

- `count_incorrect`: Count ties as wrong (conservative)
- `skip`: Skip tied samples (not counted in denominator)
- `argmax`: Use argmax (arbitrary but deterministic)

## Output Files

### Hard Negative Evaluation
```
# name    accuracy    ties    nonfinite_sims
model_1  0.8234      12      0
model_2  0.7891      8       0
```

### Skin Tone Stratified
```
model           light   medium  dark    unknown
skin_clip_v1    0.8234  0.7891  0.7123  0.6543
```

### Qualitative Visualization
PNG file showing query image + top-K nearest neighbors with similarity scores

### Anatomical benchmark
```
best results with lr: <<learning rate which gave the best results during cross validation in the training of the classifier>> and wd: <<weight decay which gave the best results during cross validation in the training of the classifier>>
test accuracy : <<result of the evaluation>>
```
### X-ray benchmark
```
best results with lr: <<learning rate which gave the best results during cross validation in the training of the classifier>> and wd: <<weight decay which gave the best results during cross validation in the training of the classifier>>
test accuracy : <<result of the evaluation>>
```
## Diagnostic Tools

### Text Tower Probe

Tests if text encoder produces diverse embeddings:
```python
probe_texts = [
    "a cat sitting on a couch",
    "retinal fundus image with drusen",
    "an x-ray showing lung consolidation",
    ...
]
# Checks pairwise cosine similarity matrix
# Reports unique embeddings count
```

### Image Tower Probe

Tests if image encoder produces diverse embeddings using synthetic colored images

### Lexical Overlap Analysis

```bash
python check_negative_overlap.py
```

Reports Jaccard similarity between positive and negative captions to assess difficulty:
- High overlap → easy negatives
- Low overlap → hard negatives

## Advanced Usage

### Custom Reference Model

Change hard negative protocol reference:
```python
REF_MODEL_NAME = "chuhac/BiomedCLIP-vit-bert-hf"
```

### Multiple Datasets

```python
EVAL_DATASETS = [
    "/path/to/dataset1.jsonl",
    "/path/to/dataset2.jsonl",
]
```

All datasets are combined before evaluation.

### Save Protocol for Reproducibility

```python
# In hard_negatives_evaluation.py, triples are automatically saved:
protocol_path = RESULTS_TXT.replace(".txt", "_protocol.json")
# Contains [(query_idx, neg1_idx, neg2_idx, neg3_idx), ...]
```

### Load Saved Protocol

```python
import json
with open("protocol.json", "r") as f:
    triples = json.load(f)
# Use triples directly in evaluate_model()
```


## Best Practices

1. **Use fixed protocol**: Build hard negative protocol once with reference model, reuse for all models
2. **Check diagnostics**: Always inspect text/image tower probes for representation collapse
3. **Monitor ties**: High tie counts indicate degenerate embeddings
4. **Verify lexical overlap**: Ensure negatives aren't trivially easy or impossibly hard
5. **Stratify analysis**: Report per-group metrics for fairness evaluation

