# Skin Disease Classification Benchmark for CLIP Models
# Author: Nazlican Turan

Evaluation framework for CLIP models on skin disease classification using frozen embeddings and Optuna hyperparameter optimization.

## Overview

This project evaluates CLIP representations by extracting frozen image embeddings and training a lightweight classifier on top. It optimizes CLIP training hyperparameters (learning rate, weight decay, frozen layers) to maximize downstream classification accuracy.

## Files

- `Benchmark.py` - Abstract base class for benchmarks
- `skin_benchmark.py` - Skin disease classification benchmark
- `train_hp_opt.py` - CLIP training with Optuna optimization
- `run_optuna_skin.py` - Entry point for optimization studies
- `evaluate_manually.py` - Standalone evaluation script
- `unpickle.py` - Inspect saved Optuna studies

## Installation

From the repository root:
```bash
pip install -e ".[experts]"
```


Requires Python 3.8+

## Data Format

JSONL files with:
```json
{
  "text": "Eczema Photos",
  "modalities": [{"value": "images/eczema-subacute-68.jpg"}]
}
```

## Usage

### Run Hyperparameter Optimization on a selected config

```bash
python run_optuna_skin.py <config.yaml> <study_id>
```

Outputs: `study_skin_<study_id>.pkl`

### Evaluate a Model

```python
from skin_benchmark import SkinDiseaseBenchmark

skin_bench = SkinDiseaseBenchmark(
    train_jsonl="/path/to/train.jsonl",
    test_jsonl="/path/to/test.jsonl",
    image_root="/path/to/images"
)

accuracy = skin_bench.evaluate("/path/to/model")
```

Or edit and run:
```bash
python evaluate_manually.py
```

### Inspect Results

```bash
python unpickle.py
```

## Hyperparameter Search Space

- `learning_rate`: 5e-6 to 5e-4
- `weight_decay`: 0.05 to 0.4
- `freezed_layers`: 0 to 8

## Evaluation Protocol

1. Extract frozen CLIP embeddings for all images
2. Train 3-layer MLP classifier with balanced class weights
3. Test multiple initializations (Xavier, Kaiming, Orthogonal), keep best
4. Return accuracy on test set

## Output Files

Each trial creates: `{output_dir}_lr{lr}_wd{wd}_nfrz{n_frz}/`
- CLIP model weights
- `skin_per_class_metrics.json` - Per-class accuracy and totals

## Advanced Features

**Confusion Matrix:**
```python
acc, per_class_acc, totals, cm, id2label = skin_bench.evaluate_with_confusion(model_path)
```

**Multi-Benchmark:**
```python
from train_hp_opt import train
study = train([bench1, bench2], config_path)  # Returns geometric mean
```

**Disable W&B:**
```python
os.environ["WANDB_DISABLED"] = "true"
```