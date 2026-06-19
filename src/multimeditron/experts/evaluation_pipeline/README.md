# Evaluation Pipeline

Toolkit for evaluating CLIP-style vision-language models on medical imaging tasks.
Two evaluation families coexist: **MLP-probe classification** (primary, integrated into training) and **retrieval benchmarks** (standalone).

---

## Directory structure

```
evaluation_pipeline/
├── Benchmark.py                    # Abstract base class
├── build_benchmarks.py             # Factory: instantiates benchmarks by name
├── load_from_clip.py               # Model loading utilities (CLIP, BiomedCLIP, checkpoints)
├── mlp_eval.py                     # Standalone MLP probe runner
│
├── benchmark_classification/       # MLP-probe classification benchmarks (main)
│   ├── base.py                     # Base class: embed → MLP → accuracy
│   ├── datasets.py                 # Dataset loading helpers
│   ├── multimediset_manifest.py    # Manifest-based JSONL loader
│   ├── ct_benchmark.py
│   ├── mri_benchmark.py
│   ├── skin_benchmark.py
│   ├── ophthalmology_benchmark.py
│   ├── ultrasound_benchmark.py
│   ├── xray_benchmark.py
│   └── histopathology_benchmark.py
│
├── retrieval/                      # Image-text retrieval benchmarks
│   ├── base_clip_evaluation.py     # 4-way forced-choice retrieval (random negatives)
│   ├── base_sim_benchmark.py       # Image-text alignment + tower diagnostics
│   ├── hard_negatives_evaluation.py
│   ├── display_most_sim.py         # Qualitative nearest-neighbor visualization
│   └── check_negative_overlap.py   # Lexical overlap analysis of negatives
│
├── scin/                           # SCIN skin benchmark
│   ├── scin_benchmark.py
│   └── hard_benchmark_scin_tone_stratified.py   # Fitzpatrick skin-tone fairness eval
│
└── disease_classification_pipeline/  # Legacy — older per-disease MLP approach
    ├── train_hp_opt.py, run_optuna_skin.py
    ├── skin_benchmark.py, Benchmark.py
    ├── evaluate_manually.py, plot_confusion_matrix.py, unpickle.py
```

---

## MLP-probe benchmarks (primary)

Each benchmark in `benchmark_classification/` follows the same protocol:
1. Freeze the CLIP vision encoder and embed all images
2. Train a small MLP on the embeddings
3. Report accuracy on the held-out test set

These are the benchmarks integrated into `train_multidomain_clip.py` via Optuna.

### Entry point: `build_benchmarks.py`

```python
from evaluation_pipeline.build_benchmarks import build_benchmarks_from_names

benchmarks = build_benchmarks_from_names(["ct", "mri", "skin", "ophthalmology", "ultrasound", "xray"])
for bench in benchmarks:
    score = bench.evaluate(model)
```

Supported names: `ct`, `mri`, `skin`, `ophthalmology`, `ultrasound`, `xray`, `histopathology`.

Each benchmark silently skips itself (`returns None`) if its dataset files are not found, so the factory never crashes in environments with partial data.

### Environment variables

| Variable | Benchmark | Description |
|---|---|---|
| `CT_MAX_TRAIN_EXAMPLES` | ct | Limit training embeddings |
| `CT_MAX_TEST_EXAMPLES` | ct | Limit test embeddings |
| `MRI_TRAIN_JSONL` | mri | Override default train manifest path |
| `MRI_TEST_JSONL` | mri | Override default test manifest path |
| `MRI_MAX_TRAIN_EXAMPLES` | mri | |
| `MRI_MAX_TEST_EXAMPLES` | mri | |
| `SKIN_INTEGRATED_MAX_TRAIN_EXAMPLES` | skin | |
| `SKIN_INTEGRATED_MAX_TEST_EXAMPLES` | skin | |
| `OPHTH_MAX_TRAIN_EXAMPLES` | ophthalmology | |
| `OPHTH_MAX_TEST_EXAMPLES` | ophthalmology | |
| `ULTRASOUND_MAX_TRAIN_EXAMPLES` | ultrasound | |
| `ULTRASOUND_MAX_TEST_EXAMPLES` | ultrasound | |
| `XRAY_DATA_ROOT` | xray | Override default NIH ChestX-ray14 root |
| `XRAY_MAX_TRAIN_EXAMPLES` | xray | |
| `XRAY_MAX_TEST_EXAMPLES` | xray | |
| `HISTO_MAX_TRAIN_EXAMPLES` | histopathology | |
| `HISTO_MAX_TEST_EXAMPLES` | histopathology | |

### Benchmark labels

| Benchmark | Classes | Source |
|---|---|---|
| CT | 3 CT findings | benchmark_splits/multimediset/ct |
| MRI | brain tumor grades | benchmark_splits/multimediset/mri |
| Skin | skin disease categories | benchmark_splits/multimediset/skin |
| Ophthalmology | diabetic_retinopathy (0), normal (1) | benchmark_splits/multimediset/eye |
| Ultrasound | 13 classes: COVID-US (0–3), BUSI (4–6), DDTI TIRADS (7–12) | benchmark_splits/multimediset/ultrasound |
| X-ray | 15 NIH radiological findings (multi-label) | NIH ChestX-ray14 |
| Histopathology | tissue subtypes | benchmark_splits/multimediset/histopathology |

---

## Integration with training

`train_multidomain_clip.py` calls `build_benchmarks_from_names(config["benchmark_selection"])` after
each Optuna trial and uses the mean score across benchmarks as the trial objective.

Config example:

```yaml
benchmark_selection:
  - ct
  - mri
  - skin
  - ophthalmology
  - ultrasound
  - xray
```

---

## Retrieval benchmarks (`retrieval/`)

Standalone scripts, not integrated into the training loop.

### 4-way forced-choice retrieval

```bash
python retrieval/base_clip_evaluation.py \
  --dataset /path/to/eval.jsonl \
  --num-samples 300 \
  --seed 14
```

Chance baseline: 25%.

### Hard negative evaluation

```bash
# Edit CLIP_CONFIGS in hard_negatives_evaluation.py, then:
python retrieval/hard_negatives_evaluation.py
```

For each query, 3 negatives are selected from visually similar images with different labels.
Output: `skin_clip_hard_benchmark.txt` with Recall@1 scores.

### Qualitative visualization

```python
from retrieval.display_most_sim import visualize_retrieval

visualize_retrieval(
    model_name_or_path="/path/to/model",
    eval_dataset="/path/to/eval.jsonl",
    k=3,
    out_path="retrieval_viz.png"
)
```

---

## SCIN skin-tone fairness evaluation (`scin/`)

```bash
# Edit CLIP_CONFIGS in hard_benchmark_scin_tone_stratified.py, then:
python scin/hard_benchmark_scin_tone_stratified.py
```

Reports Recall@1 broken down by Fitzpatrick skin type: light (FST 1–2), medium (FST 3), dark (FST 4–6).

---

## Model loading (`load_from_clip.py`)

```python
from load_from_clip import load_model

model = load_model("openai/clip-vit-base-patch32")   # vanilla CLIP
model = load_model("biomedclip")                      # BiomedCLIP
model = load_model("/path/to/checkpoint-1000")        # fine-tuned checkpoint
```

---

## Legacy: `disease_classification_pipeline/`

Older per-disease MLP pipeline, predates `benchmark_classification/`. Kept for reference.
Use `benchmark_classification/` for all new evaluations.
