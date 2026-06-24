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

## Reproducing the evaluation

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r src/multimeditron/experts/evaluation_pipeline/requirements_experts.txt
```

### 2. Set environment variables

`setup.sh` at the repo root declares all required variables. Source it, then override the two dataset paths that are pre-filled with example values pointing to the author's cluster directories:

```bash
source setup.sh
export MRI_DATASET_ROOT="/your/path/to/brain_tumor_mri/images"
export XRAY_DATA_ROOT="/your/path/to/nih_chest_xrays"
export XRAY_KAGGLE_DATA_ROOT="/your/path/to/nih_chest_xrays"
```

### 3. Dataset setup

Benchmarks fall into three groups depending on where their data comes from.

**CT and Histopathology** — generated from CT2D-glob (MultiMediset, EPFL cluster):

```bash
export MULTIMEDISET_ROOT="/lightscratch/datasets/MultiMediset/general_purpose"
python scripts/build_histopathology_splits.py
# writes benchmark_splits/multimediset/ct/ and benchmark_splits/multimediset/histopathology/
```

Labels are parsed from the **original raw file paths** in `CT2D-glob-rawpath2905.jsonl` (CT: pathology
from `seg_train_{id}-{pathology}.nii-{slice}.jpg`; histopathology: cancer type from
`CancerType-CancerType-digit-TCGA-...jpg`), matched to HuggingFace entries via their caption, then
written as explicit `label_id` fields in the JSONL manifests.

**Skin, Ophthalmology, Ultrasound** — generated from MultiMediset sub-datasets (EPFL cluster):

```bash
python scripts/build_multimediset_benchmark_splits.py
# writes benchmark_splits/multimediset/skin/, eye/, ultrasound/
```

Both scripts require access to MultiMediset on the EPFL cluster (`MULTIMEDISET_ROOT`), which is available to all lab members. The generated JSONL files are intentionally not tracked in git (they are large and contain cluster-local paths).

**MRI** — Brain Tumor MRI dataset (Masoud Nickparvar), downloaded from Kaggle:

A Kaggle account and API key are required. Place your `kaggle.json` at `~/.kaggle/kaggle.json` (see [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token).

```bash
pip install kagglehub
python -c "import kagglehub; p = kagglehub.dataset_download('masoudnickparvar/brain-tumor-mri-dataset'); print(p)"
export MRI_DATASET_ROOT="<printed path>/images"
```

Expected layout under `MRI_DATASET_ROOT`:
```
train/  glioma/  meningioma/  no_tumor/  pituitary/
test/   glioma/  meningioma/  no_tumor/  pituitary/
```

**X-ray** — NIH ChestX-ray14, downloaded from Kaggle:

```bash
python -c "import kagglehub; p = kagglehub.dataset_download('nih-chest-xrays/data'); print(p)"
export XRAY_DATA_ROOT="<printed path>"
export XRAY_KAGGLE_DATA_ROOT="<printed path>"
```

Expected layout under `XRAY_DATA_ROOT`:
```
Data_Entry_2017.csv
images_*/images/*.png     # sharded structure created by kagglehub
```

### 4. Run a full evaluation

```python
import sys
sys.path.insert(0, "src/multimeditron/experts/evaluation_pipeline")

from build_benchmarks import build_benchmarks_from_names

# bench.evaluate() takes a path (it loads the model internally)
for bench in build_benchmarks_from_names(["ct", "mri", "skin", "ophthalmology", "ultrasound", "xray", "histopathology"]):
    print(bench.name, bench.evaluate("/path/to/your/checkpoint"))

# To use a pre-loaded model instead, call evaluate_model():
# from load_from_clip import load_model
# model = load_model("/path/to/your/checkpoint")  # or "openai/clip-vit-base-patch32"
# bench.evaluate_model(model=model, model_name="my_model")
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

When called with explicit names, `build_benchmarks_from_names` raises `ValueError` if a requested benchmark cannot be built (missing dataset files or env vars). To silently skip unavailable benchmarks instead, call `build_default_benchmarks()` directly (used internally when `benchmark_selection` is absent from the config).

### Environment variables

| Variable | Benchmark | Description |
|---|---|---|
| `CT_MAX_TRAIN_EXAMPLES` | ct | Limit training embeddings |
| `CT_MAX_TEST_EXAMPLES` | ct | Limit test embeddings |
| `MRI_DATASET_ROOT` | mri | Path to Brain Tumor MRI images/ folder (must contain train/ and test/) |
| `MRI_MAX_TRAIN_EXAMPLES` | mri | |
| `MRI_MAX_TEST_EXAMPLES` | mri | |
| `SKIN_INTEGRATED_MAX_TRAIN_EXAMPLES` | skin | |
| `SKIN_INTEGRATED_MAX_TEST_EXAMPLES` | skin | |
| `OPHTH_MAX_TRAIN_EXAMPLES` | ophthalmology | |
| `OPHTH_MAX_TEST_EXAMPLES` | ophthalmology | |
| `ULTRASOUND_MAX_TRAIN_EXAMPLES` | ultrasound | |
| `ULTRASOUND_MAX_TEST_EXAMPLES` | ultrasound | |
| `XRAY_DATA_ROOT` | xray | Override default NIH ChestX-ray14 root |
| `XRAY_KAGGLE_DATA_ROOT` | xray | Path to kagglehub-downloaded NIH ChestX-ray14 root (used when `XRAY_DATA_ROOT` is unset) |
| `XRAY_MAX_TRAIN_EXAMPLES` | xray | |
| `XRAY_MAX_TEST_EXAMPLES` | xray | |
| `HISTO_MAX_TRAIN_EXAMPLES` | histopathology | |
| `HISTO_MAX_TEST_EXAMPLES` | histopathology | |

### Benchmark labels

| Benchmark | Classes | Source |
|---|---|---|
| CT | 2 classes (covid-19 infection, right lung) | benchmark_splits/multimediset/ct |
| MRI | brain tumor grades (glioma, meningioma, no_tumor, pituitary) | Kaggle: masoudnickparvar/brain-tumor-mri-dataset |
| Skin | skin disease categories | benchmark_splits/multimediset/skin |
| Ophthalmology | diabetic_retinopathy (0), normal (1) | benchmark_splits/multimediset/eye |
| Ultrasound | 13 classes: COVID-US (0–3), BUSI (4–6), DDTI TIRADS (7–12) | benchmark_splits/multimediset/ultrasound |
| X-ray | 15 NIH radiological findings (multi-label) | NIH ChestX-ray14 |
| Histopathology | tissue subtypes | benchmark_splits/multimediset/histopathology |

---

## Adding a new benchmark

`benchmark_classification/benchmark_maker.py` generates a ready-to-use benchmark file from a YAML manifest.

### 1. Write a manifest

```yaml
# Required
name: pathology               # benchmark name — becomes the .name attribute and cache-key prefix
num_classes: 8                # number of output classes
manifest_subdir: pathology    # subfolder under benchmark_splits/multimediset/

# Optional
class_name: PathologyBenchmark   # default: TitleCase(name) + "Benchmark"; set for acronyms (e.g. CTBenchmark)
docstring: "Tissue pathology classification via multimediset manifest."

labels:                       # omit if labels are dynamic / loaded from the manifest
  - adenocarcinoma
  - squamous-cell-carcinoma

max_train_examples: 8_000     # omit to use the full manifest
max_test_examples: 2_000

allowed_subdatasets:          # filter records to specific sub-datasets (like SkinBenchmark)
  - tcga_path
  - cptac

stratify_by_label: false      # true → equal class representation when sub-sampling
is_available: false           # true → generate an is_available() classmethod

seed_train: 42
seed_test: 43
```

### 2. Generate the benchmark file

```bash
# writes benchmark_classification/pathology_benchmark.py
python benchmark_classification/benchmark_maker.py my_manifest.yaml

# preview without writing
python benchmark_classification/benchmark_maker.py my_manifest.yaml --print

# write to a custom path
python benchmark_classification/benchmark_maker.py my_manifest.yaml --output /path/to/output.py
```

### 3. Register the new benchmark (manual steps)

**`benchmark_classification/__init__.py`** — add the import:
```python
from .pathology_benchmark import PathologyBenchmark
# and add 'PathologyBenchmark' to __all__
```

**`build_benchmarks.py`** — add env-var dict and builder:
```python
PATHOLOGY_ENV_VARS = {
    "max_train_examples": "PATHOLOGY_MAX_TRAIN_EXAMPLES",
    "max_test_examples":  "PATHOLOGY_MAX_TEST_EXAMPLES",
}

def _maybe_build_pathology_benchmark():
    if not _manifest_pair_exists(PathologyBenchmark.default_manifest_root):
        return None
    return PathologyBenchmark(
        max_train_examples=_parse_optional_int(os.environ.get(PATHOLOGY_ENV_VARS["max_train_examples"])),
        max_test_examples=_parse_optional_int(os.environ.get(PATHOLOGY_ENV_VARS["max_test_examples"])),
    )
```

Then add `"pathology": _maybe_build_pathology_benchmark` inside `available_builders` in `build_benchmarks_from_names()`.

The script also prints these exact steps after generating the file.

---

## Integration with training

`train_multidomain_clip.py` calls `build_benchmarks_from_names(config["benchmark_selection"])` after
each Optuna trial and uses the mean score across benchmarks as the trial objective.

### Run training + evaluation

```bash
source setup.sh
mkdir -p logs
mm train-multidomain-optuna <config.yaml> 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
```

Benchmark scores for each trial are saved to `<output_dir>/benchmark_scores.txt`. The config field `benchmark_selection` controls which benchmarks are evaluated:

```yaml
benchmark_selection:
  - ct
  - histopathology
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
