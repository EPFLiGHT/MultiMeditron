# scripts/

Utility and data-preparation scripts. Training is done via the CLI, not from here — see the root README.

---

## Data preparation

### `build_multimediset_benchmark_splits.py`

Builds the benchmark split manifests used by the evaluation pipeline and expert training configs.
Reads the raw MultiMediset datasets and produces per-domain `train_model.jsonl` / `val.jsonl` / `test.jsonl`
files under `benchmark_splits/multimediset/<domain>/`.

```bash
python scripts/build_multimediset_benchmark_splits.py
```

### `build_histopathology_splits.py` / `build_histopathology_splits.sbatch`

Same as above, for the histopathology domain (heavier, run as a SLURM job via the `.sbatch` file).

### `build_splits.py` / `build_ct2d_glob_filtered_splits.py`

Domain-specific split builders (CT2D with source filtering).

### `prep_image_datasets.py`

Downloads datasets from the MultiMediset HuggingFace repository to a local folder. Edit the
`dataset_folders` and `path_datasets` variables at the top of the script before running.

```bash
python scripts/prep_image_datasets.py
```

### `download-datasets.sh`

Shell wrapper for dataset downloads.

---

## Auditing / analysis

| Script | Purpose |
|---|---|
| `audit_multimediset_labels.py` | Check label consistency across all MultiMediset domains |
| `audit_legacy_ct_labels.py` | Audit CT label mapping against the manifest |
| `audit_legacy_mri_labels.py` | Same for MRI |
| `audit_legacy_ophthalmology_labels.py` | Same for ophthalmology |
| `analyze_ct2d_glob_ct_count.py` | Count CT slices per source in CT2D-glob |
| `analyze_ct2d_glob_sources.py` | Breakdown of CT2D-glob by dataset source |
| `analyze_ct2d_mini_content.py` | Inspect CT2D-glob-mini composition |
| `analyze_mri_glob_content.py` | Inspect MRI-glob dataset content |
| `analyze_mri_manifest.py` | Inspect MRI benchmark split manifest |

---

## Dataset processing

`dataset_processing/` contains per-domain scripts that convert raw datasets into the standard
multimediset JSONL format (`{"text": "...", "modalities": [{"type": "image", "value": "..."}]}`):

```
dataset_processing/
├── mri_expert/
│   └── process_brain_tumor.py
├── ophthalmology_expert/
│   ├── process_eyepacs.py, process_fundus.py, process_messidor2.py
│   ├── process_rfmid2.py, process_slide.py, process_uwf_iqa.py
├── skin_expert/
│   ├── process_dermnet.py, process_fitzpatrick.py, process_isic.py
│   ├── process_scin.py, process_skin10.py
└── paraphrase_jsonl.py   # augment captions with GPT paraphrases
    train_val_split.py    # create train/val splits from a processed JSONL
```

---

## clip_playground/

Experimental scripts for testing CLIP model embeddings.

- `load_from_clip.py` — library helpers for loading embeddings from a CLIP checkpoint
- `neural_covid_pneu.py` — MLP probe on COVID-US to measure embedding quality

---

## Legacy scripts (kept for reference)

These scripts predate the current CLI-based pipeline and are no longer the recommended entry point.

| Script | Notes |
|---|---|
| `train_clip.py` | Original single-domain CLIP fine-tuning script. Use `python -m multimeditron.cli experts train_expert <config>` instead. |
| `biomed_train.py` | BiomedCLIP fine-tuning variant. Superseded by `train_multidomain_clip.py`. |
| `expert_model_train.py` | First-generation expert trainer. Superseded. |
| `image_router_train.py` | ResNet-based modality router trainer. Superseded. |
| `config_us.yaml` | Example config in the old nested-dict format. Current format uses a flat list under `dataset_configs`. |
