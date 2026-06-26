# Expert Training

Per-domain CLIP-based vision-text dual encoder training with Optuna hyperparameter optimisation,
followed by three evaluation strategies to compare expert checkpoints.

---

## Prerequisites

**First time only** — create the virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
pip install -r src/multimeditron/experts/evaluation_pipeline/requirements_experts.txt
```

> `flash-attn` in that requirements file is not needed for expert training. If it fails to build
> (requires matching CUDA version and C++ compiler), remove it from the file before running the command.

**Every session:**

```bash
source setup.sh          # set env vars (HF_HOME, benchmark dataset paths, …)
source .venv/bin/activate
wandb login              # required — training calls wandb.init() at startup
```

`mm` is the project CLI, installed as part of the package. Run `mm --help` to list all commands.

---

## Key files

| File | Role |
|---|---|
| `all_medical_datasets_config.yaml` | Master spec: datamixes, base configs, HPO grid, benchmark selection |
| `config_maker.py` | Generates one YAML per (datamix × base_config × param_grid) combination |
| `configurations/` | Generated per-run configs (output of `config_maker.py`) |
| `train_multidomain_clip.py` | Main training loop — CLIP fine-tuning with Optuna HPO, calls benchmarks after each trial |
| `evaluate_expert_baselines.py` | Evaluates each domain expert checkpoint independently |
| `evaluate_expert_mixture.py` | Evaluates a frozen mixture of experts (routed by domain) |
| `evaluate_expert_soup.py` | Merges expert weights by averaging (model soup), then evaluates |
| `models/` | Trained checkpoints, named `{datamix}_{config}_lr{lr}_wd{wd}_nfrz{n}/` |

`train_clip.py` is the legacy single-domain trainer — use `train_multidomain_clip.py` for all new runs.

---

## Workflow

### 1. Build benchmark splits

The training configs reference JSONL manifests under `benchmark_splits/multimediset/`. Generate them first:

```bash
python scripts/build_multimediset_benchmark_splits.py
python scripts/build_histopathology_splits.py   # heavier — submit as a SLURM job if needed
```

### 2. Generate configs

```bash
source setup.sh && source .venv/bin/activate
mm config-maker-expert src/multimeditron/experts/all_medical_datasets_config.yaml
# writes configurations/{datamix}_{config_name}_config_{idx}.yaml
```

`all_medical_datasets_config.yaml` controls:
- **datamixes** — which JSONL manifests to include (from `benchmark_splits/multimediset/`)
- **base_configs** — fixed hyperparameter sets
- **param_ranges** — grid to sweep (e.g. `learning_rate`, `weight_decay`)
- **benchmark_selection** — which benchmarks are evaluated after each Optuna trial

### 3. Train

Replace the config filenames below with the ones generated at step 2.

Launch two configs in parallel (each gets its own `nohup` process, logs go to `<config>.log`):

```bash
mm batch-train-multidomain-optuna \
    src/multimeditron/experts/configurations/multimediset_manifest_general_train_config.yaml \
    src/multimeditron/experts/configurations/multimediset_manifest_no_histo_train_config.yaml
```

Run a single config sequentially with a timestamped log (useful to monitor progress interactively):

```bash
mm train-multidomain-optuna \
    src/multimeditron/experts/configurations/multimediset_manifest_general_train_config.yaml \
    2>&1 | tee src/multimeditron/experts/logs/train_$(date +%Y%m%d_%H%M%S).log
```

Benchmark scores per trial are saved to `<output_dir>/benchmark_scores.txt`.

> **Restarting a run:** Optuna persists its state in `<output_dir>/<config_name>_optuna.db`. If you
> re-run without deleting it, Optuna resumes from the previous study. To start fresh:
> ```bash
> rm src/multimeditron/experts/models/multimediset_manifest_general_train_config/multimediset_manifest_general_train_config_optuna.db
> rm src/multimeditron/experts/models/multimediset_manifest_no_histo_train_config/multimediset_manifest_no_histo_train_config_optuna.db
> ```

### 4. Evaluate

**Baselines** — each expert evaluated on its own domain benchmark:

```bash
python src/multimeditron/experts/evaluate_expert_baselines.py \
    --expert_root /lightscratch/users/nemo/models/ \
    --domains ct mri skin ophthalmology ultrasound xray \
    --output_csv src/multimeditron/experts/logs/expert_baseline_results.csv
```

**Mixture** — sweep all fusion strategies:

```bash
for fusion in concat mean; do
    python src/multimeditron/experts/evaluate_expert_mixture.py \
        --expert_root /lightscratch/users/nemo/models/ \
        --fusion $fusion \
        --domains ct mri skin ophthalmology ultrasound xray \
        2>&1 | tee src/multimeditron/experts/logs/expert_mixture_${fusion}_$(date +%Y%m%d_%H%M%S).log
done
```

Available fusion strategies: `concat`, `mean`.

**Soup** — weight-averaged merge then evaluated:

```bash
python src/multimeditron/experts/evaluate_expert_soup.py \
    --expert_root /lightscratch/users/nemo/models/ \
    --domains ct mri skin ophthalmology ultrasound xray \
    --output_csv src/multimeditron/experts/logs/expert_soup_results.csv
```

---

## Benchmark infrastructure

See [`evaluation_pipeline/README.md`](evaluation_pipeline/README.md) for:
- How each benchmark works (MLP-probe protocol)
- Dataset sources and required env vars
- How to add a new benchmark
