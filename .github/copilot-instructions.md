# MultiMeditron — Copilot Workspace Instructions

## Project Context

MultiMeditron is a multimodal Mixture-of-Experts medical VLM built on LLaMA-3.1-8B with 7 CLIP expert encoders (CT, MRI, Ultrasound, X-ray, Pathology, Ophthalmology, Dermatology) and an attention-based PEP gating mechanism. Training runs on CSCS Alps Clariden (GH200 ARM64 nodes).

---

## Cluster / SLURM Conventions

- **Account**: always `a127`
- **User**: `surech`
- **Node type**: GH200, 4× 96 GB GPU per node
- **Training container** (EDF): `~/.edf/multimeditron.toml`
- **Eval container** (EDF): `~/.edf/vllm.toml` — but see Eval rules below
- **Debug partition**: max 2 nodes, 30 min wall time
- **Normal partition**: up to 128 nodes
- `envsubst` does **not** support double-dollar escaping on this cluster — do not pipe scripts through it

---

## Checkpoint & Output Paths

| Item | Path |
|------|------|
| Stage 1 checkpoints | `/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/freeze/attn_pep/MultiMeditron-8B-attn-pep-alignment/` |
| Stage 2 checkpoints | `/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end-7exp/` |
| 5-expert ATTN-PEP baseline checkpoints | `/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end/` (checkpoint-700 → checkpoint-900) |
| Base LLM (Meditron3-8B) | `/capstor/store/cscs/swissai/a127/meditron/hf_cache/hub/models--OpenMeditron--Meditron3-8B/snapshots/15914bcb040cd1a4f263afcd85b84f09ad2efd95` |
| Training logs | `/users/surech/meditron/reports/` (pattern: `R-<jobname>.<jobid>.out/err`) |
| Eval results | `/users/surech/meditron/reports/lmms_eval_results/` |
| HF cache (scratch) | `/iopsstor/scratch/cscs/surech/hf` |
| HF cache (capstor, shared) | `/capstor/store/cscs/swissai/a127/meditron/hf_cache` |

---

## Training Script Usage

- **Generic launcher**: `sbatch_train.sh <config.yaml>` — submits with `multimeditron.toml` container
- **Stage 1 config**: `cookbook/sft/moe/attn/pep/stage1_alignment.yaml`
- **Stage 2 config**: `cookbook/sft/moe/attn/pep/stage2_end2end.yaml`
- Always use **exactly 128 nodes** for Stage 2 (ZeRO-3 shard count must match checkpoint)
- DeepSpeed config: `config/deepspeed.json` (not `config/deepspeed_fast.json` for production)
- To resume: set `resume_from_checkpoint: <path>` in the YAML config

---

## Eval Pipeline Rules

- **Always use `sbatch_eval.sh`** (accelerate-based, `multimeditron.toml` container) — this is the proven working eval script
- **Never use `sbatch_eval_vllm.sh`** — vLLM cannot load our custom `multimodal` model type and has been abandoned
- Standard eval command:
  ```bash
  export HF_TOKEN=<token> && sbatch --time 03:00:00 --nodes 16 sbatch_eval.sh \
    <checkpoint_path> llama gmai,slake,path_vqa
  ```
- With 4 nodes the eval takes ~3.5h; with 16 nodes ~50 min
- Results saved to `/users/surech/meditron/reports/lmms_eval_results/<checkpoint_name>/`
- Current benchmarks: `gmai` (4550 samples, CoT), `slake` (642 samples), `path_vqa` (~6700 samples)

---

## Python / ML Coding Style

- Package root: `src/multimeditron/` — always add to `PYTHONPATH` when running scripts
- Model class: `MultiModalModelForCausalLM` (in `src/multimeditron/model/model.py`)
- Config class: `MultiModalConfig`, `model_type = "multimodal"`
- Use `dtype=dtype` (not `torch_dtype=dtype`) when calling `AutoConfig.from_pretrained`
- Lazy-import optional heavy dependencies (e.g. `decord`) inside `try/except ImportError` blocks — do not import at module top-level
- Do not add `decord` to pip install lists — it is not available in the eval environment
- WandB offline sync: submit a debug job with `multimeditron.toml` container and run `wandb sync <run_dir>`

---

## Keeping These Instructions Up to Date

During every conversation, if you discover new facts that are **stable and reusable** — such as a newly confirmed checkpoint path, a fixed bug, a corrected command, a timing benchmark, or a pipeline rule — **add it to this file immediately** using the edit tool. Do not wait to be asked.

Good candidates to record:
- New checkpoint paths or latest checkpoint numbers
- Commands that were confirmed to work (with node counts, time limits, flags)
- Bugs that were fixed and what the fix was
- Things that do NOT work (e.g. vLLM, decord in eval, wrong class names)
- Timing / throughput benchmarks (e.g. eval time per node count)

Do **not** record: speculative information, temporary debugging attempts, or anything that will change on the next run.

---

## Eval Results Log

> **Baseline**: `MultiMeditron LLaMA3.1-8B ATTN-PEP` (5-expert, master branch) — figures from `cookbook/README.md`.
> checkpoint-900 was a failed 7-expert run where the gating network was NOT retrained — do not use as baseline.

| Model / Checkpoint | GMAI | SLAKE overall | SLAKE yes/no | PathVQA overall | PathVQA yes/no |
|--------------------|------|---------------|--------------|-----------------|----------------|
| ATTN-PEP 5-exp baseline (master) | 29.6% | 29.6% | 51.1% | 30.3% | 59.1% |
| checkpoint-800 (7-exp, trained gating) | 31.1% | 30.6% | 51.1% | 24.4% | 47.1% |
| checkpoint-900 (7-exp, NO retrained gating) | 29.7% | 29.6% | 50.5% | 23.9% | 47.0% |

- PathVQA yes/no regressed significantly (59.1 → 47.1%) — **root cause identified**: the 7-expert model has a severe "No" prediction bias. It predicts "No" 89.3% of the time on binary questions (TP for Yes = 10.8%, TN for No = 89.8%). Full confusion matrix in `reports/EVAL_ANALYSIS.md`.
- **PathVQA routing root cause**: 5-expert gating sends 98.2% of histopathology images to Generalist (good calibration). 7-expert gating scatters them: 52.2% Skin, 37.2% MRI, 8.8% Generalist — both Skin and MRI are out-of-distribution for histopathology and default to "No". There is **no dedicated histopathology expert** in either model.
- checkpoint-900 (gating NOT retrained) performs nearly identically to checkpoint-800 on GMAI/SLAKE but confirms the PathVQA regression is present in both gating configurations.
- **Pending eval**: 5-expert baseline (capstor checkpoint-3063) submitted as job 1752923 (16 nodes, ~50 min). This will give fresh numbers on the same eval pipeline for a fair apple-to-apple comparison.

### Per-modality results (7-exp ckpt-800 vs 5-exp ckpt-3063)

Full analysis: `reports/EVAL_ANALYSIS.md`

| Subtask | 5-exp | 7-exp | Δ |
|---|---|---|---|
| GMAI Dermatology | 30.8% | 39.5% | +8.7 ✅ |
| GMAI Ophthalmology | 35.3% | 31.8% | −3.5 ⚠️ |
| SLAKE MRI overall | 21.5% | 23.7% | +2.2 ✅ |
| SLAKE X-ray y/n | 66.7% | 52.5% | −14.2 ⚠️ |

`scripts/train_gating.py` now supports a **3-way split**: train / val / test.
- `test_split` (default 0.1) is carved off **first**, before train/val — it is deterministic and never seen during training.
- After training, the best checkpoint is evaluated on the test set and results written to `<output_dir>/test_results.json`.
- Config key: `test_split: 0.1` in `config/gating_7class.yaml`; CLI: `--test_split 0.1`; set `0` to disable.
- **Before this change**, only a 90/10 train/val split existed — the validation set could influence training via model selection, making reported val accuracy an optimistic estimate.

## Documentation Rules

- **Never edit `README.md` or `cookbook/README.md` directly** with analysis results or experiment findings. These are stable user-facing docs. Instead, create a separate document in `reports/` (e.g. `reports/EVAL_ANALYSIS.md`, `reports/DATA_AUDIT.md`) and add a one-line pointer in the README.
- `reports/EVAL_ANALYSIS.md` — full per-modality eval results, PathVQA investigation, 7-exp vs 5-exp comparison tables.
- `reports/DATA_AUDIT.md` — duplicate analysis results and deduplication recommendations.

---

## Gating Network — Test Split — tests both gating checkpoints on 200 images × 5 modality datasets.  
5-expert gating: `/capstor/.../models--ClosedMeditron--MultiMeditron-Gating/snapshots/e1d1310b6e1962857b61b0009b9a4d7e196e84fa`  
7-expert gating: `models/CLIP/MultiMeditron-Gating`

**Coverage**: 5 of 7 expert modalities tested. **MRI and Generalist could NOT be tested** — no standalone MRI-only arrow dataset exists anywhere on the cluster (checked all of `multimediset/arrow/`, all home dirs, HF cache). To add MRI, download BraTS or IXI dataset and update `DATASETS` dict in the script.

### 7-expert gating — CORRECT FOR ALL MODALITIES ✅

| Dataset | Top-1 Expert | % |
|---------|-------------|---|
| CT (ct2) | CT | 100% |
| X-ray (iu_xray) | X-ray | 100% |
| Ultrasound (BUSI) | Ultrasound | 99% |
| Eye (eye_dataset) | Ophthalmology | 100% |
| Skin (skin_dataset) | Skin | 99% |

### 5-expert gating — CT ROUTING IS BROKEN ❌

| Dataset | Top-1 Expert | % | Correct? |
|---------|-------------|---|----------|
| CT (ct2) | **Ultrasound** | 96.5% | ❌ CT expert gets 0% |
| X-ray (iu_xray) | X-ray | 97.5% | ✅ |
| Ultrasound (BUSI) | Ultrasound | 98.0% | ✅ |
| Eye (eye_dataset) | Generalist | 94.0% | ⚠️ no expert exists |
| Skin (skin_dataset) | Generalist | 96.0% | ⚠️ no expert exists |

**Key finding**: The 5-expert model was routing all CT images to the Ultrasound expert. The 7-expert retraining fixed this. The GMAI/SLAKE improvement in checkpoint-800 (+1.5%/+1%) likely reflects correct CT routing as much as the new experts.

## Model Path Registry

All published model checkpoints are on capstor at:
`/capstor/store/cscs/swissai/a127/homes/meditron/models/multimeditron/`

- **5-expert ATTN-PEP baseline (cookbook)**: `unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end/checkpoint-3063`
- **7-expert ATTN-PEP (current)**: `/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end-7exp/checkpoint-800`
- Full model path registry with all variants added to `cookbook/README.md`
