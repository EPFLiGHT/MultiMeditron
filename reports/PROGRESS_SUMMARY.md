# MultiMeditron — Progress Summary (March 27 – April 11, 2026)

Branch: `add-ophthalmology-and-dermatology-experts`  
PR: [#54 — Add ophthalmology and dermatology experts](https://github.com/EPFLiGHT/MultiMeditron/pull/54)

---

## Context: Background Before March 27

By March 21–24, the branch was set up for 7-expert training (adding Ophthalmology and Skin experts to the original 5: CT, MRI, Ultrasound, X-ray, Generalist). Key pre-period commits:

| Date | Commit | Description |
|------|--------|-------------|
| Mar 21 | `2ceb21a` | `feat: 7-class gating network training pipeline` |
| Mar 21 | `c818d35` | `infra: multi-node evaluation and vLLM eval scripts` |
| Mar 21 | `840a631` | `config: update training hyperparameters for stage 1 & 2` |
| Mar 23 | `e4d4edf` | `feat: improve gating training script for 7-class training` |
| Mar 23 | `f7abc14` | `config: update stage1 alignment for 7-expert training` |
| Mar 23 | `2402eaa` | `config: update stage2 end-to-end for 7-expert MoE` |
| Mar 24 | `7b1b715` | `Training documentation proposal` |
| Mar 24 | `8d6c069` | `Deleted vllm training script (artifact)` |

The 7-class gating model had been trained and checkpoint-800 of Stage 2 end-to-end was in place at this point.

---

## March 24 — First Full Evaluation of 7-Expert Model

**Goal**: get benchmark numbers for checkpoint-800 (the new 7-expert model) and compare against the 5-expert ATTN-PEP baseline.

### Roadblock: vLLM eval abandoned

Five successive vLLM eval attempts (jobs `1718196`, `1718362`, `1718536`, `1718755`, `1719051`) all failed. vLLM cannot load our custom `multimodal` model type. Switched to `sbatch_eval.sh` (accelerate-based) for all subsequent evals.

### Runs

| Job | Time | Checkpoint | Tasks | Nodes | Outcome |
|-----|------|-----------|-------|-------|---------|
| 1717981 | 15:36 | — | wandb sync | 1 | Synced WandB offline runs |
| 1720319 | 21:57 | checkpoint-800 (7-exp) | `gmai,slake,path_vqa` | 4 | Completed (~3.5h) |
| 1720371 | 22:58 | checkpoint-800 (7-exp) | `gmai,slake,path_vqa` | 16 | Completed (~50 min) |
| 1720838 | 23:16 | checkpoint-800 (7-exp) | `gmai_ophthalmology, gmai_dermatology` | 16 | Completed |
| 1720857 | 23:54 | checkpoint-800 (7-exp) | `gmai_ophthalmology, gmai_dermatology` | 2 | Completed |
| 1720858 | 23:31 | checkpoint-800 (7-exp) | `gmai_ophthalmology, gmai_dermatology` | 16 | Completed |

### Results (checkpoint-800 versus 5-expert baseline)

| Benchmark | 5-exp baseline | 7-exp ckpt-800 | Delta |
|-----------|---------------|----------------|-------|
| GMAI overall | 29.6% | 31.1% | **+1.5%** |
| SLAKE overall | 29.6% | 30.6% | **+1.0%** |
| SLAKE yes/no | 51.1% | 51.1% | 0.0% |
| PathVQA overall | 30.3% | 24.4% | **−5.9%** |
| PathVQA yes/no | 59.1% | 47.1% | **−12.0%** |

**Observation**: GMAI and SLAKE improved, but PathVQA yes/no regressed significantly (−12 pp). This is suspicious and was flagged for investigation.

---

## April 1 — Documentation, lmms-eval Fixes, Per-Modality Eval Setup

### Commits

| Hash | Description |
|------|-------------|
| `1238551` | `Added some documentation and comments in the code` — added `copilot-instructions.md`, VCS and write-docs Copilot skills, cookbook README extensions, RST documentation guides, and CLI module docs |
| `cff5daf` | `chore: bump lmms-eval submodule` — decord import fix (lazy import inside `try/except`); added `gmai_ophthalmology` and `gmai_dermatology` per-modality benchmark subtasks |

### Runs

**Goal**: test per-modality breakdown on the 5-expert baseline and compare with the 7-expert model.

| Job | Time | Checkpoint | Tasks | Nodes | Outcome |
|-----|------|-----------|-------|-------|---------|
| 1781599 | 18:35 | 5-exp baseline (ckpt-3063) | `gmai_ophthalmology, gmai_dermatology` | 1 | Testing setup, failed early |
| 1781645 | 21:03 | 5-exp baseline (ckpt-3063) | `gmai_ophthalmology, gmai_dermatology` | 1 | Completed |
| 1781753–1781758 | 18:51 | 5-exp baseline (ckpt-3063) | `gmai_ct, gmai_mri, gmai_endoscopy, gmai_histopathology, gmai_fundus, gmai_xray, gmai_microscopy, gmai_dermoscopy, gmai_ultrasound, gmai_oct` | 1 | Completed (multi-task per-modality breakdown) |

---

## April 2–3 — Dataset Audit, Per-Modality Comparison Runs

### Commits

| Hash | Description |
|------|-------------|
| `04ec7ce` | `Dataset analysis` — added `DATA_AUDIT.md` (full audit of all training/eval datasets), `scripts/compare_modality_results.py`, tweaked `sbatch_eval.sh` and Stage 1 YAML config |
| `16d6ad1` | `eval: add per-modality subtasks for gmai and slake benchmarks` — bumped lmms-eval submodule to add `slake_ct`, `slake_xray`, `slake_mri` per-modality SLAKE tasks |

### Runs

Head-to-head comparison of 5-expert (ckpt-3063) vs 7-expert (ckpt-800) on per-modality subtasks:

| Job | Time | Checkpoint | Tasks | Nodes | Outcome |
|-----|------|-----------|-------|-------|---------|
| 1787338 | Apr 2 14:42 | 5-exp baseline (ckpt-3063) | per-modality GMAI (10 subtasks) | 1 | Completed |
| 1787339 | Apr 2 14:42 | 7-exp ckpt-800 | per-modality GMAI (10 subtasks) | 1 | Completed |
| 1787488 | Apr 2 16:07 | 5-exp baseline (ckpt-3063) | `slake_ct, slake_xray, slake_mri` | 1 | Completed |
| 1787490 | Apr 2 16:15 | 7-exp ckpt-800 | `slake_ct, slake_xray, slake_mri` | 1 | Completed |
| 1787422 | Apr 3 02:58 | 5-exp baseline (ckpt-3063) | per-modality GMAI (10 subtasks) | 1 | Completed (larger run, ~4.9MB output) |
| 1787423 | Apr 3 02:33 | 7-exp ckpt-800 | per-modality GMAI (10 subtasks) | 1 | Completed (larger run, ~5.0MB output) |

---

## April 10 — Gating Routing Analysis (Contaminated Run)

**Goal**: understand which expert the gating network routes images to, and verify correct modality assignment. Motivated by supervisor concern about suspiciously high (100%) accuracy numbers in earlier discussions.

Script: `scripts/gating_routing_analysis.py` — tests 5-expert vs 7-expert gating on 200 images × 5 modality datasets.

| Job | Time | Outcome |
|-----|------|---------|
| 1822390 | 08:08 | Failed — script setup error |
| 1822391 | 08:11 | Partial — incomplete (3 datasets done) |
| 1822393 | 08:15 | Completed — **but results were contaminated** |

**Contamination issue discovered (job 1822393)**: The script was evaluating the gating models on the *training* splits:
- `eye_dataset` → pointed to training split (same data used to train the 7-expert gating)
- `skin_dataset` → pointed to training split (same data)

This explains why 7-expert Eye and Skin showed 100%/99% — the gating had memorised those exact images. The 5-expert CT routing bug was also masked on training data.

---

## April 11 — Clean Gating Routing Analysis (Held-Out Splits)

### Fix applied to `scripts/gating_routing_analysis.py`

Switched to held-out evaluation splits for all five modalities:

| Modality | Old (contaminated) split | New (clean) split | Size |
|----------|--------------------------|-------------------|------|
| CT | `ct2` (training) | `ct2_expert/test` | 821 rows |
| X-ray | `iu_xray` (training) | `XR-glob_expert/test` | 745 rows |
| Ultrasound | `BUSI` (training) | `BUSI_expert/test` | 156 rows |
| Eye | `eye_dataset` (**training**) | `eye_dataset/val` | 903 rows |
| Skin | `skin_dataset` (**training**) | `skin_dataset/val` | 2348 rows |

Also fixed schema handler: `modalities_images` column has schema `list[{bytes, path}]` (vs plain `image` column for eye/skin).

### Run

| Job | Time | Outcome |
|-----|------|---------|
| 1824106 | Apr 11 05:22 – 05:23 | Completed ✅ |

### Clean Results

**7-expert gating on held-out splits — CORRECT FOR ALL MODALITIES ✅**

| Dataset / split | Top-1 Expert | % |
|-----------------|-------------|---|
| CT (`ct2_expert/test`) | CT | 100% |
| X-ray (`XR-glob_expert/test`) | X-ray | 94.5% |
| Ultrasound (`BUSI_expert/test`) | Ultrasound | 97.4% |
| Eye (`eye_dataset/val`) | Ophthalmology | 100% |
| Skin (`skin_dataset/val`) | Skin | 99% |

**5-expert gating on held-out splits — CT IS BROKEN ❌**

| Dataset / split | Top-1 Expert | % | Correct? |
|-----------------|-------------|---|----------|
| CT (`ct2_expert/test`) | **Ultrasound** | 97.5% | ❌ CT gets 0% |
| X-ray (`XR-glob_expert/test`) | X-ray | 97.5% | ✅ |
| Ultrasound (`BUSI_expert/test`) | Ultrasound | 99.4% | ✅ |
| Eye (`eye_dataset/val`) | Generalist | 92.5% | ⚠️ no ophthalmology expert |
| Skin (`skin_dataset/val`) | Generalist | 97.5% | ⚠️ no skin expert |

**Key finding**: The 5-expert gating routes all CT images to the Ultrasound expert (a pre-existing bug). The 7-expert retraining fixed this. The GMAI improvement in checkpoint-800 (+1.5%) likely reflects correct CT routing. Eye and Skin routing is correct in 7-expert; in 5-expert those modalities fall through to Generalist as expected (no dedicated expert existed).

---

## Summary of Commits (March 27 – April 11)

| Date | Hash | Message |
|------|------|---------|
| Apr 1 | `1238551` | Added documentation and Copilot skills (VCS, write-docs) |
| Apr 1 | `cff5daf` | chore: bump lmms-eval submodule (decord fix + ophtho/derma tasks) |
| Apr 2 | `04ec7ce` | Dataset analysis (DATA_AUDIT.md, compare_modality_results.py) |
| Apr 2 | `16d6ad1` | eval: add per-modality subtasks for gmai and slake benchmarks |

---

## Open Questions / Next Steps

1. **PathVQA regression**: PathVQA yes/no dropped −12 pp (59.1 → 47.1%). Root cause identified — see April 15 section below.

2. **5-expert baseline fresh eval**: Job 1752923 (checkpoint-3063, 16 nodes) completed. Results incorporated into `cookbook/EVAL_ANALYSIS.md`.

3. **MRI routing unverified**: No standalone MRI-only arrow dataset was found on the cluster. To add MRI routing analysis, download BraTS or IXI and add to `DATASETS` dict in `scripts/gating_routing_analysis.py`.

4. **Generalist expert routing**: The Generalist routing (Eye → Generalist in 5-expert; Generalist fallback in 7-expert is near 0%) was not explicitly tested.

---

## April 15 — Full Evaluation Analysis & PathVQA Root Cause

### Commits & Documents

| Artifact | Description |
|----------|-------------|
| `cookbook/EVAL_ANALYSIS.md` (new) | Full per-modality eval comparison (7-exp vs 5-exp) across GMAI, SLAKE, PathVQA. Includes confusion matrices and routing analysis. |
| `cookbook/DATA_AUDIT.md` | Updated with duplicate analysis results and dedup recommendations |
| `cookbook/README.md` | Added one-line pointer to `EVAL_ANALYSIS.md` (no analysis content added directly) |
| `.github/copilot-instructions.md` | Updated with stable facts: PathVQA root cause, per-modality results, gating test split, documentation rules |
| `scripts/train_gating.py` | Added proper 3-way train/val/test split (`test_split` parameter) |
| `config/gating_7class.yaml` | Added `test_split: 0.1` |

### Eval analysis — top-level results

| Benchmark | 5-expert (ckpt-3063) | 7-expert (ckpt-800) | Δ |
|---|---|---|---|
| GMAI | 29.6% | 31.1% | +1.5 ✅ |
| SLAKE overall | 29.6% | 30.6% | +1.0 ✅ |
| SLAKE yes/no | 51.1% | 51.1% | 0.0 |
| PathVQA overall | 30.1% | 24.4% | −5.7 ❌ |
| PathVQA yes/no | 58.6% | 47.1% | −11.5 ❌ |

### Per-modality highlights

| Subtask | 5-exp | 7-exp | Δ |
|---|---|---|---|
| GMAI Dermatology | 30.8% | 39.5% | +8.7 ✅ |
| GMAI Ophthalmology | 35.3% | 31.8% | −3.5 ⚠️ |
| SLAKE MRI overall | 21.5% | 23.7% | +2.2 ✅ |
| SLAKE X-ray y/n | 66.7% | 52.5% | −14.2 ⚠️ |

### PathVQA root cause — "No" prediction bias

The 7-expert model says "No" on **89.3%** of PathVQA binary questions (vs ~61% in the 5-expert model):

| | GT = Yes (n=1,816) | GT = No (n=1,546) |
|---|---|---|
| Predicts **Yes** | 197 (10.8% TP) | 156 (10.1% FP) |
| Predicts **No** | 1,614 (88.9% FN) | 1,389 (89.8% TN) |

Yes recall collapsed from 38.7% (5-exp) to 10.8% (7-exp) — catastrophic.

### Gating test split added to `train_gating.py`

- `test_split` (default 0.1) is carved off **first**, before train/val — deterministic, never seen during training.
- After training, best checkpoint evaluated on test set → results in `<output_dir>/test_results.json`.
- Prior to this change, only a 90/10 train/val split existed — val accuracy was an optimistic estimate.

---

## April 16 — PathVQA Routing Analysis & Expert Architecture Investigation

### Script

Created `scripts/pathvqa_routing_analysis.py` — loads 500 PathVQA test images (histopathology/microscopy slides) and routes them through both the 5-expert and 7-expert gating networks.

**First attempt** (job 1870375) failed: HF cache had broken symlinks (parquet → blobs deleted). Fixed by switching from raw pyarrow parquet reading to `datasets.load_dataset()` which handles cache repair automatically.

### Runs

| Job | Time | Description | Outcome |
|-----|------|-------------|---------|
| 1870375 | 05:34 | PathVQA routing (parquet loader) | ❌ Broken symlinks in HF cache |
| 1870411 | 05:40 | PathVQA routing (datasets loader) | ✅ Completed |
| 1870342 | 05:24 | 5-exp per-modality GMAI eval (4 nodes) | ⏳ Running (~1h46 remaining) |

### PathVQA routing results (500 test images)

**5-expert gating — routes 98.2% to Generalist ✅**

| Expert | Top-1 % | Avg weight |
|---|---|---|
| **Generalist** | **98.2%** | 0.982 |
| Ultrasound | 1.0% | 0.010 |
| MRI | 0.4% | 0.003 |
| CT | 0.2% | 0.002 |
| X-ray | 0.2% | 0.002 |

**7-expert gating — scatters across Skin and MRI ❌**

| Expert | Top-1 % | Avg weight |
|---|---|---|
| **Skin** | **52.2%** | 0.453 |
| **MRI** | **37.2%** | 0.354 |
| Generalist | 8.8% | 0.121 |
| X-ray | 1.2% | 0.022 |
| Ophthalmology | 0.6% | 0.029 |
| CT | 0.0% | 0.009 |
| Ultrasound | 0.0% | 0.013 |

### Why routing changed — training data gap analysis

The gating network is a ResNet50 classifier retrained from scratch for each expert count. With 5 experts, histopathology images don't match any radiology class (CT/MRI/US/X-ray) so they default to **Generalist** (the only non-radiology bucket). With 7 experts, two new non-radiology classes compete:

- **Skin (63K dermoscopy images)**: dermoscopy and histopathology share tissue texture, pinkish/purplish staining, and close-up biological views — visually similar at the ResNet feature level.
- **MRI (PMC_VQA — 61K mixed medical)**: this grab-bag dataset includes microscopy images, so the MRI class learned features that partially overlap with histopathology.

The Generalist class (natural photos from `llava_pretrain`) now competes with Skin and Ophthalmology for the "non-radiology" space — and Skin is visually much closer to histopathology than natural photos. Result: route shifts from 98% Generalist → 52% Skin + 37% MRI.

**There is no dedicated histopathology expert in either model.** The correct fallback is the Generalist, which was well-calibrated for PathVQA binary questions (Yes recall 38.7%). When routed to Skin/MRI (out of distribution), the model defaults to "No".

### Potential fixes identified

1. Add histopathology samples to gating training data labeled as Generalist (steer away from Skin/MRI)
2. Add a dedicated Histopathology/Microscopy expert (8th class, trained on e.g. TCGA)
3. Add PathVQA-style binary QA to Stage 2 training for Skin/Generalist experts
4. Clean up MRI training data — replace `PMC_VQA` (mixed) with a pure MRI dataset (e.g. BraTS)
