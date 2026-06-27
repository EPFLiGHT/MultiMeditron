# Evaluation Analysis — 7-Expert vs 5-Expert ATTN-PEP

> **Date:** 2026-04-15
> **Branch:** `add-ophthalmology-and-dermatology-experts`
> **Models compared:**
> - 5-expert baseline: `MultiMeditron-8B-attn-pep-end2end/checkpoint-3063` (capstor)
> - 7-expert current: `MultiMeditron-8B-attn-pep-end2end-7exp/checkpoint-800` (iopsstor scratch)
>
> Raw results: `/users/surech/meditron/reports/lmms_eval_results/`

---

## 1 — Top-Level Benchmarks

| Benchmark | 5-expert (ckpt-3063) | 7-expert (ckpt-800) | Δ | Significant? |
|---|---|---|---|---|
| **GMAI** | 29.6% | 31.1% | **+1.5** | ⚠️ no (CIs overlap) |
| **SLAKE overall** | 29.6% | 30.6% | **+1.0** | ⚠️ no (CIs overlap) |
| SLAKE yes/no | 51.1% | 51.1% | 0.0 | — |
| **PathVQA overall** | 30.1% | 24.4% | **−5.7** | ✅ yes (CIs separated) |
| PathVQA yes/no | 58.6% | 47.1% | **−11.5** | ✅ yes (CIs separated) |

---

## 1.1 — Statistical Significance

Confidence intervals are **Wilson 95%** intervals computed from each benchmark's
sample size; non-overlapping intervals indicate a real difference (a conservative
test). Reproduce any value with `scripts/eval_significance.py ci --acc <p> --n <n>`.

| Metric | 5-exp 95% CI | 7-exp 95% CI | Verdict |
|---|---|---|---|
| GMAI (n=4550) | 29.6% [28.3, 30.9] | 31.1% [29.8, 32.5] | **overlap → not significant** |
| SLAKE overall (n=642) | 29.6% [26.2, 33.2] | 30.6% [27.2, 34.3] | **overlap → not significant** |
| PathVQA overall (n=6719) | 30.1% [29.0, 31.2] | 24.4% [23.4, 25.4] | separated → significant |
| PathVQA yes/no (n=3362) | 58.6% [56.9, 60.3] | 47.1% [45.4, 48.8] | separated → significant |

**Interpretation:** the headline GMAI (+1.5) and SLAKE (+1.0) gains are **within
noise** — they should not be reported as improvements without more data. The
PathVQA regressions are well outside the intervals and are real.

**Correct paired test.** Wilson CIs treat the two runs as independent; because both
models answer the *same* questions, the proper test is **McNemar's** on the paired
per-item correctness. The exact test is implemented in
`scripts/eval_significance.py mcnemar --preds-a <5exp samples.jsonl> --preds-b <7exp samples.jsonl>`
(reads lmms-eval's per-sample JSONL). It is **not run here** because the paired
per-item outputs were not retained for both checkpoints — re-running eval with
`--log_samples` and feeding the two files to the script will give the definitive
p-values, especially for the borderline GMAI/SLAKE deltas.

---

## 2 — GMAI Per-Modality (7-expert, checkpoint-800)

> No 5-expert per-modality GMAI run exists — direct delta cannot be computed.

| Modality | Accuracy |
|---|---|
| OCT | 40.0% |
| Microscopy | 39.4% |
| Dermoscopy | 39.0% |
| Histopathology | 32.9% |
| MRI | 33.0% |
| Ophthalmology | 31.8% *(new expert)* |
| Ultrasound | 31.4% |
| Fundus | 27.9% |
| CT | 28.9% |
| X-ray | 28.1% |
| Endoscopy | 25.2% |

---

## 3 — GMAI Ophthalmology & Dermatology vs Baseline

| Subtask | 5-expert (ckpt-3063) | 7-expert (ckpt-800) | Δ |
|---|---|---|---|
| GMAI Dermatology | 30.8% | 39.5% | **+8.7** ✅ |
| GMAI Ophthalmology | 35.3% | 31.8% | **−3.5** ⚠️ |

**Observations:**
- Dermatology benefited strongly from the dedicated SkinExpert (+8.7%).
- Ophthalmology regressed slightly (−3.5%) despite the dedicated OphthalmologyExpert. Likely cause: GMAI `gmai_ophthalmology` includes OCT and fundus images which the generalist expert handled well before. The new expert may be more specialised on slit-lamp / external eye images only.

---

## 4 — SLAKE Per-Modality

| Modality | 5-exp overall | 7-exp overall | Δ overall | 5-exp y/n | 7-exp y/n | Δ y/n |
|---|---|---|---|---|---|---|
| CT | 28.4% | 28.0% | −0.4 | 49.0% | 50.0% | +1.0 |
| MRI | 21.5% | 23.7% | **+2.2** ✅ | 41.8% | 50.6% | **+8.8** ✅ |
| X-ray | 37.4% | 37.7% | +0.3 | 66.7% | 52.5% | **−14.2** ⚠️ |

**Observations:**
- MRI improved strongly on both overall (+2.2%) and yes/no (+8.8%) — likely from improved routing now that CT images no longer pollute the Ultrasound expert path.
- X-ray yes/no regressed (66.7% → 52.5%). The 5-expert baseline's unusually high 66.7% may indicate a small sample size artefact (SLAKE X-ray has ~99 binary questions); worth verifying sample counts.

---

## 5 — PathVQA Failure Investigation

### Dataset composition (6,719 samples)

| Question type | Count |
|---|---|
| Binary (yes/no) | 3,366 |
| Open-ended | 3,353 |

### Binary accuracy breakdown

| | GT = Yes (n=1,816) | GT = No (n=1,546) |
|---|---|---|
| Model predicts **Yes** | **197** (10.8% TP) | 156 (10.1% FP) |
| Model predicts **No** | 1,614 (88.9% FN) | **1,389** (89.8% TN) |

- Model says "No" **89.3%** of the time (3,007 / 3,366 binary questions).
- Sensitivity (recall for Yes): **10.8%** — catastrophically low.
- Specificity (recall for No): **89.8%** — artificially high.
- Open-ended accuracy: **1.5%** (unchanged from baseline — open PathVQA is hard for all models).

### Root cause

The 7-expert model has developed a near-complete **"No" prediction bias** on PathVQA binary questions. This is the entire reason for the yes/no regression (58.6% → 47.1%).

### Routing analysis (500 test images, `scripts/pathvqa_routing_analysis.py`)

PathVQA images are histopathology / microscopy slides. The gating network routes them very differently between the two models:

**5-expert gating:**
| Expert | Top-1 % | Avg weight |
|---|---|---|
| **Generalist** | **98.2%** | 0.982 |
| Ultrasound | 1.0% | 0.010 |
| MRI | 0.4% | 0.003 |
| CT | 0.2% | 0.002 |
| X-ray | 0.2% | 0.002 |

**7-expert gating:**
| Expert | Top-1 % | Avg weight |
|---|---|---|
| **Skin** | **52.2%** | 0.453 |
| **MRI** | **37.2%** | 0.354 |
| Generalist | 8.8% | 0.121 |
| X-ray | 1.2% | 0.022 |
| Ophthalmology | 0.6% | 0.029 |
| CT | 0.0% | 0.009 |
| Ultrasound | 0.0% | 0.013 |

**Interpretation:**
- The 5-expert model routes **98.2%** of histopathology images to the Generalist — a single dominant path. The model was well-calibrated under this expert (Yes recall 38.7%, No recall 78.7%).
- The 7-expert model **scatters routing**: 52.2% to Skin, 37.2% to MRI, only 8.8% Generalist. Neither the Skin expert (trained on dermoscopy surface images) nor the MRI expert (trained on MRI scans) has seen histopathology tissue during training. They default to "No" when uncertain on these out-of-distribution inputs.
- The MRI routing (37.2%) likely occurs because microscopy tissue sections share grayscale/contrast patterns similar to MRI slices.
- There is **no dedicated histopathology expert** in either the 5-expert or 7-expert setup. The correct fallback is the Generalist, which handled it well in the 5-expert model.

### Next steps

1. Investigate whether the yes/no instruction format in PathVQA prompts is being overridden by the expert's generation style.
2. Consider adding PathVQA-format binary QA examples to Stage 2 training data for the Skin/Generalist expert.
3. Consider adding a **Histopathology/Microscopy expert** (e.g. trained on TCGA or similar) to give proper routing for this important modality.
4. Alternatively, adjust the gating training data to incWlude histopathology samples labeled as Generalist, explicitly steering them away from Skin/MRI.
