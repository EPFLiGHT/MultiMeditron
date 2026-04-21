# Agent Test Instructions

This file is used to test what a fresh agent can accomplish using only the existing documentation.
No context from previous sessions should be used. Derive all commands from `cookbook/README.md`,
`scripts/README.md`, and the source code.

---

## Tasks

1. **Add ophthalmology and skin experts to MultiMeditron and run fine-tuning** (Stage 1 alignment then Stage 2 end-to-end).
2. **Evaluate the resulting model** on the standard benchmarks.

---

## Constraints

- **Dry runs only.** All submitted jobs must use:
  - `--partition=debug`
  - `--nodes=1`
  - `--time=00:01:00`
- The jobs are expected to fail or time out — the goal is to verify that the submission commands and configs are correct, not to produce trained weights.

---

## Ignore list

The following files exist only on this branch and **must not be used**. Pretend they do not exist.

```
.json.json
.github/copilot-instructions.md
.github/skills/vcs/SKILL.md
.github/skills/write-docs/SKILL.md
.github/instructions_agent.md         ← this file
config/deepspeed_fast.json
config/gating_7class.yaml
cookbook/DATA_AUDIT.md
cookbook/REGISTRY.md
cookbook/gating/README.md
cookbook/multi-node-nccl-fix.md
cookbook/sft/moe/attn/pep/stage1_alignment_debug.yaml
cookbook/sft/moe/attn/pep/test_stage1.yaml
cookbook/sft/moe/attn/pep/test_stage2.yaml
docs/DOCUMENTATION_PLAN.md
download_data.py
sbatch_eval.sh
sbatch_test_pipeline.sh
sbatch_train.sh
sbatch_train_gating.sh
sbatch_train_gating_debug.sh
scripts/compare_modality_results.py
scripts/convert_image_datasets.py
scripts/test_gating.py
scripts/train_gating.py
third-party/lmms-eval
PROGRESS_SUMMARY.md
```
