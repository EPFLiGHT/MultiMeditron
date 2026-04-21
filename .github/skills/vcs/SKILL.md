---
name: vcs
description: 'Git version control workflow for MultiMeditron. Use when: creating commits, staging changes, pushing branches, writing commit messages, grouping changes into logical commits, preparing a PR, checking git status, reviewing what changed. Enforces the project commit convention (conventional commits with project-specific types). DO NOT USE FOR: general coding questions or SLURM/cluster operations.'
argument-hint: 'Describe what you want to commit, push, or review (e.g. "commit the new eval tasks and push")'
---

# VCS — Version Control Workflow

## Commit Message Convention

This repo uses **Conventional Commits** with a subset of types observed in the history. Always use lowercase type prefix followed by colon and space.

### Allowed Types

| Type | When to use | Example |
|------|-------------|---------|
| `feat` | New model capability, new pipeline feature, new expert | `feat: add ophthalmology CLIP expert encoder` |
| `fix` | Bug fix in code, config, or script | `fix: resolve dtype kwarg in AutoConfig.from_pretrained` |
| `config` | Changes to YAML training/eval configs only | `config: update stage2 end-to-end for 7-expert MoE` |
| `eval` | Eval scripts, lmms-eval tasks, benchmark additions | `eval: add gmai_ophthalmology and gmai_dermatology subtasks` |
| `infra` | SLURM scripts, container setup, cluster tooling | `infra: add GPU monitoring to training script` |
| `chore` | Dependency bumps, submodule updates, gitignore, cleanup | `chore: update third-party submodule pointers` |
| `docs` | README, cookbook docs, guides | `docs: replace placeholder paths in gating guide` |
| `scripts` | Standalone data/processing scripts in `scripts/` | `scripts: restore convert_image_datasets.py` |
| `test` | Debug configs, test pipelines | `test: add debug and test configs for pipeline validation` |
| `ci` | GitHub Actions, Docker workflow | `ci: stop building Dockerfile.verl in docker workflow` |

### Rules
- **All lowercase** for type and scope
- Subject line: imperative mood, no period at end, max ~72 chars
- No scope parentheses needed (e.g. `fix:` not `fix(model):`) — not used in this repo
- Free-form prose commits (no prefix) are acceptable for large exploratory changes but discouraged for PRs

---

## Branch Naming Convention

Observed pattern: `<topic>-<description-with-dashes>`

Examples from this repo:
- `add-ophthalmology-and-dermatology-experts`
- `surech-stage-work-backup`
- `fix-typo-1`
- `clean-model-config`
- `process-datasets`

**Rule**: lowercase, hyphens only, descriptive of the feature or fix.

---

## Standard Workflow

### 1. Check status before doing anything
```bash
git status
git diff --stat
```

### 2. Group changes into logical commits

Group by **concern**, not by file. Typical groupings for this project:

| Group | What belongs together |
|---|---|
| Model/code changes | `src/multimeditron/**` |
| Training configs | `cookbook/sft/**/*.yaml` |
| Eval tasks | `third-party/lmms-eval/lmms_eval/tasks/**` |
| Infrastructure | `sbatch_*.sh`, `config/deepspeed*.json` |
| Docs | `*.md`, `cookbook/*/README.md` |
| Submodules | `third-party/` pointer bumps |
| Workspace instructions | `.github/copilot-instructions.md`, `.github/skills/**` |

### 3. Stage and commit each group
```bash
git add <files>
git commit -m "<type>: <subject>"
```

### 4. Push to current branch
```bash
git push origin <branch-name>
```

Always push to the **current feature branch**, never directly to `master`.

---

## Current Branch Context

- **Active branch**: `add-ophthalmology-and-dermatology-experts`
- **Remote**: `origin` → `EPFLiGHT/MultiMeditron` on GitHub
- **Default branch**: `master`
- PRs merge into `master`

### Common Pending Changes (recurring across sessions)

| Files | Suggested commit |
|---|---|
| `third-party/lmms-eval/lmms_eval/tasks/gmai/gmai_ophthalmology.yaml` | `eval: add gmai_ophthalmology subtask` |
| `third-party/lmms-eval/lmms_eval/tasks/gmai/gmai_dermatology.yaml` | `eval: add gmai_dermatology subtask` |
| `third-party/lmms-eval/lmms_eval/tasks/gmai/utils.py` | (include with above eval commit) |
| `cookbook/gating/README.md` | `docs: replace placeholder paths in gating guide` |
| `.github/copilot-instructions.md` | `chore: update copilot workspace instructions` |
| `.github/skills/**` | `chore: add VCS skill` |

---

## Safety Rules

- **Never** `git push --force` or `git reset --hard` without explicit user confirmation
- **Never** commit directly to `master`
- **Never** include secrets (HF tokens, WandB keys) in commits — check with `git diff --staged` before committing
- Submodule pointer bumps (`third-party/`) should be their own `chore:` commit

---

## Checking What's Pending

```bash
# Unstaged changes
git status

# What changed vs master
git diff master...HEAD --stat

# Full log of current branch vs master
git log master..HEAD --oneline
```
