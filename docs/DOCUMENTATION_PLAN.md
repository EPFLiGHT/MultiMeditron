# MultiMeditron — Unified Documentation Plan

## 1. Current State Inventory

### 1.1 Sphinx Site (`docs/source/`)

| File | Content | Quality |
|---|---|---|
| `index.rst` | Landing page, overview, toctree | Good |
| `guides/quickstart.rst` | pip + Docker install, first inference | Good |
| `guides/training.rst` | YAML config walkthrough, single/multi-node launch | Good, generic |
| `guides/add_modality.rst` | Step-by-step modality extension guide | Good |
| `guides/configuration.rst` | YAML field reference | Adequate |
| `guides/dataset_format.rst` | Arrow/Parquet + JSONL, validation CLI | Good |
| `guides/modalities/image.rst` | Image-specific format + PIL examples | Good |
| `guides/known_issues.rst` | Docker permissions fix | Minimal (single issue) |
| `ref/*.rst` | Auto-generated API reference (sphinx-apidoc) | Depends on docstrings |

**Extensions**: autodoc, napoleon, sphinx_tabs, sphinx_click · **Theme**: shibuya

### 1.2 Markdown Files

| File | Lines | Audience | Quality |
|---|---|---|---|
| `README.md` | 144 | External users | Good — badges, features, setup, inference |
| `cookbook/README.md` | 277 | Lab researchers | Good — experiment matrix, eval table, full cookbook guide |
| `cookbook/gating/README.md` | 625 | Internal (surech) | Complete — MoE gating training guide |
| `cookbook/multi-node-nccl-fix.md` | 109 | Internal | Good troubleshooting guide |
| `scripts/README.md` | 101 | Internal | Outdated — older router/expert training |
| `scripts/clip_playground/README.md` | 10 | Internal | Stub |
| `ui/README.md` | 52 | Users | Adequate — Gradio UI usage |
| `models/CLIP/*/README.md` (×6) | 198 each | External | **Unfilled HuggingFace model card templates** |
| `.github/copilot-instructions.md` | ~80 | CI/Copilot | Working internal doc |

### 1.3 Python Docstring Coverage (41% overall)

| Module | Total | Documented | Coverage |
|---|---|---|---|
| `(root)` — `__init__`, `profiling`, `__main__` | 21 | 1 | **5%** |
| `cli` | 25 | 5 | **20%** |
| `dataset.preprocessor` | 19 | 0 | **0%** |
| `utils` | 7 | 0 | **0%** |
| `experts` | 19 | 6 | **32%** |
| `tools` | 6 | 2 | **33%** |
| `train` | 8 | 3 | **38%** |
| `model.modalities` | 69 | 30 | 43% |
| `model` | 52 | 29 | 56% |
| `dataset.loader` | 9 | 8 | 89% |
| `dataset`, `dataset.loader.image`, `model.modalities.moe`, `model.projectors` | — | — | 100% |

---

## 2. Gaps Analysis

### 2.1 Missing Topics (not covered anywhere)

| Topic | Impact | Where it should go |
|---|---|---|
| **MoE / Mixture-of-Experts architecture** | High | Sphinx guide + architecture overview |
| **Evaluation pipeline** (`sbatch_eval.sh`, lmms-eval) | High | Sphinx guide (currently only in cookbook README) |
| **CSCS / SLURM deployment** | Medium | Sphinx guide (CSCS-specific deployment) |
| **Gating mechanism** (PEP, attention-based routing) | Medium | Sphinx guide (linked from MoE guide) |
| **Adding new experts** (not just new modalities) | Medium | Sphinx guide or extend `add_modality.rst` |
| **Checkpoint management** (resuming, converting, ZeRO shards) | Medium | Sphinx guide or training.rst section |
| **WandB logging / offline sync** | Low | Training guide addendum |

### 2.2 Content Duplication

| Content | Locations | Resolution |
|---|---|---|
| Installation instructions | `README.md`, `quickstart.rst`, `cookbook/README.md` | Keep README.md minimal (link to Sphinx). Keep cookbook self-contained for lab users. |
| Inference example | `README.md`, `quickstart.rst` | Same code — keep in sync or DRY via include |
| Training launch commands | `training.rst`, `cookbook/README.md`, `sbatch_train.sh` | Sphinx = generic guide, cookbook = lab-specific recipes |
| Eval instructions | `cookbook/README.md` only | Add to Sphinx as a proper guide |

### 2.3 Stale / Stub Content

| Item | Problem | Action |
|---|---|---|
| `models/CLIP/*/README.md` (×6) | Blank HuggingFace model card templates | Fill with real model info or delete |
| `scripts/README.md` | References older router training workflow | Update or mark as legacy |
| `scripts/clip_playground/README.md` | 10-line stub | Flesh out or remove |
| `cookbook/README.md` → `envsubst` tip | Broken on CSCS (documented in copilot-instructions) | Add warning |

---

## 3. Proposed Documentation Architecture

```
docs/source/
├── index.rst                          # Landing page (keep as-is)
├── guides/
│   ├── guide.rst                      # Parent toctree
│   ├── quickstart.rst                 # Install + first inference (keep)
│   ├── training.rst                   # Generic training guide (keep, extend)
│   ├── evaluation.rst                 # NEW — eval pipeline guide
│   ├── moe.rst                        # NEW — MoE architecture & gating
│   ├── add_modality.rst               # Extending modalities (keep)
│   ├── add_expert.rst                 # NEW — adding a new CLIP expert
│   ├── configuration.rst              # YAML reference (keep, extend for MoE fields)
│   ├── dataset_format.rst             # Dataset format (keep)
│   │   └── modalities/image.rst       # Image format (keep)
│   ├── deployment.rst                 # NEW — CSCS/SLURM deployment guide
│   ├── known_issues.rst               # Known issues (keep, extend)
│   └── troubleshooting.rst            # NEW — merge NCCL fix + other issues
├── ref/
│   └── modules.rst                    # Auto-generated API reference (keep)
```

### Markdown stays as-is for:
- `README.md` — GitHub landing page (external users)
- `cookbook/README.md` — Self-contained lab cookbook (researchers)
- `cookbook/gating/README.md` — Internal gating training guide
- `.github/copilot-instructions.md` — Internal Copilot context
- `ui/README.md` — Gradio UI usage

---

## 4. Implementation Phases

### Phase 1: Foundation — Docstrings (Priority: Highest)

Docstrings feed into the auto-generated API reference. Without them, `ref/*.rst` pages are empty shells.

**Target modules** (ordered by impact):

1. `dataset.preprocessor` (0% → 80%+) — 19 items, critical data pipeline
2. `utils` (0% → 80%+) — 7 items, small and quick
3. `cli` (20% → 80%+) — 25 items, user-facing entry points
4. `(root)` / `profiling.py` (5% → 50%+) — 21 items, callbacks + profiling
5. `experts` (32% → 80%+) — 19 items, core MoE logic
6. `train` (38% → 80%+) — 8 items, training loop
7. `tools` (33% → 80%+) — 6 items

**Goal**: Raise overall coverage from 41% → 75%+

**Style**: Google-style docstrings (napoleon-compatible). One-line summary + Args/Returns/Raises blocks. No over-documenting trivial `__init__` methods.

### Phase 2: New Sphinx Guides (Priority: High)

#### 2a. `guides/evaluation.rst`
- What benchmarks are supported (GMAI, SLAKE, PathVQA, + subtask filters)
- How to run eval: `sbatch_eval.sh` usage, accelerate-based pipeline
- Node count vs. time tradeoffs
- Reading results from output directory
- **Source material**: `cookbook/README.md` eval section, `sbatch_eval.sh`

#### 2b. `guides/moe.rst`
- Architecture overview: multi-expert CLIP encoders, projection, gating
- Fusion methods (attn vs. avg) and projection strategies (PEP vs. shared)
- Gating mechanism (attention-based PEP routing)
- Training stages for MoE (Stage 1 alignment → Stage 2 end-to-end)
- **Source material**: `cookbook/gating/README.md`, model code

#### 2c. `guides/add_expert.rst`
- How to add a new CLIP expert encoder (e.g. ophthalmology, dermatology)
- Config changes, gating retraining, checkpoint compatibility
- Distinct from `add_modality.rst` (which covers new modality types, not new experts for existing modality)

#### 2d. `guides/deployment.rst`
- CSCS-specific: EDF files, SLURM sbatch scripts, node types
- Environment setup (`.env` file, HF_HOME, WANDB)
- Generic cluster deployment notes
- **Source material**: `cookbook/README.md` CSCS section

### Phase 3: Extend Existing Guides (Priority: Medium)

| Guide | Changes |
|---|---|
| `training.rst` | Add MoE config example, checkpoint resumption, DeepSpeed notes |
| `configuration.rst` | Document MoE-specific YAML fields (`experts`, `gating_path`, `fusion_method`, `expert_projection`) |
| `known_issues.rst` | Add: envsubst on CSCS, NCCL GDR level, ZeRO shard count must match node count |
| `guide.rst` (toctree) | Add entries for new pages: evaluation, moe, add_expert, deployment |

### Phase 4: Cleanup (Priority: Low)

| Item | Action |
|---|---|
| `models/CLIP/*/README.md` | Fill model cards with: architecture, training data, intended use, limitations. OR delete if not publishing to HF Hub. |
| `scripts/README.md` | Mark as legacy or update to current workflow |
| `scripts/clip_playground/README.md` | Flesh out or delete |
| Content sync: `README.md` ↔ `quickstart.rst` | Ensure inference example stays identical |

---

## 5. Style Rules (from `.github/skills/write-docs/SKILL.md`)

- **Docstrings**: Google-style, napoleon-compatible
- **RST guides**: No emojis, professional tone, numbered steps for procedures
- **Markdown READMEs**: Emojis in section headers, badges at top
- **Code examples**: Always include language hint (python, bash, yaml)
- **Cross-references**: Use `:ref:` and `:any:` in RST; relative links in markdown
- **API links**: Use `:class:`, `:meth:`, `:func:` roles in RST to link to autodoc pages

---

## 6. Success Metrics

> **Status (2026-06-05):** docstring coverage now measures **~75%** across
> `src/multimeditron` (the "41%" below was the original baseline) — target met;
> remaining gaps are mostly trivial dunders / `forward` overrides. New Sphinx
> guides added: `moe`, `evaluation`, `add_expert`, `deployment`, `troubleshooting`.

| Metric | Baseline | Target | Now |
|---|---|---|---|
| Docstring coverage | 41% | 75%+ | ~75% ✅ |
| Sphinx guide pages | 7 | 11 | 12 ✅ |
| Zero-coverage modules | 2 (`preprocessor`, `utils`) | 0 | 0 ✅ |
| Unfilled model cards | 6 | 0 | 6 (deferred) |
| Stale/stub markdown files | 2 | 0 | 1 |
