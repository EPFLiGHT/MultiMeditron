---
name: write-docs
description: 'Write documentation for MultiMeditron code. Use when: writing or improving Python docstrings, writing README files, documenting a new module or class, writing a cookbook guide, documenting a script, explaining a training pipeline or eval step. Enforces the project documentation style extracted from the existing codebase. DO NOT USE FOR: inline code comments or SLURM job scripts.'
argument-hint: 'What to document (e.g. "docstrings for the new gating module", "README for the eval tasks", "cookbook guide for Stage 2 training")'
---

# Write Documentation — MultiMeditron Style Guide

## Where Documentation Lives

| Content type | Location | Format |
|---|---|---|
| Project overview, install, inference example | `README.md` | Markdown with emojis + badges |
| Training/eval pipeline guides | `cookbook/*/README.md` | Markdown, no emojis in body, tables + ASCII diagrams |
| Script usage | `scripts/README.md` | Compact markdown, bullet lists |
| Module/class/function descriptions | `src/multimeditron/**/*.py` | Google-style docstrings |
| Cluster-specific process notes | `cookbook/*.md` | Fenced steps + shell blocks |

---

## Python Docstring Style — Google Format

All Python docstrings in this project use **Google style**. No NumPy or reStructuredText.

### Class docstring
One concise sentence describing purpose. No sections needed for simple classes.

```python
class MultimodalConfig(PretrainedConfig):
    """
    Configuration class for a multimodal model that integrates various modalities with a language model.
    """
```

### `__init__` docstring
When `__init__` has meaningful parameters, document with `Args:` and `Raises:` sections.

```python
def __init__(self, config: MultimodalConfig, bootstrap=False):
    """
    Initialize a MultiModalModelForCausalLM instance.

    Brief description of what this sets up (1–2 sentences).

    Args:
        config (MultimodalConfig): The configuration object containing model parameters,
            including modality configurations, vocabulary size, and other settings.
        bootstrap (bool, optional): If True, loads pretrained weights from config path.
            Defaults to False.

    Raises:
        ValueError: If multiple modality configurations of the same type are provided.
    """
```

### Method docstring — with numbered procedure list
For methods that perform a sequence of operations, use a numbered list in the body:

```python
def freeze_for_alignment(self):
    """
    Freezes model parameters for alignment training.

    This method prepares the model for alignment training by:

    1. Freezing only the modality parts of each modality processor (keeping projections trainable)
    2. Freezing the entire language model

    This configuration is useful when aligning modality representations with
    the language model's embedding space while keeping the core LM frozen.
    """
```

### Method docstring — with `Returns:`
```python
def to_dict(self):
    """
    Converts the MultimodalConfig object to a dictionary representation.

    This method extends the parent class's to_dict method by properly handling
    the modalities list, converting each ModalityConfig object to its dictionary
    representation.

    Returns:
        dict: Dictionary containing all configuration parameters, with modalities
              properly serialized.
    """
```

### Rules
- First line: one sentence, no period at end, starts with a capital letter
- Blank line after first sentence before sections
- `Args:` — type in parentheses, default noted inline (e.g. `Defaults to False.`)
- `Returns:` — type colon description
- `Raises:` — ExceptionType: when it's raised
- Multi-line arg descriptions: indent 4 spaces from the arg name
- Do **not** add docstrings to trivial one-liners or property accessors

---

## Markdown Style — README & Guides

### Top-level README (`README.md`)
- Badges and centered logo at top (HTML `<div align="center">`)
- Section headers use `## 🚀 Key Features` — emoji + title, H2 level
- Feature lists use `* **Bold label:** Description` style
- Code blocks always tagged: ` ```bash `, ` ```python `, ` ```yaml `
- Sections: Key Features → Architecture → Setup → Inference Example → (further detail)

### Cookbook / Pipeline Guides (`cookbook/*/README.md`)
- Title is H1, first paragraph is a plain description + audience callout in a `>` blockquote
- Include a **Table of Contents** with anchor links for guides longer than 3 sections
- Section headers: `## 🏗️ Step Name` — always include a relevant emoji + step number for sequential guides
- ASCII diagrams for architecture (indented code block, no language tag)
- Tables for reference data (paths, parameters, flags)
- Steps written as numbered lists with sub-bullets for details
- Shell commands in fenced ` ```bash ` blocks
- Use `> **Note:**` or `> **Warning:**` for callouts
- Horizontal rules `---` to separate major sections

### Example section header pattern
```markdown
## 🧭 Step 1 — Train the Gating Network

Brief overview sentence.

### 📐 Architecture

...detail...

### ⚙️ Training Command

```bash
sbatch ...
```
```

### Emoji palette used in this project
| Emoji | Usage |
|---|---|
| 🚀 | Features, getting started |
| 🏗️ | Architecture, setup |
| ⚙️ | Config, training commands |
| 🧭 | Steps, navigation |
| 📐 | Architecture details |
| 📊 | Results, evaluation tables |
| 🧠 | Model / ML concepts |
| 🔗 | Links, connections |
| 💬 | Inference, chat |

---

## Procedure

### 1. Identify what to document
- **New Python class/method** → add Google-style docstrings in place
- **New cookbook step** → append to existing `cookbook/*/README.md` or create new `README.md`
- **New script** → add usage section to `scripts/README.md`
- **New eval task** → document in the relevant cookbook section

### 2. Gather context before writing
- Read the function signature and body to understand what it does
- Check if a parent class has docstrings — extend them, don't contradict
- For pipeline guides, run the command yourself (or check logs) to document real outputs

### 3. Write to match existing style
- Match heading depth of surrounding content
- Use present tense for class/method descriptions ("Freezes model parameters…" not "This will freeze…")
- Use active voice ("Converts the config…" not "The config is converted…")
- For pipeline docs: always include the exact command with real paths, not placeholder `<values>`

### 4. Quality check
- [ ] Every public class has a one-line docstring
- [ ] Every `__init__` with >3 params has `Args:` section
- [ ] Every method with a non-obvious return has `Returns:`
- [ ] Code blocks have language tags
- [ ] Real paths used in shell examples (not `<placeholder>`)
- [ ] No secrets (HF tokens, API keys) in documentation
