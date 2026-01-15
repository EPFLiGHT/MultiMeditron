# MultiMeditron Cookbook

This cookbook contains configuration files and training recipes for the MultiMeditron suite of multimodal medical AI models.

## 📁 Cookbook Structure

The cookbook is organized into two main categories:

### `sft/single_clip/`
Contains configurations for single vision encoder models:
- **`qwen_biomedclip/`** - Qwen3-4B with BiomedCLIP
- **`apertus_biomedclip/`** - Apertus-8B with BiomedCLIP  
- **`llama_biomedclip/`** - LLaMA3.1-8B with BiomedCLIP
- **`llama_clip/`** - LLaMA3.1-8B with standard CLIP

Each model directory contains:
- `stage1_alignment.yaml` - First stage alignment training
- `stage2_end2end.yaml` - End-to-end fine-tuning

### `sft/moe/`
Contains configurations for Mixture of Experts (MoE) models with different fusion strategies:

#### Fusion Methods:
- **`attn/`** - Cross-attention fusion
- **`avg/`** - Average fusion  
- **`cat/`** - Concatenation fusion

#### Expert Configurations:
- **`pep/`** - Per-expert projection
- **`shared/`** - Shared projection

Each MoE configuration contains both alignment and end-to-end training stages.

## 🧪 Experiment Mapping

| Experiment Name | Base LLM | Vision Encoder | Cookbook Path |
|-----------------|-----------|----------------|---------------|
| MultiMeditron Qwen3-4B BiomedCLIP | Qwen3-4B | BiomedCLIP | `sft/single_clip/qwen_biomedclip/` |
| MultiMeditron Apertus-8B BiomedCLIP | Apertus-8B | BiomedCLIP | `sft/single_clip/apertus_biomedclip/` |
| MultiMeditron LLaMA3.1-8B BiomedCLIP | LLaMA3.1-8B | BiomedCLIP | `sft/single_clip/llama_biomedclip/` |
| MultiMeditron LLaMA3.1-8B CLIP | LLaMA3.1-8B | CLIP | `sft/single_clip/llama_clip/` |
| MultiMeditron LLaMA3.1-8B ATTN-PEP | LLaMA3.1-8B | MultiMeditron ATTN-PEP | `sft/moe/attn/pep/` |
| MultiMeditron LLaMA3.1-8B ATTN-SHARED | LLaMA3.1-8B | MultiMeditron ATTN-SHARED | `sft/moe/attn/shared/` |
| MultiMeditron LLaMA3.1-8B AVG-PEP | LLaMA3.1-8B | MultiMeditron AVG-PEP | `sft/moe/avg/pep/` |
| MultiMeditron LLaMA3.1-8B AVG-SHARED | LLaMA3.1-8B | MultiMeditron AVG-SHARED | `sft/moe/avg/shared/` |

## 📊 Model Evaluation

::: {#tab:end2end_eval}
  --------------------------------------- ---------- ------------- ---------- ---------- ----------- ---------- ----------
  **Model name**                           **GMAI**   **PathVQA**                         **SLAKE**             
                                                          y/n       open-end   overall       y/n      open-end   overall
  **Open weights**                                                                                              
  MultiMeditron Qwen3-4B BiomedCLIP          35.3        57.4         2.4        29.9       55.6        27.7       30.1
  MultiMeditron Apertus-8B BiomedCLIP        34.2        57.4         1.2        29.9       51.3         21        23.6
  MultiMeditron LLaMA3.1-8B BiomedCLIP       36.6        55.7         3.4        29.5       48.1        22.4       24.5
  MultiMeditron LLaMA3.1-8B CLIP              34         60.6         5.6        33.1       50.5        28.5       30.3
  MultiMeditron LLaMA3.1-8B ATTN-PEP         29.6        59.1         1.5        30.3       51.1        27.6       29.6
  MultiMeditron LLaMA3.1-8B ATTN-SHARED      28.6        56.9          2         29.5        46         25.8       27.5
  MultiMeditron LLaMA3.1-8B AVG-PEP          30.7        46.5         2.5        24.5       47.6        25.8       27.6
  MultiMeditron LLaMA3.1-8B AVG-SHARED      29.7        46.8         2.6        24.2       49.5        23.7       25.8
  Random                                    25.70         50           \-         \-         50          \-         \-
  --------------------------------------- ---------- ------------- ---------- ---------- ----------- ---------- ----------

  : Evaluation of MultiMeditron after the end-to-end phase and other
  multimodal models on medical image--text benchmarks.
:::

## 🚀 Usage

Each configuration file can be used to train the corresponding model. The training process consists of two stages:

1. **Stage 1 - Alignment**: Aligns the vision encoder with the language model
2. **Stage 2 - End-to-End**: Fine-tunes the entire multimodal model

Example usage (single node):
```bash
# Train single CLIP model
torchrun --nproc-per-node $GPUS_PER_NODE -m multimeditron train --config sft/single_clip/qwen_biomedclip/stage1_alignment.yaml
torchrun --nproc-per-node $GPUS_PER_NODE -m multimeditron train --config sft/single_clip/qwen_biomedclip/stage2_end2end.yaml
```

For a multi-node setup, please refer to 



