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


| Model name                                   | GMAI | PathVQA y/n | PathVQA open-end | PathVQA overall | SLAKE y/n | SLAKE open-end | SLAKE overall |
|---------------------------------------------|------|-------------|------------------|-----------------|-----------|---------------|---------------|
| **Open weights**                            |      |             |                  |                 |           |               |               |
| MultiMeditron Qwen3-4B BiomedCLIP           | 35.3 | 57.4        | 2.4              | 29.9            | 55.6      | 27.7          | 30.1          |
| MultiMeditron Apertus-8B BiomedCLIP         | 34.2 | 57.4        | 1.2              | 29.9            | 51.3      | 21.0          | 23.6          |
| MultiMeditron LLaMA3.1-8B BiomedCLIP        | 36.6 | 55.7        | 3.4              | 29.5            | 48.1      | 22.4          | 24.5          |
| MultiMeditron LLaMA3.1-8B CLIP              | 34.0 | 60.6        | 5.6              | 33.1            | 50.5      | 28.5          | 30.3          |
| MultiMeditron LLaMA3.1-8B ATTN-PEP          | 29.6 | 59.1        | 1.5              | 30.3            | 51.1      | 27.6          | 29.6          || **MultiMeditron LLaMA3.1-8B ATTN-PEP (7-exp)** | **31.1** | **47.1** | –        | **24.4**¹       | **51.1**  | –             | **30.6**      || MultiMeditron LLaMA3.1-8B ATTN-SHARED       | 28.6 | 56.9        | 2.0              | 29.5            | 46.0      | 25.8          | 27.5          |
| MultiMeditron LLaMA3.1-8B AVG-PEP           | 30.7 | 46.5        | 2.5              | 24.5            | 47.6      | 25.8          | 27.6          |
| MultiMeditron LLaMA3.1-8B AVG-SHARED        | 29.7 | 46.8        | 2.6              | 24.2            | 49.5      | 23.7          | 25.8          |
| Random                                      | 25.7 | 50.0        | –                | –               | 50.0      | –             | –             |

> **Note:** The cookbook numbers above were produced from the **final checkpoint** of each model's Stage 2 training run (see path registry below).
>
> ¹ PathVQA overall for 7-expert regressed vs. the 5-expert. The yes/no regression (59.1 → 47.1%) is under investigation. GMAI and SLAKE improved with the addition of Ophthalmology and Skin experts.

## 📍 Model Path Registry (CSCS Capstor)

All published model checkpoints live on **capstor** at:
```
/capstor/store/cscs/swissai/a127/homes/meditron/models/multimeditron/
```

### End-to-end trained models (`unfreeze/`)

| Cookbook Name | Capstor Path (relative to root) | Final Ckpt | Last Trained | Author |
|---|---|---|---|---|
| ATTN-PEP (5-expert) | `unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end/` | **3063** | 2026-01-01 | mzhang |
| ATTN-SHARED | `unfreeze/attn_shared/MultiMeditron-8B-attn-shared-end2end/` | **1532** | 2025-12-22 | theoschiff |
| AVG-PEP | `unfreeze/avg_pep/MultiMeditron-8B-avg-pep-end2end/` | **1532** | 2025-12-22 | theoschiff |
| AVG-SHARED | `unfreeze/avg_shared/MultiMeditron-8B-avg-shared-end2end/` | **1532** | 2025-12-22 | theoschiff |
| CAT-PEP | `unfreeze/cat_pep/MultiMeditron-8B-cat-pep-end2end/` | **1532** | 2026-01-07 | theoschiff |
| CAT-SHARED | `unfreeze/cat_shared/MultiMeditron-8B-cat-shared-end2end/` | **1532** | 2026-01-07 | theoschiff |
| Qwen3-4B BiomedCLIP | `unfreeze/single_clip/MultiMeditron-Qwen-4B-End2End-BiomedCLIP/` | **3063** | 2026-01-27 | mzhang |
| Apertus-8B BiomedCLIP | `unfreeze/single_clip/MultiMeditron-Apertus-8B-End2End-BiomedCLIP/` | **3063** | 2025-12-30 | mzhang |
| LLaMA3.1-8B BiomedCLIP | `unfreeze/single_clip/MultiMeditron-Llama-8B-End2End-BiomedCLIP/` | **3063** | 2026-01-03 | mzhang |
| LLaMA3.1-8B CLIP | `unfreeze/single_clip/MultiMeditron-Llama-8B-End2End-CLIP/` | **3063** | 2026-01-26 | mzhang |

> `unfreeze/avg_pep/MultiMeditron-8B-avg-pep-full/` exists but is empty.
>
> None of these models have been published to HuggingFace Hub yet. The HF dataset is at [OpenMeditron/MultiMediset](https://huggingface.co/datasets/OpenMeditron/MultiMediset).

### 7-expert ATTN-PEP (in-progress, iopsstor)

| Model | Path | Latest Ckpt | Last Trained | Author |
|---|---|---|---|---|
| ATTN-PEP 7-expert | `/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end-7exp/` | **800** | 2026-03-24 | surech |

### Alignment models (`freeze/`)

Alignment-stage (Stage 1) checkpoints are stored under `freeze/` with the same variant naming:
`attn_pep`, `attn_shared`, `avg_pep`, `avg_shared`, `cat_pep`, `cat_shared`, `single_clip`,
plus MoE-specific variants: `moe_avg_pep`, `moe_avg_shared`, `moe_cat_pep`, `moe_cat_shared`.

For the full dataset and model reference (including paths, descriptions, and data formats), see [cookbook/REGISTRY.md](REGISTRY.md).

## 📈 7-Expert vs 5-Expert Comparison

Comparison of the new 7-expert ATTN-PEP model (checkpoint-800) against the published 5-expert cookbook baseline:

| Benchmark | 5-expert ATTN-PEP (cookbook) | 7-expert checkpoint-800 | Δ |
|---|---|---|---|
| **GMAI** | 29.6% | 31.1% | **+1.5** |
| **SLAKE overall** | 29.6% | 30.6% | **+1.0** |
| SLAKE yes/no | 51.1% | 51.1% | 0.0 |
| **PathVQA overall** | 30.3% | 24.4% | **−5.9** |
| PathVQA yes/no | 59.1% | 47.1% | **−12.0** |
| GMAI Dermatology | – | 39.5% | *new* |
| GMAI Ophthalmology | – | 31.8% | *new* |

> **Key finding:** GMAI and SLAKE improved, but PathVQA (especially binary yes/no) regressed significantly. See `reports/EVAL_ANALYSIS.md` for the full per-modality breakdown and PathVQA failure investigation.

## 🚀 Usage

### Prerequisites

#### On the CSCS cluster

1. Connect to the CSCS

```bash
ssh clariden
```

2. Download the EDF file in your `$HOME`:
```bash
curl https://raw.githubusercontent.com/EPFLiGHT/MultiMeditron/refs/heads/master/cookbook/assets/edf.toml -o ~/.edf/multimeditron.toml
```

3. Claim a job using srun. Make sure to replace the `<ACCOUNT>` by your actual CSCS account (Hint: your account should have the form `axxx` where `xxx` is some number)

```bash
srun --time=1:29:59 --partition debug -A <ACCOUNT> --environment=~/.edf/multimeditron.toml --pty bash
```


#### Other clusters

To run a training you need access to NVIDIA GPUs. If needed, make sure to claim a job to get access to GPUs and run the next steps inside the following Docker images for the dependencies:

```bash
michelducartier24/multimeditron-git:latest-amd64 # For AMD64 architecture
michelducartier24/multimeditron-git:latest-arm64 # For ARM64 architecture
```

Alternatively, you can also install multimeditron directly with pip:
```bash
git clone https://github.com/EPFLiGHT/MultiMeditron.git
cd MultiMeditron
pip install -e ".[flash-attn]"
```


### Setup environment

Create a `.env` file. We provide an example below for researchers working on the CSCS cluster:

```bash
export WORKING_DIR=$(pwd)

# Path to store the datasets
export STORAGE_ROOT=$STORE/meditron/multimediset/arrow

# Path to store the models
export MODEL_ROOT=$SCRATCH/multimeditron/checkpoints

# Huggingface 
export HF_TOKEN="<hf_token>"
export HF_HOME=$SCRATCH/hf

# Number of processes to use for dataset preprocessing
export DS_NUM_PROC=64

# WandB
export WANDB_API_KEY="<wandb_token>" # Optional if you don't want to log to WandB
export WANDB_MODE="online" # Set to "offline" if you don't want to log to the remote WandB server
export WANDB_DIR=$SCRATCH/multimeditron/wandb

# Multi node training configuration
export NNODES=4
export NUM_PROC=4 # 4 GPUs per node (adapt if needed accordingly)

export ACCOUNT=a127
```

Make sure to replace the `$HF_TOKEN` and `$WANDB_API_KEY` by your actual tokens.

In your terminal, run:

```bash
source .env
```

### Download data (optional if you are part of the LiGHT organization in the CSCS)

The data is available on huggingface at [OpenMeditron/MultiMediset](https://huggingface.co/datasets/OpenMeditron/MultiMediset). You can download the data by running:

```py
from datasets import load_dataset, get_dataset_config_names
import os

STORAGE_ROOT = os.environ["STORAGE_ROOT"]
DS_NUM_PROC = int(os.environ["DS_NUM_PROC"])

dataset_name = "OpenMeditron/MultiMediset"
configs = get_dataset_config_names(dataset_name)

for split_name in configs:
    split_dir = os.path.join(STORAGE_ROOT, split_name)
    split_dataset = load_dataset(dataset_name, split_name, num_proc=DS_NUM_PROC)
    split_dataset.save_to_disk(split_dir)
```

Your data is stored in `$STORAGE_ROOT`

### Launching a training

We provide an example to reproduce MultiMeditron Qwen3-4B BiomedCLIP.

Download the configurations:

```bash
mkdir config
curl https://raw.githubusercontent.com/EPFLiGHT/MultiMeditron/refs/heads/master/cookbook/deepspeed.json -o config/deepspeed.json
curl https://raw.githubusercontent.com/EPFLiGHT/MultiMeditron/refs/heads/master/cookbook/sft/single_clip/qwen_biomedclip/stage1_alignment.yaml -o config/config_alignment.yaml
curl https://raw.githubusercontent.com/EPFLiGHT/MultiMeditron/refs/heads/master/cookbook/sft/single_clip/qwen_biomedclip/stage2_end2end.yaml -o config/config_end2end.yaml
```

> **Note:** Do **not** pipe YAML configs through `envsubst` on this cluster — it will replace all `$SLURM_…` variables with empty strings.

Each configuration file can be used to train the corresponding model. The training process consists of two stages:

1. **Stage 1 - Alignment**: Aligns the vision encoder with the language model
2. **Stage 2 - End-to-End**: Fine-tunes the entire multimodal model

#### Single node training

Example usage (single node):

```bash
# Train single CLIP model. We set number of processes to 4 because there is 4 GPUs per node on the clariden cluster, modify as needed.
torchrun --nproc-per-node ${NUM_PROC:-4} -m multimeditron train --config config/config_alignment.yaml
torchrun --nproc-per-node ${NUM_PROC:-4} -m multimeditron train --config config/config_end2end.yaml
```

#### Multi-node training (CSCS)

1. Connect to the login node

```bash
ssh clariden
```

If necessary create and to the `source .env` again.

2. Download the sbatch script

```bash
curl https://raw.githubusercontent.com/EPFLiGHT/MultiMeditron/refs/heads/master/cookbook/training_template.sh -o training_template.sh
```

3. Run the training with `sbatch_train.sh`, passing the config YAML as the first argument:

```bash
# Stage 1 — alignment (freeze vision encoders)
sbatch sbatch_train.sh cookbook/sft/moe/attn/pep/stage1_alignment.yaml

# Stage 2 — end-to-end (unfreeze everything)
sbatch sbatch_train.sh cookbook/sft/moe/attn/pep/stage2_end2end.yaml
```

> **Note:** Do **not** pipe the config through `envsubst` on this cluster — the shell escaping for SLURM `$$` variables is not supported and will silently replace all SLURM variables with empty strings.

4. To check the logs of the run:

```bash
cd ~/meditron/reports/
tail -f R-multimeditron-train.<jobid>.out
```

Replace `<jobid>` with the SLURM job ID printed by `sbatch`. You can list all logs with `ls -lt ~/meditron/reports/`.


### Training the gating network

The gating network is a ResNet50 classifier that routes each input image to the most appropriate CLIP expert.  On CSCS it is trained separately (no container needed) using image datasets organised as `ImageFolder`-compatible directory trees.

#### 1. Prepare data

Create one sub-folder per expert class **inside** `train/` and `test/`:

```
data/
  train/
    CT/           ← CT scan images
    MRI/
    Ultrasound/
    X-ray/
    Ophthalmology/
    Skin/
    Generalist/
  test/
    CT/
    ...           ← same structure
```

Image sources used for the 7-class model (see `config/gating_7class.yaml`):
- CT → `image_ct2` (training split)  
- X-ray → `image_iu_xray`  
- Ultrasound → `image_BUSI`, `image_COVID_US`, `image_DDTI`  
- Ophthalmology → `eye_dataset/train`  
- Skin → `skin_dataset/train`

#### 2. Train

```bash
python3 scripts/image_router_train.py \
  --data_dir data/ \
  --resnet_size 50 \
  --batch_size 32 \
  --num_epochs 20 \
  --lr 0.0001 \
  --output_dir models/CLIP/MultiMeditron-Gating
```

#### 3. Wrap into a HuggingFace checkpoint

After training, convert the raw `.pt` weights into a `GatingNetwork` HF checkpoint so it can be loaded with `GatingNetwork.from_pretrained()`:

```python
from multimeditron.model.modalities.moe.gating import GatingNetwork, GatingNetworkConfig
import torch

config = GatingNetworkConfig(
    num_classes=7,
    top_k=1,
    image_processor_path="openai/clip-vit-base-patch32",
    class_names=["CT", "Generalist", "MRI", "Ultrasound", "X-ray", "Ophthalmology", "Skin"],
)
model = GatingNetwork(config, resnet_path="output/resnet50_weights.pt")
model.save_pretrained("models/CLIP/MultiMeditron-Gating")
```

#### 4. Verify routing

```bash
sbatch sbatch_gating_analysis.sh
```

Expected: each modality dataset is routed to its dedicated expert with ≥ 94% top-1 accuracy (see `scripts/gating_routing_analysis.py` for held-out evaluation sets).


### Evaluation

To evaluate MultiMeditron, we use the [EPFLiGHT/lmms-eval](https://github.com/EPFLiGHT/lmms-eval) pipeline.

The recommended way to run evaluation on the CSCS cluster is via the `sbatch_eval.sh` launcher, which handles multi-node accelerate setup, container environment, and output paths automatically:

```bash
export HF_TOKEN=<your_hf_token>

# Standard 3-benchmark eval (16 nodes, ~50 min)
sbatch --time 03:00:00 --nodes 16 sbatch_eval.sh \
  <checkpoint_path> llama gmai,slake,path_vqa

# Per-modality GMAI breakdown (10 subtasks)
sbatch --time 03:00:00 --nodes 4 sbatch_eval.sh \
  <checkpoint_path> llama \
  gmai_ct,gmai_mri,gmai_xray,gmai_ultrasound,gmai_endoscopy,gmai_histopathology,gmai_fundus,gmai_microscopy,gmai_dermoscopy,gmai_oct

# New expert subtasks (ophthalmology + dermatology)
sbatch --time 03:00:00 --nodes 4 sbatch_eval.sh \
  <checkpoint_path> llama gmai_ophthalmology,gmai_dermatology

# Per-modality SLAKE breakdown
sbatch --time 03:00:00 --nodes 4 sbatch_eval.sh \
  <checkpoint_path> llama slake_ct,slake_mri,slake_xray

# Save per-sample predictions (for error analysis)
sbatch --time 03:00:00 --nodes 16 sbatch_eval.sh \
  <checkpoint_path> llama path_vqa "" true
```

Results are saved to `~/meditron/reports/lmms_eval_results/<checkpoint_name>/`.

> **Tokenizer type:** Use `llama` for all LLaMA3.1-8B models on this branch.
> For Qwen3 or Apertus models use `qwen3` / `apertus` respectively.

> **Note:** `sbatch_eval_vllm.sh` is **not supported** — vLLM cannot load our custom `multimodal` model type. Always use `sbatch_eval.sh`.

#### Timing reference

| Nodes | Wall time (3 benchmarks) |
|---|---|
| 4 | ~3.5 h |
| 16 | ~50 min |

#### Gating network analysis

To verify that the gating network routes each modality to the correct expert (uses held-out splits, no GPU required):

```bash
sbatch sbatch_gating_analysis.sh
```

See `scripts/gating_routing_analysis.py` for dataset paths and methodology.

