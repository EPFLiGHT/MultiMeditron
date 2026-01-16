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
| MultiMeditron LLaMA3.1-8B ATTN-PEP          | 29.6 | 59.1        | 1.5              | 30.3            | 51.1      | 27.6          | 29.6          |
| MultiMeditron LLaMA3.1-8B ATTN-SHARED       | 28.6 | 56.9        | 2.0              | 29.5            | 46.0      | 25.8          | 27.5          |
| MultiMeditron LLaMA3.1-8B AVG-PEP           | 30.7 | 46.5        | 2.5              | 24.5            | 47.6      | 25.8          | 27.6          |
| MultiMeditron LLaMA3.1-8B AVG-SHARED        | 29.7 | 46.8        | 2.6              | 24.2            | 49.5      | 23.7          | 25.8          |
| Random                                      | 25.7 | 50.0        | –                | –               | 50.0      | –             | –             |


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

# Number of dataset processes for the huggingface library
export NUM_PROC=64

# WandB
export WANDB_API_KEY="<wandb_token>" # Optional if you don't want to log to WandB
export WANDB_MODE="online" # Set to "offline" if you don't want to log to the remote WandB server
export WANDB_DIR=$SCRATCH/multimeditron/wandb

# Multi node training configuration
export NNODES=4
export NUM_PROC=4 # 4 GPUs per node (adapt if needed accordingly)
```

Make sure to replace the `$HF_TOKEN` and `$WANDB_API_KEY` by your actual tokens.

In your terminal, run:

```bash
source .env
```

### Download data

The data is available on huggingface at [OpenMeditron/MultiMediset](https://huggingface.co/datasets/OpenMeditron/MultiMediset). You can download the data by running:

```py
from datasets import load_dataset
import os

STORAGE_ROOT = os.environ["STORAGE_ROOT"]
NUM_PROC = os.environ["NUM_PROC"]

dataset_name = "OpenMeditron/MultiMediset"

ds_dict = load_dataset(dataset_name, num_proc=NUM_PROC)

for split_name, split_dataset in ds_dict.items():
    split_dir = os.path.join(STORAGE_ROOT, split_name)
    split_dataset.save_to_disk(split_dir)
```

Your data is stored in `$STORAGE_ROOT`

### Launching a training

We provide an example to reproduce MultiMeditron Qwen3-4B BiomedCLIP.

Download the configurations:

```bash
mkdir config
curl https://raw.githubusercontent.com/EPFLiGHT/MultiMeditron/refs/heads/master/cookbook/deepspeed.json -o config/deepspeed.json
envsubst <<< "$(curl https://raw.githubusercontent.com/EPFLiGHT/MultiMeditron/refs/heads/master/cookbook/sft/single_clip/qwen_biomedclip/stage1_alignment.yaml)" > config/config_alignment.yaml
envsubst <<< "$(curl https://raw.githubusercontent.com/EPFLiGHT/MultiMeditron/refs/heads/master/cookbook/sft/single_clip/qwen_biomedclip/stage2_end2end.yaml)" > config/config_end2end.yaml
```

Each configuration file can be used to train the corresponding model. The training process consists of two stages:

1. **Stage 1 - Alignment**: Aligns the vision encoder with the language model
2. **Stage 2 - End-to-End**: Fine-tunes the entire multimodal model

#### Single node training

Example usage (single node):

```bash
# Train single CLIP model
torchrun --nproc-per-node $GPUS_PER_NODE -m multimeditron train --config config/config_alignment.yaml
torchrun --nproc-per-node $GPUS_PER_NODE -m multimeditron train --config config/config_end2end.yaml
```

#### Multi-node training (CSCS)

1. Connect to the login node

```bash
ssh clariden
```

2. Download the sbatch script

```bash
curl https://raw.githubusercontent.com/EPFLiGHT/MultiMeditron/refs/heads/master/cookbook/training_template.sh -o training_template.sh
```

3. Substitute the environment variable. Make sure that you have created the environment as described [here](#setup_environment)

```bash
envsubst < training_template.sh > training.sh
```

4. Run the script

```bash
# For alignment
sbatch training.sh config/config_alignment.yaml

# For end2end
sbatch training.sh config/config_end2end.yaml
```

### Evaluation

To evaluate MultiMeditron, we use the [EPFLiGHT/lmms-eval](https://github.com/EPFLiGHT/lmms-eval) pipeline.

If you don't use the MultiMeditron Docker image, you need to install the `lmms-eval` pipeline using the following command. If you use, the MultiMeditron Docker image, you can skip this step

```bash
git clone https://github.com/EPFLiGHT/lmms-eval.git
cd lmms-eval
pip install -e .
```

To evaluate a trained model, run:

```bash
python3 -m accelerate.commands.launch \
    --num_processes $NUM_PROC \
    -m lmms_eval \
    --model multimeditron \
    --model_args pretrained="$CHECKPOINT",tokenizer_type="$TOKENIZER_TYPE",device_map="auto" \
    --tasks gmai,slake,path_vqa \
    --batch_size 1 \
```

Replace the `$NUM_PROC` by the the number of GPUs on your node, the `$CHECKPOINT` variable by your model checkpoint path, and the `$TOKENIZER_TYPE` by the tokenizer you used for training the multimodal model.

You can get the `$TOKENIZER_TYPE` by looking at the configuration file:

```bash
cat config/config_alignment.yaml
```

And check the line

```
tokenizer_type: qwen3  # (or apertus, llama depending on your setup)
```

The available tokenizer types are:

- `qwen3`
- `apertus`
- `llama`
