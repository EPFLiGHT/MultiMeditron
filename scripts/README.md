# How to train the router ?
The router's goal is to redirect a specific modality to the right expert model.
## 1. Format the data
The categorized data should be put in the folders `data/train/{modality}/{category}` and `data/test/{modality}/{category}`.
## 2. Train the router
Run the following command:
```bash
python3 image_router_train.py --data_dir=data/images --resnet_size=50 --batch_size=32 --num_epochs=10 --lr=0.001 --output_dir=output
```

All CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--resnet_size` | 50 | ResNet variant: 18, 34, or 50 |
| `--max_samples_per_class` | 1000 | Cap on samples per class for class-balanced training |
| `--lr` | 0.0001 | Learning rate |
| `--batch_size` | 16 | Batch size |
| `--data_dir` | `data/images/` | Root directory; must contain `train/` and `test/` subdirs with one folder per class |
| `--output_dir` | `output` | Where to save the trained model weights |
| `--num_epochs` | 20 | Number of training epochs |

# How to train the expert model?
The expert model's goal is to extract features from the data. For images, we use a CLIP model fine-tuned on captionized images.

## 1. Format the data
Update the ImageTextDataset class to properly format the data into encoded images and text.

## 2. Train the expert model
Run the following command:
```bash
python3 expert_model_train.py --data_url=<your huggingface dataset> --batch_size=32 --num_epochs=10
```

# `biomed_train.py`

## 1. What is it for?

Fine-tuning BiomedCLIP with datasets formatted according to the standard we agreed on.

BiomedCLIP is available on HuggingFace Hub, but the code of `expert_model_train.py` does not work for it due to the model inputs and outputs being different with BiomedCLIP. 

`expert_model_train.py` is also not fit for reading a dataset consisting of a `.jsonl` file containing examples with text and modalities. The ImageTextDataset class has been adapted in `biomed_train.py`, so that it can take the `.jsonl` file and, assuming the links to modalities are relative to the `.jsonl` file, the script is able to properly load the data and fine-tune the model with it.

## 2. How to use?

`python3 biomed_train.py --data_url chexpert/chexpert.jsonl --output_dir chexpert_test --num_epochs 20 --save_model True`

Use the command `python3 biomed_train.py -h` for additional help.

# `gating_routing_analysis.py`

## 1. What is it for?

Analyses routing behaviour of both the 5-expert and 7-expert gating networks.
For each of 5 modality-pure held-out datasets it reports:
- Top-1 routed expert and percentage of images assigned to it
- Average softmax weight per expert

Use this to verify that each modality is routed to the correct specialist expert
after a gating retrain, or to diagnose CT/US confusion bugs (see April 2026 audit).

## 2. How to use?

Run directly on the login node (no GPU required — ResNet50 on CPU):

```bash
python3 scripts/gating_routing_analysis.py
```

Or submit via the dedicated SLURM script (debug partition, ~1 min):

```bash
sbatch sbatch_gating_analysis.sh
```

Results are printed to stdout and saved to `/users/surech/meditron/reports/R-gating-routing-analysis.<jobid>.out`.

---

# `compare_modality_results.py`

## 1. What is it for?

Produces a side-by-side Markdown table comparing per-modality GMAI accuracy between the 5-expert baseline (checkpoint-3063) and the 7-expert model (checkpoint-800).
Also shows GMAI department subtasks (ophthalmology, dermatology) and sample counts.

## 2. How to use?

```bash
python3 scripts/compare_modality_results.py
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--results-root` | `/users/surech/meditron/reports/lmms_eval_results` | Root directory of lmms-eval output directories |
| `--model-a` | auto-discovered | Explicit path to model A result directory |
| `--model-b` | auto-discovered | Explicit path to model B result directory |

---

# `prep_image_datasets.py`

## 1. What is it for?

This script can be used to easily download datasets from the MultiMediset to the desired folder. 

The script downloads the jsonl file and the corresponding ZIP / parquet archives, and unzips the archives.

## 2. How to use?

First, open the script in a code editor to specify several things:
- **`dataset_folders` is a dictionary in which you specify which datasets from the MultiMediset you want to download**. The keys are the name of the local folder in which you want to download datasets, and the values are lists of datasets to download in the folder corresponding to the key. This allows you to arrange datasets as you wish. You can write a path instead of a folder name, if you need a more specific hierarcy.
- **`path_datasets` is a string, the local path in which you want to download the datasets**. Then for instance, if you follow the example of the default configuration of the script, the DDTI dataset will be downloaded at `../datasets/US/DDTI`.
- **`path_to_dataset_repo` is a dictionary that defines for each of the datasets the path to their parent folder on the repository**. Please do not put a / at the end of the str.

Once you have done it, you can simply run the script and let it download your datasets as you configured it. You may enable `hf_transfer` to accelerate the download by defining the corresponding environment variable.

# `train_clip.py` and `config_us.yaml`

## 1. What is it for?

This script can be used to fine-tune a CLIP-based model with datasets. With a proper configuration file, it can prepare a dataset or a mixture of datasets, train a CLIP-based model and save the result at a given local path.

The configuration file looks like this:

```yaml
output_dir: "../training_clip/clip-splade-US-train"
vision_model_name: "openai/clip-vit-base-patch32"
text_model_name: "naver/splade-v3"
dataset_configs:
  - BUSI:
      dataset_name: "/mloscratch/homes/nemo/training/US/BUSI/BUSI-train.jsonl"
      image_column: "modalities"
      caption_column: "text"
      weight: 78
  - CAMUS:
      dataset_name: "/mloscratch/homes/nemo/training/US/CAMUS/CAMUS-train.jsonl"
      image_column: "modalities"
      caption_column: "text"
      weight: 2118

remove_unused_columns: false
do_train: true
per_device_train_batch_size: 64
learning_rate: 5.0e-4
warmup_steps: 2000
weight_decay: 0.2
save_steps: 0.1
num_train_epochs: 3
dataloader_drop_last: true
overwrite_output_dir: true
```

`output_dir` is a relative or absolute path to which the fine-tuned model has to be saved. `vision_model_name` is strongly recommended to be kept at `openai/clip-vit-base-patch32`, as differing models may have different inputs. This script is made specifically for this model. The same applies to `text_model_name`.

For `dataset_configs`, write the details of each dataset you want to include in the training. The `weight` parameter has to be defined relative to other datasets, so that each dataset has a total weight of `weight/(sum of the weights of all datasets)` in the mixture. Consider that this parameter does not care about the number of examples in each dataset. The mixture is made of randomly drawn examples, according to the distribution defined by the weights and the draw is stopped whenever one of the dataset runs out of examples.

`do_train: true` can be replaced by `do_eval: true` to switch to the evaluation mode.

The remaining parameters are standard parameters for ML.

## 2. How to use?

Define the configuration in the configuration file, then run it with `python3 train_clip.py config_us.yaml` (you may replace `config_us.yaml` with another configuration file that has the same parameters).