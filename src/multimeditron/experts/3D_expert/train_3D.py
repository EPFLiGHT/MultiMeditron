#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training a CLIP like dual encoder models using text and vision encoders in the library.

The script can be used to train CLIP like models for languages other than English by using
a text encoder pre-trained in the desired language.
"""

import logging
import os
import numpy as np
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist
from datasets import load_dataset, interleave_datasets

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    VisionTextDualEncoderModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#from transformers.utils import check_min_version
#check_min_version("4.47.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")
#torch.multiprocessing.set_start_method('spawn')

# Definition of config arguments
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    text_model_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Text encoder model name/path (e.g. FacebookAI/roberta-base)"}
    )

@dataclass
class DatasetConfig:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="modalities",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    weight: Optional[float] = field(
        default=1.0, metadata={"help": "The weight to assign to this dataset during training."}
        )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    dataset_configs: Optional[List[DatasetConfig]] = field(
        default=None, metadata={"help": "Dataset configuration for training and evaluation."}
    )

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

def collate_fn(examples):
    """
    Stack the examples into a format fit for training.
    """
    
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "images": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.arange(len(examples) * torch.cuda.device_count(), dtype=torch.long),
        "return_loss": True,
    }

def compute_metrics(eval_pred):
    """
    Compute retrieval accuracy metrics for CLIP-style models.
    Note: Since evaluation happens in batches and logits are concatenated,
    we compute per-sample accuracy based on the diagonal being the correct match.
    """
    logits, labels = eval_pred
    
    # logits shape after concatenation: (num_samples, batch_size) or similar
    # labels shape: (num_samples,)
    
    # For each sample, check if the model predicted the correct index
    # In contrastive learning, the correct match is at the diagonal position
    predictions = np.argmax(logits, axis=1)
    
    # The labels contain the correct indices (typically 0, 1, 2, ... for each batch)
    accuracy = np.mean(predictions == labels)
    
    # Compute top-5 accuracy if we have enough classes
    top5_acc = 0.0
    if logits.shape[1] >= 5:
        top5_preds = np.argsort(logits, axis=1)[:, -5:]
        top5_acc = np.mean([labels[i] in top5_preds[i] for i in range(len(labels))])
    
    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_acc,
    }

def expand_to_image_pairs(dataset, config):
    """
    Expand dataset so each .npy file in a folder gets paired with the text.
    If a folder contains multiple .npy files, create one row per .npy file,
    all sharing the same text description.
    """
    expanded_rows = []
    base_dir = os.path.dirname(config.dataset_name)
    
    for row in dataset:
        text = row[config.caption_column].replace("<attachment>", "")
        folder_path = os.path.join(base_dir, row[config.image_column][0]["value"])
        
        # Find all .npy files in the folder
        if os.path.isdir(folder_path):
            npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            if npy_files:
                for npy_file in npy_files:
                    expanded_rows.append({
                        "image_path": os.path.join(folder_path, npy_file),
                        "caption": text
                    })
            else:
                logger.warning(f"No .npy files found in {folder_path}")
        elif os.path.isfile(folder_path + ".npy"):
            # Handle case where path points directly to a file (without .npy extension)
            expanded_rows.append({
                "image_path": folder_path + ".npy",
                "caption": text
            })
        elif os.path.isfile(folder_path):
            # Handle case where path points directly to a file
            expanded_rows.append({
                "image_path": folder_path,
                "caption": text
            })
        else:
            logger.warning(f"Path not found: {folder_path}")
    
    return expanded_rows

def get_combined_dataset(dataset_configs: List[DatasetConfig], model_args: ModelArguments):
    """
    Generate a random mixture of datasets based on the relative weights registered in the configuration.
    """
    from datasets import Dataset

    datasets = []
    probabilities = []
    logger.info(f"Loading datasets: {dataset_configs}")
    for config in dataset_configs:
        # Load individual dataset
        if config.get("dataset_name", None) is None:
            assert(len(config) == 1)
            config = config[list(config.keys())[0]]
        config = DatasetConfig(**config)

        if config.dataset_name.endswith(".jsonl"): #path to a jsonl
            dataset = load_dataset(
                "json",
                config.dataset_config_name,
                cache_dir=model_args.cache_dir,
                keep_in_memory=False,
                data_dir=config.data_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                data_files=config.dataset_name,
                )
        else:
            dataset = load_dataset(
                config.dataset_name,
                config.dataset_config_name,
                cache_dir=model_args.cache_dir,
                keep_in_memory=False,
                data_dir=config.data_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
        
        # For each dataset, expand to create one row per .npy file
        if "train" in dataset:
            if config.dataset_name.endswith(".jsonl"):
                # Expand dataset: create (image, text) pairs for each .npy file
                expanded_rows = expand_to_image_pairs(dataset["train"], config)
                dataset["train"] = Dataset.from_list(expanded_rows)
                logger.info(f"Expanded dataset from {len(dataset['train'])} rows to {len(expanded_rows)} (image, text) pairs")
            else:
                # Legacy path for non-jsonl datasets
                def find_img_path(row):
                    return {config.caption_column: row[config.caption_column], config.image_column: os.path.join(os.path.dirname(config.dataset_name), row[config.image_column][0]["value"])}
                dataset["train"] = dataset["train"].map(find_img_path)
                dataset["train"] = dataset["train"].rename_column(config.image_column, "image_path").rename_column(config.caption_column, "caption")
                dataset["train"] = dataset["train"].map(lambda x: {"caption": x["caption"].replace("<attachment>","")})
        
        # Repeat dataset according to epochs weight
        probabilities.append(config.weight)
        datasets.append(dataset["train"])
    
    # Normalize weights
    probabilities = np.array(probabilities)
    probabilities = probabilities / np.sum(probabilities)
    
    # Combine all datasets
    combined_dataset = interleave_datasets(datasets, probabilities=probabilities)
    return combined_dataset.train_test_split(test_size=0.2)

def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))

    # 2. Setup logging and training args
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    
    logger.info(f"Training/evaluation parameters {training_args}")

    # Training args
    training_args.dataloader_drop_last = True
    training_args.dataloader_num_workers = 4
    training_args.logging_steps = 10
    training_args.fp16 = True
    training_args.gradient_accumulation_steps = 2
    training_args.ddp_find_unused_parameters = True
    training_args.remove_unused_columns = False  # Keep pixel_values for collate_fn

    #if torch.cuda.device_count() > 1:  # Check if multiple GPUs are available
    #    dist.init_process_group(backend="nccl")  # Initializes distributed training

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    if data_args.dataset_configs is not None:
        dataset = get_combined_dataset(data_args.dataset_configs, model_args)
    else:
        raise ValueError("Please provide dataset configs")
        
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # 5. Load pretrained model, tokenizer, and image processor
    if model_args.text_model_name:
        # Dual encoder path
        logger.info(f"Loading dual encoder with vision model GoodBaiBai88/M3D-CLIP "
                f"and text model {model_args.text_model_name}")

        device = torch.device(training_args.device)
        torch.cuda.empty_cache()
        
        # Load model and disable gather_loss for single GPU training
        model = AutoModel.from_pretrained(
            "GoodBaiBai88/M3D-CLIP",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=True
        )
        
        # Disable gather_loss if not running distributed (requires init_process_group)
        if not dist.is_initialized():
            model.gather_loss = False
            logger.info("Disabled gather_loss for single GPU training")
        
        model = model.to(device=device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_args.text_model_name)
    else:
        raise ValueError("Missing text model name")
    config = model.config

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train or training_args.do_eval:
        column_names = dataset["train"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # 6. Get the column names for input/target.
    
    image_column = "image_path"
    caption_column = "caption"

    # 7. Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def transform_images(examples):
        images = []
        for image_file in examples[image_column]:
            img_npy = torch.from_numpy(np.load(image_file))
            #image = model.encode_image(img_npy.unsqueeze(0).float())[:, 0]
            images.append(img_npy)
        
        #examples["pixel_values"] = model.encode_image(torch.stack(images).to(device=device).float()).detach().cpu()
        examples["pixel_values"] = torch.stack(images)

        return examples
    
    def transform_image(example):
        image_file = example[image_column]
        img_npy = torch.from_numpy(np.load(image_file))

        #example["pixel_values"] = model.encode_image(torch.stack([img_npy]).to(device=device).float()).detach().cpu()
        example["pixel_values"] = img_npy
        return example

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.info(f"Dataset length: {len(train_dataset)}")

        train_dataset = train_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        # Transform images on the fly (lazy loading) to avoid loading all images into RAM
        train_dataset.set_transform(transform_images)

    if training_args.do_eval:
        test_dataset = dataset["test"]
        test_dataset = test_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )
        # Transform images on the fly (lazy loading) to avoid loading all images into RAM
        test_dataset.set_transform(transform_images)

    # 8. Initialize our trainer
    print("Report to:", training_args.report_to)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        except RuntimeError as e:
            print("error", e)
            train_result = None
        
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        if train_result:
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. Write Training Stats and push to hub.
    finetuned_from = model_args.model_name_or_path
    # If from a local directory, don't set `finetuned_from` as this is required to be a valid repo. id on the Hub.

    if finetuned_from is None or os.path.isdir(finetuned_from):
        finetuned_from = None
    kwargs = {"finetuned_from": finetuned_from, "tasks": "contrastive-image-text-modeling"}
    for dataset in data_args.dataset_configs:
        if dataset.get("dataset_name", None) is None:
            assert(len(dataset) == 1)
            dataset = dataset[list(dataset.keys())[0]]
        dataset = DatasetConfig(**dataset)
        if dataset.dataset_name is not None:
            if not hasattr(kwargs, "dataset_tags"):
                kwargs["dataset_tags"] = []
            kwargs["dataset_tags"].append(dataset.dataset_name)

    trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()