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
import sys
from datetime import datetime
from io import BytesIO
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
import transformers
import wandb
from datasets import concatenate_datasets, interleave_datasets, load_dataset, load_from_disk
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    VisionTextDualEncoderModel,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

# Disable WANDB if needed
# os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# from transformers.utils import check_min_version
# check_min_version("4.47.0.dev0")
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt",
)


# Definition of config arguments
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    image_processor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
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
    vision_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Vision encoder model name/path (e.g. openai/clip-vit-base-patch32)"
        },
    )
    text_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Text encoder model name/path (e.g. FacebookAI/roberta-base)"
        },
    )


@dataclass
class DatasetConfig:
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The data directory containing input files."}
    )
    image_column: Optional[str] = field(
        default="modalities",
        metadata={
            "help": "The name of the column in the datasets containing the full image file paths."
        },
    )
    caption_column: Optional[str] = field(
        default="text",
        metadata={
            "help": "The name of the column in the datasets containing the image captions."
        },
    )
    weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "The weight to assign to this dataset during training."},
    )
    domain: Optional[str] = field(
        default=None,
        metadata={"help": "Semantic training domain used for balanced sampling (e.g. ct, xray, ultrasound)."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_configs: Optional[List[DatasetConfig]] = field(
        default=None,
        metadata={"help": "Dataset configuration for training and evaluation."},
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
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize(
                [image_size], interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x


def collate_fn(examples):
    """
    Stack the examples into a format fit for training.
    """

    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor(
        [example["input_ids"] for example in examples], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [example["attention_mask"] for example in examples], dtype=torch.long
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }


def get_combined_dataset(
    dataset_configs: List[DatasetConfig], model_args: ModelArguments
):
    """Build a domain-balanced training mixture with early subsetting."""

    def _normalize_config(config):
        if isinstance(config, DatasetConfig):
            parsed = config
        else:
            if config.get("dataset_path", None) is None:
                assert len(config) == 1
                config = config[list(config.keys())[0]]
            parsed = DatasetConfig(**config)
        if not parsed.domain:
            raise ValueError(
                f"Dataset {parsed.dataset_path or parsed.data_dir!r} is missing a 'domain' field in the config."
            )
        return parsed

    def _load_dataset(config: DatasetConfig):
        if config.dataset_path is not None and os.path.isdir(config.dataset_path):
            return load_from_disk(config.dataset_path, keep_in_memory=True)
        if config.dataset_path.endswith(".jsonl"):
            return load_dataset(
                "json",
                config.dataset_config_name,
                cache_dir=model_args.cache_dir,
                keep_in_memory=False,
                data_dir=config.data_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                data_files=config.dataset_path,
            )
        return load_dataset(
            config.dataset_path,
            config.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=config.data_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    def _get_train_split(dataset):
        return dataset["train"] if "train" in dataset else dataset

    def _allocate_counts(sizes: list[int], total_budget: int) -> list[int]:
        if not sizes:
            return []
        total_size = sum(sizes)
        if total_size <= total_budget:
            return sizes
        raw = [size * total_budget / total_size for size in sizes]
        counts = [int(value) for value in raw]
        remainder = total_budget - sum(counts)
        order = sorted(
            range(len(sizes)),
            key=lambda idx: (raw[idx] - counts[idx], sizes[idx]),
            reverse=True,
        )
        for idx in order[:remainder]:
            counts[idx] += 1
        counts = [min(count, size) for count, size in zip(counts, sizes, strict=True)]
        shortfall = total_budget - sum(counts)
        if shortfall > 0:
            for idx in order:
                available = sizes[idx] - counts[idx]
                if available <= 0:
                    continue
                take = min(shortfall, available)
                counts[idx] += take
                shortfall -= take
                if shortfall == 0:
                    break
        return counts

    def _subset_before_mapping(train_split, count: int):
        if count >= len(train_split):
            return train_split
        return train_split.shuffle(seed=42).select(range(count))

    def _standardize_split(train_split, config: DatasetConfig):
        if config.dataset_path.endswith(".jsonl"):
            def find_img_path(row):
                return {
                    config.caption_column: row[config.caption_column],
                    config.image_column: os.path.join(
                        os.path.dirname(config.dataset_path),
                        row[config.image_column][0]["value"],
                    ),
                }
            train_split = train_split.map(find_img_path)

        def standardize_sample(row):
            image_value = row[config.image_column]

            # Normalize all image payloads to one Arrow-compatible schema so
            # concatenate/interleave don't have to align mixed image types.
            if (
                isinstance(image_value, dict)
                and "type" in image_value
                and "value" in image_value
                and set(image_value.keys()) == {"type", "value"}
            ):
                image_value = image_value["value"]

            if isinstance(image_value, str):
                image_value = [{"bytes": b"", "path": image_value}]
            elif isinstance(image_value, dict) and set(image_value.keys()) >= {"bytes", "path"}:
                image_value = [{
                    "bytes": image_value.get("bytes") or b"",
                    "path": image_value.get("path") or "",
                }]
            elif isinstance(image_value, list):
                normalized_images = []
                for image_entry in image_value:
                    if isinstance(image_entry, str):
                        normalized_images.append({"bytes": b"", "path": image_entry})
                    elif isinstance(image_entry, dict) and set(image_entry.keys()) >= {"bytes", "path"}:
                        normalized_images.append({
                            "bytes": image_entry.get("bytes") or b"",
                            "path": image_entry.get("path") or "",
                        })
                    elif (
                        isinstance(image_entry, dict)
                        and "type" in image_entry
                        and "value" in image_entry
                        and set(image_entry.keys()) == {"type", "value"}
                    ):
                        normalized_images.append({"bytes": b"", "path": image_entry["value"]})
                    else:
                        raise TypeError(f"Unsupported image list entry type: {type(image_entry)}")
                image_value = normalized_images
            else:
                raise TypeError(f"Unsupported image payload type: {type(image_value)}")

            return {
                "image_path": image_value,
                "caption": str(row[config.caption_column]).replace("<attachment>", ""),
            }

        train_split = train_split.map(
            standardize_sample,
            remove_columns=train_split.column_names,
        )
        return train_split.select_columns(["image_path", "caption"])

    logger.info(f"Loading datasets: {dataset_configs}")
    parsed_configs = [_normalize_config(config) for config in dataset_configs]

    per_dataset_entries = []
    domain_sizes = defaultdict(int)
    for config in parsed_configs:
        dataset = _load_dataset(config)
        train_split = _get_train_split(dataset)
        split_length = len(train_split)
        per_dataset_entries.append({
            "config": config,
            "train_split": train_split,
            "length": split_length,
        })
        domain_sizes[config.domain] += split_length

    if not domain_sizes:
        raise ValueError("No datasets were loaded for training.")

    target_per_domain = min(domain_sizes.values())
    logger.info("Domain sizes before balancing: %s", dict(domain_sizes))
    logger.info("Balanced domain budget: %s example(s) per domain", target_per_domain)

    domain_entries = defaultdict(list)
    for entry in per_dataset_entries:
        domain_entries[entry["config"].domain].append(entry)

    balanced_domain_splits = []
    for domain, entries in domain_entries.items():
        allocations = _allocate_counts([entry["length"] for entry in entries], target_per_domain)
        logger.info(
            "Domain %s allocation: %s",
            domain,
            [
                {
                    "dataset": entry["config"].dataset_path or entry["config"].data_dir,
                    "kept": kept,
                    "available": entry["length"],
                }
                for entry, kept in zip(entries, allocations, strict=True)
            ],
        )
        standardized_splits = []
        for entry, kept_count in zip(entries, allocations, strict=True):
            if kept_count <= 0:
                continue
            sampled_split = _subset_before_mapping(entry["train_split"], kept_count)
            standardized_splits.append(_standardize_split(sampled_split, entry["config"]))
        if not standardized_splits:
            continue
        domain_dataset = concatenate_datasets(standardized_splits).shuffle(seed=42)
        if len(domain_dataset) > target_per_domain:
            domain_dataset = domain_dataset.select(range(target_per_domain))
        balanced_domain_splits.append(domain_dataset)

    if not balanced_domain_splits:
        raise ValueError("No balanced domain datasets could be built.")

    combined_dataset = interleave_datasets(
        balanced_domain_splits,
        probabilities=[1.0 / len(balanced_domain_splits)] * len(balanced_domain_splits),
        seed=42,
        stopping_strategy="all_exhausted",
    )
    return combined_dataset.train_test_split(test_size=0.1, seed=42)


def main(config_path: str):
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_yaml_file(
        yaml_file=os.path.abspath(config_path)
    )

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

    # Append timestamp to output_dir to avoid conflicts between runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args.output_dir = f"{training_args.output_dir}_{timestamp}"

    # Training args
    training_args.dataloader_drop_last = True
    training_args.dataloader_num_workers = 4
    training_args.logging_steps = 50
    training_args.fp16 = False
    training_args.bf16 = True
    training_args.gradient_accumulation_steps = 2
    if not os.environ.get("WANDB_DISABLED", False):  # setup wandb
        training_args.report_to = ["wandb"]

        run_name = f"Training CLIP {os.path.basename(config_path)}"
        training_args.run_name = run_name
        wandb.init(
            project="Training CLIP", name=run_name, config=training_args.to_dict()
        )

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
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
    if model_args.vision_model_name and model_args.text_model_name:
        # Dual encoder path
        logger.info(
            f"Loading dual encoder with vision model {model_args.vision_model_name} "
            f"and text model {model_args.text_model_name}"
        )

        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            model_args.vision_model_name,
            model_args.text_model_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        ).to(dtype=torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.text_model_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
        )

        image_processor = AutoImageProcessor.from_pretrained(
            model_args.vision_model_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    config = model.config

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train or training_args.do_eval:
        column_names = dataset["train"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    # 6. Get the column names for input/target.

    image_column = "image_path"
    caption_column = "caption"

    # 7. Preprocessing the datasets.
    # Initialize torchvision transforms and jit it for faster processing.
    image_transformations = Transform(
        config.vision_config.image_size,
        image_processor.image_mean,
        image_processor.image_std,
    )
    image_transformations = torch.jit.script(image_transformations)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(
            captions,
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples


    def transform_images(examples):
        images = []
        for image_file in examples[image_column]:
            if isinstance(image_file, list):
                if not image_file:
                    raise TypeError("Empty image list")
                image_file = image_file[0]

            if isinstance(image_file, str):
                image = read_image(image_file, mode=ImageReadMode.RGB)
            elif isinstance(image_file, dict):
                if image_file.get("bytes") is not None:
                    image = torch.from_numpy(
                        np.array(Image.open(BytesIO(image_file["bytes"])).convert("RGB"))
                    ).permute(2, 0, 1)
                elif image_file.get("path"):
                    image = read_image(image_file["path"], mode=ImageReadMode.RGB)
                else:
                    raise TypeError(f"Unsupported dict image format: {image_file.keys()}")
            else:
                image = torch.from_numpy(np.array(image_file)).permute(2, 0, 1)

            images.append(image)

        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples



    def filter_corrupt_images(examples):
        valid_images = []

        for image_file in examples[image_column]:
            try:
                if isinstance(image_file, list):
                    if not image_file:
                        raise TypeError("Empty image list")
                    image_file = image_file[0]

                if isinstance(image_file, str):
                    Image.open(image_file).verify()

                elif isinstance(image_file, dict):
                    if image_file.get("bytes") is not None:
                        Image.open(BytesIO(image_file["bytes"])).verify()
                    elif image_file.get("path"):
                        Image.open(image_file["path"]).verify()
                    else:
                        raise TypeError(f"Unsupported dict image format: {image_file.keys()}")

                else:
                    Image.fromarray(np.array(image_file)).verify()

                valid_images.append(True)

            except Exception as e:
                logger.warning(f"Invalid image skipped: {e}")
                valid_images.append(False)

        return valid_images


    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.info(f"Dataset length: {len(train_dataset)}")
        train_dataset = train_dataset.filter(
            filter_corrupt_images,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
        logger.info(f"Dataset length without corrupt images: {len(train_dataset)}")

        train_dataset = train_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
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

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        test_dataset.set_transform(transform_images)

    # 8. Initialize our trainer
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
        image_processor.save_pretrained(training_args.output_dir)
        if train_result:
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)

            if not os.environ.get("WANDB_DISABLED", False):
                wandb.log({"train": train_result.metrics})
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if not os.environ.get("WANDB_DISABLED", False):
            wandb.log({"eval": metrics})

    # 11. Write Training Stats and push to hub.
    finetuned_from = model_args.model_name_or_path
    # If from a local directory, don't set `finetuned_from` as this is required to be a valid repo. id on the Hub.

    if finetuned_from is None or os.path.isdir(finetuned_from):
        finetuned_from = None
    kwargs = {
        "finetuned_from": finetuned_from,
        "tasks": "contrastive-image-text-modeling",
    }
    for dataset in data_args.dataset_configs:
        if dataset.get("dataset_path", None) is None:
            assert len(dataset) == 1
            dataset = dataset[list(dataset.keys())[0]]
        dataset = DatasetConfig(**dataset)
        if dataset.dataset_path is not None:
            if not hasattr(kwargs, "dataset_tags"):
                kwargs["dataset_tags"] = []
            kwargs["dataset_tags"].append(dataset.dataset_path)

    trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main(sys.argv[1])
