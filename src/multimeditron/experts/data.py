#!/usr/bin/env python
# coding=utf-8
"""Dataset loading, domain balancing, and image utilities for multi-domain CLIP training."""
import functools
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional

import numpy as np
import torch
import transformers
import yaml
from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk
from PIL import Image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt",
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Pretrained model name or path."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for pretrained models."})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast tokenizer."})
    token: Optional[str] = field(default=None, metadata={"help": "HF auth token."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code from Hub."})
    vision_model_name: Optional[str] = field(default=None, metadata={"help": "Vision encoder name/path."})
    text_model_name: Optional[str] = field(default=None, metadata={"help": "Text encoder name/path."})


@dataclass
class DatasetConfig:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name (HuggingFace hub)."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "Dataset configuration name."})
    data_dir: Optional[str] = field(default=None, metadata={"help": "Data directory path."})
    image_column: str = field(default="modalities", metadata={"help": "Column containing image file paths."})
    caption_column: str = field(default="text", metadata={"help": "Column containing image captions."})
    domain: Optional[str] = field(default=None, metadata={"help": "Semantic training domain (e.g. ct, xray)."})
    manifest_path: Optional[str] = field(default=None, metadata={"help": "Path to MultiMediset JSONL manifest."})
    max_examples: Optional[int] = field(default=None, metadata={"help": "Per-dataset example cap."})
    weight: float = field(default=1.0, metadata={"help": "Sampling weight for dataset mixing."})


@dataclass
class DataTrainingArguments:
    dataset_configs: Optional[List[str]] = field(default=None, metadata={"help": "Dataset configuration for training."})
    freeze: bool = field(default=False, metadata={"help": "Freeze encoder weights during training."})
    max_seq_length: int = field(default=512, metadata={"help": "Max tokenized sequence length."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Truncate training set to this size."})
    max_examples_per_domain: Optional[int] = field(default=None, metadata={"help": "Global per-dataset example cap."})
    target_per_domain: Optional[int] = field(default=None, metadata={"help": "Fixed per-domain example budget for balancing."})


class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.transforms(x)
        return x


def collate_fn(examples):
    examples = [
        e for e in examples
        if e.get("pixel_values") is not None
        and e.get("input_ids") is not None
        and e.get("attention_mask") is not None
    ]
    if not examples:
        raise ValueError("All samples in the batch were invalid")

    pixel_values_list, input_ids_list, attention_mask_list = zip(*[
        (e["pixel_values"], e["input_ids"], e["attention_mask"]) for e in examples
    ])
    return {
        "pixel_values": torch.stack(pixel_values_list),
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        "return_loss": True,
    }


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@functools.lru_cache(maxsize=None)  # many manifest records share the same source_root; cache avoids reloading
@functools.lru_cache(maxsize=None)
def _load_manifest_source_dataset(source_root):
    if (
        os.path.exists(os.path.join(source_root, "state.json"))
        or os.path.exists(os.path.join(source_root, "dataset_dict.json"))
        or os.path.exists(os.path.join(source_root, "train", "state.json"))
    ):
        loaded = load_from_disk(source_root, keep_in_memory=False)
        if hasattr(loaded, "keys"):
            return {split: loaded[split] for split in loaded.keys()}
        return {"train": loaded}

    train_jsonl = os.path.join(source_root, "MRI-glob-train.jsonl")
    test_jsonl = os.path.join(source_root, "MRI-glob-test.jsonl")
    if os.path.exists(train_jsonl) and os.path.exists(test_jsonl):
        return {"train": _read_jsonl(train_jsonl), "test": _read_jsonl(test_jsonl)}

    jsonl = os.path.join(source_root, "MRI-glob.jsonl")
    if os.path.exists(jsonl):
        return {"all": _read_jsonl(jsonl)}

    raise FileNotFoundError(f"Could not load source dataset from manifest root: {source_root}")


def _dict_img_payload(d):
    if isinstance(d, dict):
        if d.get("bytes") is not None:
            return {"bytes": d["bytes"]}
        if d.get("path"):
            return {"path": d["path"]}
    return None


def _first_image_payload(row, source_root):
    # Datasets use heterogeneous column names; try image → modalities_images → modalities in order.
    payload = _dict_img_payload(row.get("image"))
    if payload is not None:
        return payload

    modalities_images = row.get("modalities_images")
    if isinstance(modalities_images, list) and modalities_images:
        payload = _dict_img_payload(modalities_images[0])
        if payload is not None:
            return payload

    modalities = row.get("modalities") or []
    if modalities and isinstance(modalities[0], dict):
        value = modalities[0].get("value")
        if isinstance(value, str) and value:
            if os.path.isabs(value):
                return value
            for candidate in [
                os.path.join(source_root, value),
                os.path.join(source_root, os.path.basename(value)),
                os.path.join(source_root, "images", os.path.basename(value)),
            ]:
                if os.path.exists(candidate):
                    return candidate
            return os.path.join(source_root, value)

    return None


def _normalize_config(config):
    if not isinstance(config, DatasetConfig):
        if config.get("dataset_name") is None and "manifest_path" not in config:
            assert len(config) == 1, f"Expected a single-key wrapper dict, got keys: {list(config.keys())}"
            config = config[next(iter(config))]
        config = DatasetConfig(**config)
    if not config.domain:
        raise ValueError(
            f"Dataset {config.dataset_name or config.data_dir or config.manifest_path!r} is missing a 'domain' field in the config."
        )
    return config


def get_combined_dataset(dataset_configs, model_args, max_examples_per_domain=None, target_per_domain=None):
    """Build a domain-balanced training mixture, sampling each domain down to the smallest domain size."""

    def _load_manifest_dataset(config):
        manifest_path = os.path.abspath(config.manifest_path)
        records = _read_jsonl(manifest_path)
        effective_max = config.max_examples if config.max_examples is not None else max_examples_per_domain
        if effective_max is not None and len(records) > effective_max:
            rng = np.random.default_rng(42)
            selected_indices = rng.choice(len(records), size=effective_max, replace=False)
            records = [records[int(idx)] for idx in selected_indices]
        standardized = []
        skipped = 0

        for record in records:
            source_root = record["source_root"]
            source_dataset = _load_manifest_source_dataset(source_root)
            source_split = record["source_split"]
            source_index = int(record["source_index"])
            if source_split not in source_dataset:
                skipped += 1
                continue
            row = dict(source_dataset[source_split][source_index])
            caption = (
                str(row.get("text") or row.get("caption") or "")
                .replace("<attachment>", "")
                .strip()
            )
            if not caption and record.get("label"):
                caption = f"A {record.get('benchmark', config.domain)} medical image showing {record['label']}."
            if not caption:
                skipped += 1
                continue
            standardized.append(
                {
                    "image_path": json.dumps({
                        "__manifest_source__": True,
                        "source_root": source_root,
                        "source_split": source_split,
                        "source_index": source_index,
                    }),
                    "caption": caption,
                    "domain": config.domain,
                    "benchmark": record.get("benchmark", config.domain),
                    "dataset": record.get("dataset"),
                }
            )

        if skipped:
            logger.warning("Skipped %s manifest record(s) from %s", skipped, manifest_path)
        if not standardized:
            raise ValueError(f"No usable training examples were loaded from manifest {manifest_path}")
        return Dataset.from_list(standardized)

    def _load_dataset(config):
        if config.manifest_path:
            return _load_manifest_dataset(config)
        if config.data_dir and (
            os.path.exists(os.path.join(config.data_dir, "state.json"))
            or os.path.exists(os.path.join(config.data_dir, "dataset_dict.json"))
            or os.path.exists(os.path.join(config.data_dir, "train", "state.json"))
        ):
            return load_from_disk(config.data_dir, keep_in_memory=True)
        if config.dataset_name.endswith(".jsonl"):
            return load_dataset(
                "json",
                config.dataset_config_name,
                cache_dir=model_args.cache_dir,
                keep_in_memory=False,
                data_dir=config.data_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                data_files=config.dataset_name,
            )
        return load_dataset(
            config.dataset_name,
            config.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=config.data_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    def _allocate_counts(sizes, total_budget):
        # Proportional allocation with largest-remainder rounding, then a shortfall pass if any
        # dataset is smaller than its proportional share.
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

    def _standardize_split(train_split, config):
        if config.manifest_path:
            return train_split.select_columns(["image_path", "caption"])

        is_jsonl = config.dataset_name.endswith(".jsonl")

        def standardize_sample(row):
            image_value = row[config.image_column]
            if is_jsonl:
                image_value = os.path.join(
                    os.path.dirname(config.dataset_name),
                    image_value[0]["value"],
                )
            else:
                if isinstance(image_value, list):
                    image_value = image_value[0] if image_value else None
                if isinstance(image_value, dict) and set(image_value.keys()) == {"type", "value"}:
                    image_value = image_value["value"]  # HuggingFace Arrow image column format
            return {
                "image_path": image_value,
                "caption": str(row[config.caption_column]).replace("<attachment>", ""),
            }

        return train_split.map(standardize_sample, load_from_cache_file=True).select_columns(["image_path", "caption"])

    logger.info(f"Loading datasets: {dataset_configs}")
    parsed_configs = [_normalize_config(config) for config in dataset_configs]

    per_dataset_entries = []
    domain_sizes = defaultdict(int)
    for config in parsed_configs:
        dataset = _load_dataset(config)
        train_split = dataset["train"] if "train" in dataset else dataset
        split_length = len(train_split)
        per_dataset_entries.append({"config": config, "train_split": train_split, "length": split_length})
        domain_sizes[config.domain] += split_length

    if not domain_sizes:
        raise ValueError("No datasets were loaded for training.")

    logger.info("Domain sizes before balancing: %s", dict(domain_sizes))

    smallest_domain = min(domain_sizes.values())
    if target_per_domain is None:
        target_per_domain = smallest_domain
    elif target_per_domain > smallest_domain:
        logger.warning(
            "target_per_domain=%d capped to smallest domain size=%d",
            target_per_domain,
            smallest_domain,
        )
        target_per_domain = smallest_domain

    domain_entries = defaultdict(list)
    for entry in per_dataset_entries:
        domain_entries[entry["config"].domain].append(entry)

    all_splits = []
    for domain, entries in domain_entries.items():
        sizes = [entry["length"] for entry in entries]
        counts = _allocate_counts(sizes, target_per_domain)

        allocation_log = [
            {"dataset": entry["config"].dataset_name, "kept": count, "available": entry["length"]}
            for entry, count in zip(entries, counts)
        ]
        logger.info("Domain %s allocation: %s", domain, allocation_log)

        for entry, count in zip(entries, counts):
            split = entry["train_split"]
            if count < len(split):
                split = split.shuffle(seed=42).select(range(count))
            all_splits.append(_standardize_split(split, entry["config"]))

    if not all_splits:
        raise ValueError("No domain datasets could be built.")

    combined_dataset = interleave_datasets(all_splits, stopping_strategy="all_exhausted")
    return combined_dataset.train_test_split(test_size=0.1, seed=42)


def load_image_any(image_obj):
    """Load an RGB PIL image from a file path (str), bytes dict, manifest pointer dict, or PIL.Image."""
    if image_obj is None:
        raise ValueError("image is None")
    if isinstance(image_obj, str):
        if image_obj.startswith('{"__manifest_source__"'):
            return load_image_any(json.loads(image_obj))
        with Image.open(image_obj) as img:
            return img.convert("RGB")
    if isinstance(image_obj, dict):
        if image_obj.get("__manifest_source__"):
            source_root = image_obj["source_root"]
            source_dataset = _load_manifest_source_dataset(source_root)
            row = dict(source_dataset[image_obj["source_split"]][int(image_obj["source_index"])])
            payload = _first_image_payload(row, source_root)
            if payload is None:
                raise ValueError("manifest source row has no usable image payload")
            return load_image_any(payload)
        if image_obj.get("bytes") is not None:
            with Image.open(BytesIO(image_obj["bytes"])) as img:
                return img.convert("RGB")
        if image_obj.get("path"):
            with Image.open(image_obj["path"]) as img:
                return img.convert("RGB")
        raise ValueError("dict image payload has neither bytes nor path")
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_obj)}")


def data_processing(config_path):
    """Parse YAML config, set up logging, detect last checkpoint, and return the combined dataset."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(
        yaml_file=os.path.abspath(config_path),
        allow_extra_keys=True,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    training_args.dataloader_num_workers = 4

    last_checkpoint = None
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}
    overwrite_output_dir = getattr(training_args, "overwrite_output_dir", raw_config.get("overwrite_output_dir", False))
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not overwrite_output_dir:
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

    if data_args.dataset_configs is not None:
        dataset = get_combined_dataset(
            data_args.dataset_configs,
            model_args,
            max_examples_per_domain=data_args.max_examples_per_domain,
            target_per_domain=data_args.target_per_domain,
        )
    else:
        raise ValueError("Please provide dataset configs")

    return model_args, data_args, training_args, dataset, last_checkpoint
