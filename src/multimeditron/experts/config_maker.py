"""Generate a grid of training configs from a datamix + hyperparameter spec.

Input YAML format (see all_medical_datasets_config.yaml for an example):

    datamixes:
      all_domains:
        dataset_configs:
          - manifest_path: /path/to/benchmark_splits/multimediset/ct/train_model.jsonl
            domain: ct
            weight: 1.0

    base_configs:
      standard:
        learning_rate: 5.0e-05
        weight_decay: 0.05
        warmup_steps: 2000
        num_train_epochs: 1
        lr_scheduler_type: cosine

    param_ranges:
      learning_rate: [5.0e-05, 1.0e-04, 5.0e-04]
      weight_decay: [0.05, 0.2]

    common_config:
      vision_model_name: openai/clip-vit-base-patch32
      text_model_name: FacebookAI/roberta-base
      cache_dir: ""  # set to your local cache directory
      fp16: true
      bf16: false
      ...

    benchmark_selection:
      - ct3d
      - mri
      - skin
      - ophthalmology
      - ultrasound
      - xray

    output_dir: src/multimeditron/experts/configurations
    models_dir: src/multimeditron/experts/models

Each (datamix × base_config × param_grid) combination produces one YAML file named
{datamix}_{config_name}_config_{idx}.yaml.
"""

import itertools
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    manifest_path: str
    domain: str
    weight: float = 1.0


class Datamix(BaseModel):
    dataset_configs: list[DatasetConfig] = Field(default_factory=list)


class BaseConfig(BaseModel):
    learning_rate: float = 5.0e-5
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    num_train_epochs: int = 1
    lr_scheduler_type: str = "cosine"


class CommonConfig(BaseModel):
    vision_model_name: str = "openai/clip-vit-base-patch32"
    text_model_name: str = "FacebookAI/roberta-base"
    cache_dir: str = ""
    fp16: bool = True
    bf16: bool = False
    do_train: bool = True
    do_eval: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1.0e-6
    per_device_eval_batch_size: int = 64
    dataloader_drop_last: bool = True
    remove_unused_columns: bool = False
    overwrite_output_dir: bool = True
    save_steps: int = 150
    max_examples_per_domain: int = 30000
    target_per_domain: int = 16000


class Configurations(BaseModel):
    datamixes: dict[str, Datamix] = Field(default_factory=dict)
    base_configs: dict[str, BaseConfig] = Field(default_factory=lambda: {"standard": BaseConfig()})
    param_ranges: dict[str, list[Any]] = Field(default_factory=dict)
    common_config: CommonConfig = Field(default_factory=CommonConfig)
    benchmark_selection: list[str] = Field(default_factory=list)
    output_dir: str = "src/multimeditron/experts/configurations"
    models_dir: str = "src/multimeditron/experts/models"


def load_configurations(config_path: str) -> Configurations:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Normalize datamixes: each value may be a plain dict with dataset_configs
    datamixes = {}
    for name, mix_data in (raw.get("datamixes") or {}).items():
        if isinstance(mix_data, dict):
            datamixes[name] = Datamix(**mix_data)
        else:
            datamixes[name] = Datamix()

    base_configs = {}
    for name, cfg in (raw.get("base_configs") or {}).items():
        base_configs[name] = BaseConfig(**cfg)

    common_config = CommonConfig(**(raw.get("common_config") or {}))

    return Configurations(
        datamixes=datamixes,
        base_configs=base_configs,
        param_ranges=raw.get("param_ranges") or {},
        common_config=common_config,
        benchmark_selection=raw.get("benchmark_selection") or [],
        output_dir=raw.get("output_dir", "src/multimeditron/experts/configurations"),
        models_dir=raw.get("models_dir", "src/multimeditron/experts/models"),
    )


def main(config_path: str) -> None:
    configs = load_configurations(config_path)

    output_dir = Path(configs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_names = list(configs.param_ranges.keys())
    param_values = list(configs.param_ranges.values())
    grid = list(itertools.product(*param_values)) if param_values else [()]

    n_generated = 0

    for datamix_name, datamix in configs.datamixes.items():
        for config_name, base_config in configs.base_configs.items():
            for idx, combination in enumerate(grid, start=1):
                # Start from common config
                out: dict[str, Any] = configs.common_config.model_dump()

                # Layer base config on top
                out.update(base_config.model_dump())

                # Layer grid-search overrides on top
                if param_names:
                    out.update(dict(zip(param_names, combination)))

                # Output model dir
                out["output_dir"] = (
                    f"{configs.models_dir}/{datamix_name}_{config_name}_config_{idx}"
                )

                # Dataset configs (manifest-based format)
                out["dataset_configs"] = [
                    {"manifest_path": ds.manifest_path, "domain": ds.domain, "weight": ds.weight}
                    for ds in datamix.dataset_configs
                ]

                # Benchmark selection
                if configs.benchmark_selection:
                    out["benchmark_selection"] = configs.benchmark_selection

                filename = f"{datamix_name}_{config_name}_config_{idx}.yaml"
                filepath = output_dir / filename
                with open(filepath, "w") as f:
                    yaml.dump(out, f, default_flow_style=False, sort_keys=True)

                n_generated += 1

    print(f"Generated {n_generated} config(s) in '{output_dir}'.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not config_path:
        print("Usage: config_maker.py <config.yaml>", file=sys.stderr)
        sys.exit(1)
    main(config_path)
