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
import copy
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import optuna
import torch
import wandb
import yaml

from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    VisionTextDualEncoderModel,
    Trainer,
    TrainerCallback,
    set_seed,
)

from multimeditron.experts.lion.modeling_clip import (
    OpenCLIPVisionTextDualEncoderModel,
    VisionTextDualEncoderConfig,
)
from multimeditron.experts.data import (
    Transform,
    collate_fn,
    load_image_any,
    _normalize_config,
    data_processing,
)

# os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)


class OptunaPruningCallback(TrainerCallback):
    """Reports -loss to Optuna at each logging step so MedianPruner can kill bad trials early."""

    def __init__(self, trial):
        self.trial = trial

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and state.global_step > 0:
            self.trial.report(-logs["loss"], step=state.global_step)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()


def _load_model(model_args):
    """Load model, tokenizer and image processor — OpenCLIP path or standard VisionTextDualEncoder."""
    _OPENCLIP_MODEL_NAME = "CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k"
    if model_args.vision_model_name == _OPENCLIP_MODEL_NAME:
        logger.info("Loading OpenCLIP dual encoder: %s", model_args.vision_model_name)
        model_id = "calpt/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k"
        config = VisionTextDualEncoderConfig.from_pretrained(model_id)
        config.vision_config.hidden_act = "gelu"
        model = OpenCLIPVisionTextDualEncoderModel.from_pretrained(model_id, config=config)
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        logger.info(
            "Loading dual encoder: vision=%s text=%s",
            model_args.vision_model_name,
            model_args.text_model_name,
        )
        # Force use_safetensors to avoid PyTorch security issue (CVE-2025-32434)
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            model_args.vision_model_name,
            model_args.text_model_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
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
    return model, tokenizer, image_processor


def training(
    model_args, data_args, training_args, dataset, n_freeze, last_checkpoint, trial=None
):
    """Train and/or evaluate the dual encoder; save model, metrics and model card. Returns (train_loss, model)."""
    if not (training_args.do_train or training_args.do_eval):
        logger.info("Nothing to do — pass do_train and/or do_eval.")
        return None, None

    set_seed(training_args.seed)

    model, tokenizer, image_processor = _load_model(model_args)

    if n_freeze > 0:
        for i, layer in enumerate(model.vision_model.vision_model.encoder.layers):
            if i < n_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        for i in range(n_freeze):
            for param in model.text_model.encoder.layer[i].parameters():
                param.requires_grad = False

    config = model.config

    image_transformations = Transform(
        config.vision_config.image_size,
        image_processor.image_mean,
        image_processor.image_std,
    )
    logger.info(
        "vision_config.image_size=%s image_processor.image_mean=%s image_processor.image_std=%s",
        config.vision_config.image_size,
        image_processor.image_mean,
        image_processor.image_std,
    )
    image_transformations = torch.jit.script(image_transformations)  # JIT for faster per-sample transform

    def transform_batch(examples):
        text_inputs = tokenizer(
            list(examples["caption"]),
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        pixel_values = []
        for image_obj in examples["image_path"]:
            try:
                pil_image = load_image_any(image_obj)
                image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1)
                pixel_values.append(image_transformations(image))
            except Exception as e:
                logger.warning(f"Skipping invalid image sample: {str(image_obj)[:300]}, Error: {str(e)}")
                pixel_values.append(None)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        examples["pixel_values"] = pixel_values
        return examples

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        logger.info(f"Dataset length: {len(train_dataset)}")
        train_dataset.set_transform(transform_batch)

    if training_args.do_eval:
        test_dataset = dataset["test"]
        test_dataset.set_transform(transform_batch)

    pruning_callbacks = [OptunaPruningCallback(trial)] if trial is not None else []
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        callbacks=pruning_callbacks,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint is not None else last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        image_processor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        if not os.environ.get("WANDB_DISABLED", False) and wandb.run is not None:
            wandb.log({"train": train_result.metrics})
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if not os.environ.get("WANDB_DISABLED", False) and wandb.run is not None:
            wandb.log({"eval": metrics})

    finetuned_from = model_args.model_name_or_path
    if finetuned_from is None or os.path.isdir(finetuned_from):
        finetuned_from = None
    kwargs = {"finetuned_from": finetuned_from, "tasks": "contrastive-image-text-modeling"}
    for ds in (_normalize_config(c) for c in data_args.dataset_configs):
        if ds.dataset_name is not None:
            if "dataset_tags" not in kwargs:
                kwargs["dataset_tags"] = []
            kwargs["dataset_tags"].append(ds.dataset_name)

    trainer.create_model_card(**kwargs)
    train_loss = train_result.metrics["train_loss"] if training_args.do_train else None
    return train_loss, model


def objective(trial, bench_list, config_path, model_args, data_args, training_args_base, dataset, last_checkpoint):
    """Optuna objective: sample hyperparams, train, run benchmarks, return mean benchmark score."""
    training_args = copy.deepcopy(training_args_base)
    lr = trial.suggest_float("learning_rate", 1.0e-6, 5.0e-4, log=True)
    wd = trial.suggest_float("weight_decay", 1.0e-4, 0.1, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    n_frz = trial.suggest_int("freezed_layers", 0, 8) if data_args.freeze else 0
    if data_args.freeze:
        logger.info("Freezing %d encoder layers", n_frz)
    training_args.learning_rate = lr
    training_args.weight_decay = wd
    training_args.warmup_ratio = warmup_ratio
    training_args.output_dir = f"{training_args.output_dir}_lr{lr}_wd{wd}_nfrz{n_frz}"
    logger.info("Trial: lr=%s, wd=%s, warmup_ratio=%s, n_freeze=%s", lr, wd, warmup_ratio, n_frz)
    if not os.environ.get("WANDB_DISABLED", False):
        training_args.report_to = ["wandb"]
        run_name = f"Training CLIP {os.path.basename(config_path)}_lr:{lr}_wd:{wd}_wr:{warmup_ratio}_nfrz:{n_frz}"
        training_args.run_name = run_name
        wandb.init(project="Training CLIP", name=run_name, config=training_args.to_dict())
    loss_value, model = training(model_args, data_args, training_args, dataset, n_frz, last_checkpoint, trial=trial)

    model.eval()
    benchmark_results, benchmark_names, benchmark_metrics = [], [], []
    for benchmark in bench_list:
        name = benchmark.__class__.__name__
        result = benchmark.evaluate(training_args.output_dir)
        benchmark_results.append(result["score"])
        benchmark_names.append(name)
        benchmark_metrics.append(result)
        logger.info("Benchmark %s score: %s", name, result["score"])

    with open(os.path.join(training_args.output_dir, "benchmark_scores.txt"), "w") as f:
        f.write(f"Run name: {training_args.run_name}\n")
        f.write(f"Learning rate: {training_args.learning_rate}\n")
        f.write(f"Weight decay: {training_args.weight_decay}\n")
        f.write(f"Warmup ratio: {training_args.warmup_ratio}\n")
        f.write(f"Freezed layers: {n_frz}\n")
        f.write("Primary metric: macro-F1 (single-label) / micro-F1 (multi-label)\n\n")
        for name, metrics in zip(benchmark_names, benchmark_metrics):
            f.write(f"[{name}]\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")

    res = float(sum(benchmark_results) / len(benchmark_results))
    if wandb.run is not None:
        wandb.finish()
    return res


def train(bench_list, config_path):
    """Create an Optuna study, run HPO over the config, and return the completed study."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    model_args, data_args, training_args, dataset, last_checkpoint = data_processing(config_path)

    def objective_wrapper(trial):
        return objective(trial, bench_list, config_path, model_args, data_args, training_args, dataset, last_checkpoint)

    output_dir = config_dict.get("output_dir", ".")
    n_trials = config_dict.get("n_trials", 25)
    study_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(config_path).stem)
    storage_path = Path(output_dir) / f"{study_slug}_optuna.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=500, interval_steps=50),
        study_name=f"multidomain_clip_{study_slug}",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="maximize",
    )
    logger.info("Starting Optuna study with %d trials", n_trials)
    study.optimize(objective_wrapper, n_trials=n_trials)
    return study


def plot_study(study, output_dir="."):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Best hyperparameters: %s", study.best_params)

    best_params_path = output_dir / "best_params.json"
    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value}, f, indent=2)
    logger.info("Best params saved to %s", best_params_path)

    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / "parallel_coordinate.html"))
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(output_dir / "param_importances.html"))
        logger.info("Optuna plots saved to %s", output_dir)
    except Exception as exc:
        logger.warning("Could not generate Optuna plots (plotly installed?): %s", exc)


def main(config_path):
    from multimeditron.experts.evaluation_pipeline.build_benchmarks import build_benchmarks_from_names

    logger.info("Using config: %s", os.path.abspath(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    benchmark_selection = config_dict.get("benchmark_selection")
    bench_list = build_benchmarks_from_names(benchmark_selection)

    study = train(bench_list, config_path)
    plot_study(study, output_dir=config_dict.get("output_dir", "."))
    return study


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    main(args.config_file)
