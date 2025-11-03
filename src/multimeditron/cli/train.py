from multimeditron.cli import EPILOG, main_cli
from multimeditron.model.model import MultimodalConfig, MultiModalModelForCausalLM, bootstrap
from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.train.trainer import MultimodalTrainer, TRAINING_MAPPING
from multimeditron.profiling import NvtxAnnotationCallback
from transformers import AutoTokenizer, TrainingArguments
from datasets import concatenate_datasets, load_dataset, load_from_disk
from multimeditron.model.modalities import AutoModality
from multimeditron.dataset.loader import AutoModalityLoader
from multimeditron.model.model import MultiModalModelForCausalLM, MultimodalConfig
from tqdm import tqdm as _tqdm
from PIL import PngImagePlugin
from datasets import config as datasets_config
from pathlib import Path
from transformers.trainer_utils import get_last_checkpoint

import deepspeed
import torch
import os
import yaml
import wandb
import multiprocessing
import click
import logging


logger = logging.getLogger(__name__)

PngImagePlugin.MAX_TEXT_CHUNK = 2 ** 30

def is_dataset_folder(folder: str) -> bool:
    return os.path.exists(os.path.join(folder, datasets_config.DATASET_INFO_FILENAME)) and \
        os.path.exists(os.path.join(folder, datasets_config.DATASET_STATE_JSON_FILENAME))

def is_jsonl(path: str) -> bool:
    filename, extension = os.path.splitext(path)
    return extension == ".jsonl"

def build_datasets(config):
    packed_datasets = []

    # use env vars set by torchrun
    rank = int(os.environ.get("RANK", "0"))

    # give each process fair slice of CPUs (per node)
    # if SLURM_CPUS_PER_TASK is set, prefer it; else fallback to cpu_count
    cpus_visible = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", os.environ.get("NPROC_PER_NODE", "1")))
    num_proc = max(1, cpus_visible // gpus_per_node)

    tqdm = (lambda *a, **k: _tqdm(*a, disable=(rank != 0), **k))

    for ds_config in tqdm(config["datasets"], desc="Concatenating datasets"):
        if is_dataset_folder(ds_config["packed_path"]):
            dataset = load_from_disk(ds_config['packed_path'])
        else:
            dataset = load_dataset(ds_config["packed_path"], num_proc=num_proc)["train"]
        packed_datasets.append(dataset)

    ds = concatenate_datasets(packed_datasets).shuffle(seed=config.get("seed", 0))
    return ds


@main_cli.command(epilog=EPILOG)
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to the configuration file(s) in YAML format.")
@click.option("--trust-remote-code/--no-trust-remote-code", default=False, help="Whether to trust remote code when loading models from HuggingFace.")
@click.option("--seed", "-s", default=0, help="Seed of random")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose printing")
def train(config: str,
          trust_remote_code: bool = False,
          seed: int = 0,
          verbose: bool = False):
    
    with open(config) as f:
        config_dict = yaml.safe_load(f)
    
    ATTACHMENT_TOKEN = config_dict["attachment_token"]

    # load resume flag
    resume_flag = bool(config_dict.get("resume_from_checkpoint", False))
    
    # determinism for reproducible resumes.
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    training_args = TrainingArguments(**config_dict["training_args"])

    output_dir = training_args.output_dir
    output_dir_path = Path(output_dir) if output_dir else None
    if output_dir_path:
        output_dir_path.mkdir(parents=True, exist_ok=True)

    last_ckpt = None
    if resume_flag:
        try:
            if output_dir and os.path.isdir(output_dir):
                last_ckpt = get_last_checkpoint(output_dir)
                if last_ckpt:
                    logger.info(f"[resume=true] Found last checkpoint under output_dir: {last_ckpt}")
        except Exception as e:
            logger.warning(f"Failed to probe last checkpoint in output_dir: {e}")

    base_model_path = config_dict.get("base_model", None)
    base_model_is_checkpoint = (
        isinstance(base_model_path, str)
        and os.path.isdir(base_model_path)
        and os.path.basename(base_model_path).startswith("checkpoint-")
    )

    # Selection order when resume=true:
    #   1) latest checkpoint in output_dir (resuming same run)
    #   2) else base_model if it is a checkpoint-* path (e.g. second-stage starting point)
    # When resume=false: don't resume from anything.
    if resume_flag:
        if last_ckpt:
            resume_ckpt = last_ckpt
        elif base_model_is_checkpoint:
            resume_ckpt = base_model_path
            logger.info(f"[resume=true] Using base_model as resume checkpoint: {resume_ckpt}")
        else:
            resume_ckpt = None
            logger.info("[resume=true] No checkpoint found; proceeding without resume_from_checkpoint.")
    else:
        resume_ckpt = None
        logger.info("[resume=false] Fresh training (no resume_from_checkpoint).")

    # === Tokenizer === 
    tokenizer = AutoTokenizer.from_pretrained(config_dict["base_llm"], padding_side='right', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {'additional_special_tokens': [ATTACHMENT_TOKEN]}
    tokenizer.add_special_tokens(special_tokens)
    attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)

    # === Model ===
    torch.set_default_dtype(torch.bfloat16)
    
    modalities_config = []
    for modality in config_dict.get("modalities", []):
        modalities_config.append(AutoModality.config_from_dict(modality))

    modalities_loader = dict()
    for loader in config_dict["loaders"]:
        loader_copy = loader.copy()
        loader_type = loader_copy.pop("loader_type")
        modality_type = loader_copy.pop("modality_type")
        modalities_loader[modality_type] = AutoModalityLoader.from_name(loader_type, **loader_copy)

    with deepspeed.zero.Init(dtype=torch.bfloat16):
        if config_dict.get("base_model", None) is None:
            # no base model, bootstrap brand-new model.
            model = bootstrap(config_dict, tokenizer, attachment_token_idx, modalities_config)
        else:
            # load starting weights from base_model (can be a hub id or local checkpoint dir).
            model = MultiModalModelForCausalLM.from_pretrained(
                config_dict["base_model"], 
                truncation=config_dict.get("truncation", False),
                max_sequence_length=config_dict.get("max_sequence_length", None)
            )

    model.train()
    processors = model.processors()

    # === Dataset ===
    dataset = build_datasets(config_dict)
    
    trainer_callbacks = []
    if os.environ.get('ENABLE_NSYS') == '1' and not os.environ.get('ENABLE_BENCHY') == '1':
        trainer_callbacks.append(NvtxAnnotationCallback())
    
    trainer = MultimodalTrainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForMultimodal(
                tokenizer=tokenizer, 
                modality_processors=processors,
                modality_loaders=modalities_loader,
                tokenizer_type=config_dict["tokenizer_type"],
                attachment_token_idx=attachment_token_idx,
                use_2d_position_ids=config_dict.get("use_2d_position_ids", False),
            ),
            train_dataset=dataset,
            training_mode=TRAINING_MAPPING[config_dict["training_mode"]],
            pytorch_profiler_config=config_dict.get("pytorch_profiler", None),
            callbacks=trainer_callbacks,
    )

    # === Weights & Biases ===
    wandb_run = None

    wandb_dir_env = os.environ.get("WANDB_DIR", "").strip()
    run_name = training_args.run_name or config_dict["training_args"]["run_name"]

    wandb_state_dir = None
    if wandb_dir_env:
        wandb_state_dir = Path(wandb_dir_env) / "state" / run_name
        try:
            wandb_state_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"W&B state dir (via $WANDB_DIR): {wandb_state_dir}")
        except Exception as e:
            logger.warning(f"Could not create W&B state dir at {wandb_state_dir}: {e}")
            wandb_state_dir = None  # fall back below

    # fallback to output_dir if WANDB_DIR not set or not usable.
    if wandb_state_dir is None:
        if output_dir_path:
            wandb_state_dir = output_dir_path
            logger.info(f"W&B state dir fallback: {wandb_state_dir}")
        else:
            wandb_state_dir = Path.cwd()
            logger.info(f"W&B state dir fallback to CWD: {wandb_state_dir}")

    wandb_id_file = wandb_state_dir / "wandb_run_id.txt"

    if torch.distributed.get_rank() == 0:
        # only attempt to append to the SAME run when resume_flag==true
        reuse_existing_run = resume_flag

        existing_id = None
        if reuse_existing_run and wandb_id_file.exists():
            try:
                existing_id = wandb_id_file.read_text().strip() or None
            except Exception as e:
                logger.warning(f"Could not read {wandb_id_file}: {e}")

        wandb_kwargs = dict(
            project="MultiMeditron",
            config=config_dict,
            name=run_name,
        )
        if reuse_existing_run and existing_id:
            wandb_kwargs.update(id=existing_id, resume="allow")

        wandb_run = wandb.init(**wandb_kwargs)

        # always persist the current id so a later run with resume=true can append.
        try:
            wandb_id_file.write_text(wandb_run.id)
        except Exception as e:
            logger.warning(f"Could not write {wandb_id_file}: {e}")

        # attach deepspeed config
        import json
        with open(config_dict["training_args"]["deepspeed"], "r") as ds_file:
            deepspeed_config = json.load(ds_file)
        wandb_run.config.update({"deepspeed_config": deepspeed_config})

    # === Train (resume or fresh) ===
    if resume_ckpt is not None:
        logger.info(f"Training: resuming from checkpoint: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        logger.info("Training: starting fresh (no resume_from_checkpoint).")
        trainer.train()
    
    if torch.distributed.get_rank() == 0 and wandb_run is not None:
        wandb_run.finish()
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
