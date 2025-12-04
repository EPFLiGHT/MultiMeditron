from multimeditron.cli import EPILOG, main_cli
from multimeditron.model.model import MultimodalConfig, MultiModalModelForCausalLM, bootstrap
from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.train.trainer import MultimodalTrainer, TRAINING_MAPPING
from multimeditron.profiling import NvtxAnnotationCallback
from transformers import AutoTokenizer, TrainingArguments
from datasets import concatenate_datasets, load_dataset, load_from_disk
from multimeditron.model.modalities import AutoModality
from multimeditron.dataset.loader import AutoModalityLoader
from multimeditron.model.model import MultiModalModelForCausalLM, MultimodalConfig, ChatTemplate
from tqdm import tqdm as _tqdm
from PIL import PngImagePlugin
from datasets import config as datasets_config
from pathlib import Path

import deepspeed
import torch
import os
import yaml
import wandb
import multiprocessing
import click
import logging
import json

logger = logging.getLogger(__name__)

PngImagePlugin.MAX_TEXT_CHUNK = 2 ** 30

def is_dataset_folder(folder: str) -> bool:
    return os.path.exists(os.path.join(folder, datasets_config.DATASET_INFO_FILENAME)) and \
        os.path.exists(os.path.join(folder, datasets_config.DATASET_STATE_JSON_FILENAME))

def is_jsonl(path: str) -> bool:
    filename, extension = os.path.splitext(path)
    return extension == ".jsonl"

def is_main_process() -> bool:
    # safe main-process check for DDP/torchrun
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0

def build_datasets(config):
    packed_datasets = []

    # use env vars set by torchrun
    rank = int(os.environ.get("RANK", "0"))

    # give each process fair slice of CPUs (per node)
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
    
    # Disable randomness
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    training_args = TrainingArguments(**config_dict["training_args"])

    # === Tokenizer === 
    tokenizer = AutoTokenizer.from_pretrained(config_dict["base_llm"], padding_side='right', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    chat_template = ChatTemplate.from_name(config_dict["tokenizer_type"])

    special_tokens_list = list(chat_template.special_tokens.values())

    special_tokens_list.append(config_dict["attachment_token"])

    special_tokens = {'additional_special_tokens': special_tokens_list}
    tokenizer.add_special_tokens(special_tokens)
    
    # Create a model
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
            model = bootstrap(config_dict, tokenizer, modalities_config)
        else:
            # load starting weights from base_model (hub id or local checkpoint dir).
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
                chat_template=chat_template,
                attachment_token=config_dict["attachment_token"],
                use_2d_position_ids=config_dict.get("use_2d_position_ids", False),
            ),
            train_dataset=dataset,
            training_mode=TRAINING_MAPPING[config_dict["training_mode"]],
            pytorch_profiler_config=config_dict.get("pytorch_profiler", None),
            callbacks=trainer_callbacks,
    )

    # === Weights & Biases ===
    wandb_run = None
    run_name = training_args.run_name or config_dict["training_args"]["run_name"]
    
    # get resume flag and wandb run id from config
    wandb_run_id = config_dict.get("wandb_run_id", None)  # string or None
    resume_flag = bool(config_dict.get("resume_from_checkpoint", False))

    if is_main_process():
        wandb_kwargs = dict(
            project="MultiMeditron",
            config=config_dict,
            name=run_name,
        )
        
        if wandb_run_id and resume_flag:
            wandb_kwargs.update(id=str(wandb_run_id), resume="allow")
        elif wandb_run_id:
            # allow attaching to a fixed id even without checkpoint resume if user wants
            wandb_kwargs.update(id=str(wandb_run_id))

        wandb_run = wandb.init(**wandb_kwargs)

        # attach deepspeed config
        with open(config_dict["training_args"]["deepspeed"], "r") as ds_file:
            deepspeed_config = json.load(ds_file)
        wandb_run.config.update({"deepspeed_config": deepspeed_config})

    # === Train (resume or fresh) ===
    if resume_flag:
        # always resume from the user-provided base_model checkpoint path
        resume_ckpt = config_dict.get("base_model", None)
        logger.info(f"Training: resuming from checkpoint provided in config.base_model: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        logger.info("Training: starting fresh (no resume_from_checkpoint).")
        trainer.train()
    
    if is_main_process() and wandb_run is not None:
        wandb_run.finish()
    
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
