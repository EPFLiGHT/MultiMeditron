from multimeditron.model.model import MultimodalConfig, MultiModalModelForCausalLM, bootstrap
from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.train.trainer import MultimodalTrainer, TRAINING_MAPPING
from multimeditron.profiling import NvtxAnnotationCallback
from transformers import AutoTokenizer, TrainingArguments
from datasets import concatenate_datasets, load_dataset, load_from_disk
from multimeditron.model.modalities import AutoModality
from multimeditron.dataset.loader import AutoModalityLoader
from multimeditron.model.model import MultiModalModelForCausalLM, MultimodalConfig
from tqdm import tqdm
import torch
import os
import yaml
from PIL import PngImagePlugin
from datasets import config as datasets_config
import wandb
import multiprocessing
import click
from multimeditron.cli import EPILOG, main_cli
import logging

logger = logging.getLogger(__name__)

PngImagePlugin.MAX_TEXT_CHUNK = 2 ** 30

def is_dataset_folder(folder: str) -> bool:
    return os.path.exists(os.path.join(folder, datasets_config.DATASET_INFO_FILENAME)) and \
        os.path.exists(os.path.join(folder, datasets_config.DATASET_STATE_JSON_FILENAME))

def build_datasets(config):
    packed_datasets = []
    
    num_proc = multiprocessing.cpu_count()
    logger.info(f"Detected {num_proc} CPU cores, using all for dataset processing.")

    for ds_config in tqdm(config["datasets"], desc="Concatenating datasets"):
        if is_dataset_folder(ds_config["packed_path"]):
            dataset = load_from_disk(ds_config['packed_path'])
        else:
            dataset = load_dataset(ds_config["packed_path"], num_proc=num_proc)["train"]
        
        packed_datasets.append(dataset)

    ds = concatenate_datasets(packed_datasets).shuffle()

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
    
    # Disable randomness
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    training_args = TrainingArguments(**config_dict["training_args"])
    
    # Create the base model
    tokenizer = AutoTokenizer.from_pretrained(config_dict["base_llm"], padding_side='right', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {'additional_special_tokens': [ATTACHMENT_TOKEN]}
    tokenizer.add_special_tokens(special_tokens)
    
    attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)
    
    # Create a model
    torch.set_default_dtype(torch.bfloat16)
    
    # Get modalities from configuration
    modalities_config = []
    for modality in config_dict["modalities"]:
        modalities_config.append(AutoModality.config_from_dict(modality))

    modalities_loader = dict()
    for loader in config_dict["loaders"]:
        loader_copy = loader.copy()
        loader_type = loader_copy.pop("loader_type")
        modality_type = loader_copy.pop("modality_type")
        modalities_loader[modality_type] = AutoModalityLoader.from_name(loader_type, **loader_copy)

    import deepspeed
    with deepspeed.zero.Init(dtype=torch.bfloat16):
        if config_dict.get("base_model", None) is None:
            model = bootstrap(config_dict, tokenizer, attachment_token_idx, modalities_config)
        else:
            multimodal_config = MultimodalConfig(
                    hidden_size=config_dict["token_size"],
                    vocab_size=len(tokenizer),
                    attachment_token_idx=attachment_token_idx,
                    eos_token_idx=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
                    modalities=modalities_config,
                    llm_path=config_dict["base_llm"],
            )
            model = MultiModalModelForCausalLM.from_pretrained(config_dict["base_model"], config=multimodal_config)
    
    model.train()
    
    processors = model.processors()
    
    # build_datasets uses distributed env (for sharding) initialized in training args
    dataset = build_datasets(config_dict)
    
    trainer_callbacks = []
    if os.environ.get('ENABLE_NSYS') == '1' and not os.environ.get('ENABLE_BENCHY') == '1':  # benchy already launches profiler
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
            ),
            train_dataset=dataset,
            training_mode=TRAINING_MAPPING[config_dict["training_mode"]],
            pytorch_profiler_config=config_dict.get("pytorch_profiler", None),
            callbacks=trainer_callbacks,
    )
    
    if torch.distributed.get_rank() ==0:
        run = wandb.init(project="MultiMeditron", config = config_dict ,name = config_dict["training_args"]["run_name"])
    
        import json
        with open(config_dict["training_args"]["deepspeed"], "r") as ds_file:
            deepspeed_config = json.load(ds_file)
        run.config.update({"deepspeed_config": deepspeed_config})
    
    trainer.train()
    
    if torch.distributed.get_rank() ==0:
        run.finish()
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
