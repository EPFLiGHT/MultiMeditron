from multimeditron.cli import EPILOG, CONFIG_PATH, main_cli
from datasets import load_dataset
import click
import logging
import os


logger = logging.getLogger(__name__)

@main_cli.command(epilog=EPILOG, context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to the configuration file(s) in YAML format.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def preprocess_ds(ctx, config: str = None, verbose: bool = False):
    """
    Preprocess the dataset according to the configuration file.
    """
    from hydra import initialize_config_dir, compose

    if config is None:
        with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.2"):
            cfg = compose(config_name="preprocess", overrides=ctx.args)
    else:
        config_dir = os.path.dirname(os.path.abspath(config))
        config_name = os.path.basename(config)
        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            cfg = compose(config_name=config_name, overrides=ctx.args)

    if hasattr(cfg, "verbose") and cfg.verbose is not None:
        if not verbose:
            logger.info("Overriding verbose mode from command line to configuration file.")
            verbose = cfg.verbose
    
    # Here you can add more preprocessing logic based on the configuration
    logger.debug(f"Preprocessing with the following configuration: {cfg}")

    # Reset any randomness for reproducibility
    logger.debug("Setting random seeds for reproducibility...")
    import torch
    import torch.multiprocessing as mp

    torch.set_num_threads(1)
    mp.set_sharing_strategy("file_system")
    mp.set_start_method("spawn", force=True)

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Match on the dataset source type
    match cfg.source.type:
        case "hf":
            ds = load_dataset(**cfg.source.kwargs)
            logger.info(f"Loaded dataset from HuggingFace: {cfg.source.kwargs}")

            if cfg.source.split is not None:
                ds = ds[cfg.source.split]
                logger.info(f"Using split: {cfg.source.split}")

        case "jsonl":
            from multimeditron.model.jsonl_generator import JSONLGenerator
            from datasets import Dataset

            logger.info(f"Loaded dataset from JSONL file: {cfg.source.kwargs.path}")
            ds = JSONLGenerator(cfg.source.kwargs.path)
            ds = Dataset.from_generator(lambda: ds)

        case _:
            raise ValueError(f"Unsupported dataset source type: {cfg.source.type}")

    from multimeditron.dataset.preprocessor.registry import run_processors
    if hasattr(cfg, "processes") and cfg.processes is not None:
        ds = run_processors(ds, cfg.num_processes, cfg.processes)

    # Create the base mode with fast tokenizer
    if cfg.tokenizer.enable:
        logger.info(f"Loading the tokenizer from model: {cfg.tokenizer.model}")
        from transformers import AutoTokenizer

        if cfg.tokenizer.model is None:
            raise ValueError("Tokenizer model must be specified if tokenizer is enabled.")

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.model, dtype=torch.bfloat16, use_fast=cfg.tokenizer.use_fast)
        
        logger.debug("Overwriting pad token to eos token.")
        tokenizer.pad_token = tokenizer.eos_token

        # Add special tokens (for attachments). SHOULD NOT BE USED OUTSIDE OF ATTACHMENT CONTEXT
        special_tokens = {'additional_special_tokens': [cfg.tokenizer.attachment_token]}
        tokenizer.add_special_tokens(special_tokens)

        logger.info("Tokenizing the dataset...")
        raise NotImplementedError("Tokenization logic is not implemented yet.")
    
    # Save the preprocessed dataset
    logger.info(f"Saving the preprocessed dataset to {cfg.output}...")
    ds.to_parquet(
        cfg.output,
    )

@main_cli.command(epilog=EPILOG)
@click.argument("dataset")
@click.argument("output", type=click.Path())
@click.argument("model", type=str)
@click.option("--attachment-token", "-a", type=str, default="<|reserved_special_token_0|>", help="Special token to represent attachments in the text.")
@click.option("--registry-type", "-r", type=click.Choice(["path", "hdf5", "wids"]), default="path", help="Type of the dataset registry to use.")
@click.option("--num-processes", "-n", type=int, default=32, help="Number of processes to use for tokenization.")
def preprocess_ds_legacy(dataset,
                  output,
                  model, 
                  attachment_token,
                  registry_type, 
                  num_processes):
    """
    Preprocess and tokenize the dataset for training.
    """
    import torch
    import torch.multiprocessing as mp

    torch.set_num_threads(1)
    mp.set_sharing_strategy("file_system")
    mp.set_start_method("spawn", force=True)

    # Disable randomness
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create the base mode with fast tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model, dtype=torch.bfloat16, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens = {'additional_special_tokens': [attachment_token]}
    tokenizer.add_special_tokens(special_tokens)

    # Create a model
    torch.set_default_dtype(torch.bfloat16)

    # Load the configuration
    print("Saving dataset to", output)
    from multimeditron.dataset.registry.registry import get_registry
    from multimeditron.model.jsonl_generator import JSONLGenerator
    from multimeditron.dataset.preprocessor.modality_preprocessor import ModalityRetriever
    from datasets import Dataset

    registry_builder = get_registry(registry_type)

    if dataset.endswith(".jsonl"):
        base_path = os.path.dirname(dataset)
        with registry_builder(base_path=base_path) as registry:
            ds = JSONLGenerator(dataset)
            ds = Dataset.from_generator(lambda: ds)
    else:
        ds = load_dataset(dataset)

    # processor_wrapper = ModalityRetriever(registry)
    ds = ds.map(
        make_map_fn_legacy("train", os.path.basename(dataset)),
        batched=False,
        writer_batch_size=num_processes,
        num_proc=num_processes,
        with_indices=True
    )

    # ds = ds.map(
    #     processor_wrapper.merge_modality_with_sample,
    #     batched=False,
    #     writer_batch_size=num_processes,
    #     num_proc=num_processes)

    ds.to_parquet(
        output
    )

def make_map_fn_legacy(split, data_source):
    def process_fn(i_data, idx):
        o_data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "assistant",
                    "content": i_data["prompt"],
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": i_data["response"],
            },
            "extra_info": {
                "split": split,
                "index": idx,
            }
        }
        return o_data
    return process_fn
        
