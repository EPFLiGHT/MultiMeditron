from multimeditron.cli import EPILOG, CONFIG_PATH, main_cli
from multimeditron.utils import get_torch_dtype
from datasets import load_dataset
import click
import logging
import os


logger = logging.getLogger(__name__)

@main_cli.command(epilog=EPILOG, context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to the configuration file(s) in YAML format.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--head", "-h", type=int, default=None, help="If set only process the first N lines")
@click.option("--display", is_flag=True, help="If set display the dataset instead of saving it. Used with --head.")
@click.pass_context
def preprocess_ds(ctx, config: str = None, verbose: bool = False, head: int = None, display: bool = False):
    """
    Preprocess the dataset according to the configuration file.
    """
    from hydra import initialize_config_dir, compose

    if config is None:
        with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.2"):
            cfg = compose(config_name="preprocess-ds", overrides=ctx.args)
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

        case "jsonl":
            from multimeditron.model.jsonl_generator import JSONLGenerator
            from datasets import Dataset

            logger.info(f"Loaded dataset from JSONL file: {cfg.source.kwargs.path}")
            ds = JSONLGenerator(cfg.source.kwargs.path)
            ds = Dataset.from_generator(lambda: ds)

        case "parquet" | "csv":
            logger.info(f"Loaded dataset from {cfg.source.type} file: {cfg.source.kwargs.path}")
            ds = load_dataset(cfg.source.type, data_files=cfg.source.kwargs.path)

        case _:
            raise ValueError(f"Unsupported dataset source type: {cfg.source.type}")

    # If head is set, only filter first N lines
    if head is not None:
        logger.info(f"Filtering the dataset to the first {head} lines...")
        ds = ds.select(range(head))

    from multimeditron.dataset.preprocessor import run_preprocessors
    if hasattr(cfg, "processes") and cfg.processes is not None:
        ds = run_preprocessors(ds, cfg.num_processes, cfg.processes)

    # Create the base mode with fast tokenizer
    if cfg.tokenizer.enable:
        if cfg.tokenizer.model is None:
            raise ValueError("Tokenizer model must be specified if tokenizer is enabled.")
        if cfg.tokenizer.attachment_token is None:
            raise ValueError("Attachment token must be specified if tokenizer is enabled.")

        # Loading the tokenizer
        logger.info(f"Loading the tokenizer from model: {cfg.tokenizer.model}")
        from transformers import AutoTokenizer

        dtype = get_torch_dtype(cfg.tokenizer.get("dtype", "float32"))
        use_fast = cfg.tokenizer.get("use_fast", True)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.model, dtype=dtype, use_fast=use_fast)
        
        logger.debug("Overwriting pad token to eos token.")
        tokenizer.pad_token = tokenizer.eos_token

        # Add special tokens (for attachments). SHOULD NOT BE USED OUTSIDE OF ATTACHMENT CONTEXT
        special_tokens = {'additional_special_tokens': [cfg.tokenizer.attachment_token]}
        tokenizer.add_special_tokens(special_tokens)

        logger.info("Tokenizing the dataset...")
        logger.warning("This code is not tested and not depended upon, modify as needed. It is just a template.")
        ds = ds.map(
            lambda dt: tokenizer(
                dt[cfg.tokenizer.text_field],
                truncation=True,
                padding="max_length",
                max_length=cfg.tokenizer.max_length,
            ),
            batched=False,
            writer_batch_size=cfg.num_processes,
            num_proc=cfg.num_processes,
        )
    
    # If head is set, display the first N lines and exit
    if display:
        logger.info("Displaying the dataset:")
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(show_header=True, header_style="bold magenta")

        # First iterate to find all columns
        columns = set()
        for item in ds:
            for key in item.keys():
                columns.add(key)
        columns = list(columns)
        columns.sort()
        table.add_column("Id")
        for col in columns:
            table.add_column(col)

        # Then add rows
        for idx, item in enumerate(ds):
            row = [str(idx)]
            for col in columns:
                row.append(str(item.get(col, "")))
            table.add_row(*row)
        console.print(table)
        return
    
    # Save the preprocessed dataset
    logger.info(f"Saving the preprocessed dataset to {cfg.output}...")
    ds.to_parquet(
        cfg.output,
    )