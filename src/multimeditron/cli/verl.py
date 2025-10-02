from typing import Optional
from multimeditron.cli import EPILOG, CONFIG_PATH, main_cli
from .utils import split_host_port
import yaml
import click
import os
import logging
import ray


logger = logging.getLogger(__name__)

@main_cli.command(epilog=EPILOG, context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to the configuration file(s) in YAML format.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode.")
@click.option("--trust-remote-code/--no-trust-remote-code", default=False, help="Whether to trust remote code when loading models from HuggingFace.")
@click.option("--dryrun", is_flag=True, help="Perform a dry run without executing the training.")
@click.option("--config-out", "-o", type=click.Path(), help="Path to save the final configuration used for training (in YAML format).")
@click.pass_context
def verl(ctx,
         config: Optional[str] = None,
         trust_remote_code: bool = False,
         verbose: bool = False,
         debug: bool = False,
         dryrun: bool = False,
         config_out: Optional[str] = None):
    from hydra import initialize_config_dir, compose

    if config is None:
        with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.2"):
            cfg = compose(config_name="verl_trainer", overrides=ctx.args)
    else:
        config_dir = os.path.dirname(os.path.abspath(config))
        config_name = os.path.basename(config)
        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            cfg = compose(config_name=config_name, overrides=ctx.args)

    if hasattr(cfg.ray, "debug") and cfg.ray.debug is not None:
        if not debug:
            logger.info("Overriding debug mode from command line to configuration file.")
            debug = cfg.ray.debug
        
    if hasattr(cfg.ray, "verbose") and cfg.ray.verbose is not None:
        if not verbose:
            logger.info("Overriding verbose mode from command line to configuration file.")
            verbose = cfg.ray.verbose

    # Save final configuration if needed
    if config_out is not None:
        logger.info(f"Saving final configuration to {config_out}...")
        from omegaconf import OmegaConf
        with open(config_out, "w") as f:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, sort_keys=False)
        logger.info(f"Final configuration saved to {config_out}")
    
    # If dryrun, we just print the configuration and exit
    if dryrun:
        logger.info("Dry run enabled. The training will not be executed.")

    # Setup the trust remote code globally

    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        kwargs = {
            "runtime_env": {
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "INFO" if debug else "WARN",
                    "VLLM_LOGGING_LEVEL": "INFO" if debug else "ERROR",
                },
            },
        }

        if cfg.ray.num_cpus is not None:
            kwargs["num_cpus"] = cfg.ray.num_cpus

        if debug:
            logger.info("Ray debug mode is enabled.")
            kwargs["runtime_env"]["env_vars"]["RAY_DEBUG"] = "1"
            kwargs["runtime_env"]["env_vars"]["RAY_DEBUG_POST_MORTEM"] = "1"
        else:
            logger.info("Ray debug mode is disabled.")
            kwargs["runtime_env"]["env_vars"]["RAY_DEBUG"] = "0"
            kwargs["runtime_env"]["env_vars"]["RAY_DEBUG_POST_MORTEM"] = "0"

        if cfg.ray.dashboard is not None:
            host, port = split_host_port(cfg.ray.dashboard, default_port=8265)
            kwargs["dashboard_host"] = host
            kwargs["dashboard_port"] = port
            kwargs["include_dashboard"] = True
        else:
            kwargs["include_dashboard"] = False

        ray.init(
            **kwargs
        )
    else:
        logger.warning("Ray is already initialized. Skipping ray.init(), ray configuration will be partially ignored.")
        
    from multimeditron.verl import TaskRunner
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(cfg, trust_remote_code=trust_remote_code, verbose=verbose, dryrun=dryrun))

