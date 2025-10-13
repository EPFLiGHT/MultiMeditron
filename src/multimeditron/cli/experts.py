from multimeditron.cli import EPILOG, main_cli
from multimeditron.experts.config_maker import main as config_maker_main
from multimeditron.experts.train_clip import main as train_clip_main
import click

@main_cli.command(epilog=EPILOG)
@click.argument("config_file", type=click.Path(exists=True), required=True)
def train_expert(config_file):
    """
    Run train_clip.py with the specified YAML configuration file.
    
    Arguments:
        config_file: Path to the YAML configuration file.
    """
    train_clip_main(config_file)

@main_cli.command(epilog=EPILOG)
@click.argument("config_files", nargs=-1, type=click.Path(exists=True))
def batch_train_expert(config_files):
    """
    Run train_clip.py for each specified YAML configuration file in parallel with nohup.
    
    Arguments:
        config_files: Paths to the YAML configuration files.
    """
    import os
    import subprocess

    processes = []
    for config_file in config_files:
        log_file = f"{os.path.splitext(config_file)[0]}.log"
        with open(log_file, "w") as log_f:
            process = subprocess.Popen(
                ["nohup", "python", "-m", "multimeditron.cli.experts", "train_expert", config_file],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp  # To prevent signals from being sent to the child process
            )
            processes.append(process)
            print(f"Started training for {config_file}, logging to {log_file}")

    for process in processes:
        process.wait()
        print(f"Process {process.pid} finished.")

@main_cli.command(epilog=EPILOG)
@click.argument("configs", type=click.Path(exists=True), required=True)
def config_maker_expert(configs):
    """
    Run config_maker.py to make configurations based on datasets and hyperparameter ranges.
    
    Arguments:
        configs: Path to the YAML file containing dataset mixes and hyperparameter ranges.
    """
    config_maker_main(configs)