from multimeditron.cli import EPILOG, main_cli
import click


@main_cli.command(epilog=EPILOG)
@click.argument("config_file", type=click.Path(exists=True), required=True)
def train_expert(config_file):
    """
    Run train_clip.py with the specified YAML configuration file.

    Arguments:
        config_file: Path to the YAML configuration file.
    """
    from multimeditron.experts.train_clip import main as train_clip_main
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
                [
                    "nohup",
                    "python",
                    "-m",
                    "multimeditron.cli.experts",
                    "train_expert",
                    config_file,
                ],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,  # To prevent signals from being sent to the child process
            )
            processes.append(process)
            print(f"Started training for {config_file}, logging to {log_file}")

    for process in processes:
        process.wait()
        print(f"Process {process.pid} finished.")


@main_cli.command(epilog=EPILOG)
@click.argument("config_file", type=click.Path(exists=True), required=True)
def train_multidomain_optuna(config_file):
    """
    Run the multidomain CLIP training pipeline with Optuna HPO.

    Benchmarks are loaded from the benchmark_selection field in the config YAML.

    Arguments:
        config_file: Path to the YAML configuration file.
    """
    from multimeditron.experts.train_multidomain_clip import main as train_multidomain_main
    train_multidomain_main(config_file)


@main_cli.command(epilog=EPILOG)
@click.argument("config_files", nargs=-1, type=click.Path(exists=True))
def batch_train_multidomain_optuna(config_files):
    """
    Run train_multidomain_clip.py for each YAML config in parallel with nohup.

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
                [
                    "nohup",
                    "python",
                    "-m",
                    "multimeditron.cli.experts",
                    "train_multidomain_optuna",
                    config_file,
                ],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,
            )
            processes.append(process)
            print(f"Started training for {config_file}, logging to {log_file}")

    for process in processes:
        process.wait()
        print(f"Process {process.pid} finished.")


@main_cli.command(epilog=EPILOG)
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--benchmarks",
    "-b",
    multiple=True,
    help=(
        "Benchmark names to run (repeatable). Default: all available. "
        "Known names: brain_tumor_mri, ct, histopathology, "
        "ophthalmology, scin, skin, ultrasound, xray."
    ),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Path to a JSON file where results will be saved.",
)
def run_benchmark(model_path, benchmarks, output):
    """
    Evaluate a trained CLIP expert model on one or more benchmarks.

    MODEL_PATH: Path to the trained model directory.
    """
    import json
    import sys
    from pathlib import Path

    eval_dir = Path(__file__).resolve().parent.parent / "experts" / "evaluation_pipeline"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    from multimeditron.experts.evaluation_pipeline.build_benchmarks import (
        build_benchmarks_from_names,
    )

    bench_list = build_benchmarks_from_names(list(benchmarks) if benchmarks else None)

    results = {}
    for bench in bench_list:
        bench_name = getattr(bench, "name", bench.__class__.__name__)
        print(f"\n--- Running benchmark: {bench_name} ---")
        metrics = bench.evaluate(model_path)
        results[bench_name] = metrics

    print("\n=== Benchmark Results ===")
    for bench_name, metrics in results.items():
        score = metrics.get("score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {bench_name}: score={score_str}  {metrics}")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output}")


@main_cli.command(epilog=EPILOG)
@click.argument("configs", type=click.Path(exists=True), required=True)
def config_maker_expert(configs):
    """
    Run config_maker.py to make configurations based on datasets and hyperparameter ranges.

    Arguments:
        configs: Path to the YAML file containing dataset mixes and hyperparameter ranges.
    """
    from multimeditron.experts.config_maker import main as config_maker_main
    config_maker_main(configs)
