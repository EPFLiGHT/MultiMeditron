# MultiMeditron

MultiMeditron pitch, link to paper on arxiv, etc

## Install Dependencies

Build and run our Docker image.
All scripts assume execution from the repository root inside the container.

```
docker build -t project-name -f docker/Dockerfile .
docker run --gpus all -it \
  -v $(pwd):/workspace \
  project-name
```

If Docker is not used, install dependencies manually:
```
pip install -r requirements.txt
```

Note: Maybe it would be nice to offer this as a python package?
Do we already have `pyproject.toml`? Then the instructions here could be `pip install -e .`

## Running our Code

Model checkpoints are published on huggingface. To run download a checkpoint and run generate a reply, use this `generate` helper script. 
```
generate.sh examples/sample_input.?
```

## Reproduce the Paper

All experiments are configured using Hydra. Configuration files are stored in the `cookbooks/` directory.

- The **main recipe**, `cookbooks/main.yaml` represents the final model configuration reported in the paper.
- **Ablation recipes** live in `cookbooks/ablations/`.
- Evaluation-specific settings live under `cookbooks/eval/`.

Before you can run through the training, you need to download our dataset via
```
download.sh
```

This main training writes checkpoints to `checkpoints/main/`. You can use our `main.yaml` configuration to reproduce the training run of the MultiMeditron paper, or provide your own configuration.
```bash
bash scripts/train.sh cookbooks/main.yaml
```

Ablations are designed to execute a well defined set of ablations and write write checkpoints to a separate subdirectory `checkpoints/ablations/<ablation_name>/`
```bash
bash scripts/ablate.sh
```

Evaluation is handled by a single entry point:
```bash
bash scripts/eval.sh
```

By default, this script:
 - Finds the latest checkpoint for the main model and all ablations
 - Runs all available benchmarks
 - Saves raw evaluation outputs to `data/eval/`

Evaluation does not produce plots or tables directly. It only generates structured data.
Analysis and visualization are intentionally separated from evaluation.

Aggregate and post-process results:
```bash
python scripts/analyze.py
```

Generate plots and figures:
```bash
python scripts/plot.py
```

All plots should be generated solely from files in `data/eval`, ensuring full reproducibility.

## The Data Pipeline

You can download preprocessed data via `download.sh`. This will generate ready-to-use data formatted to be compatible with our training code.

Alternatively you can run `download_raw.sh` and `python preprocess.py` to download and preprocess the data yourself. 
The preprocessing uses thirdparty LLM tools:
 - it requires an OpenAI API key in an env variable
 - it is not fully deterministic. The data produced by `download.sh` and `download_raw.sh && python preprocess.py` may differ significantly due to changes in thirdparty APIs.

## Extend the Paper

MultiMeditron mini pitch v2. It's designed to be reproducible, extensible, modular.
You can for example write your own modality projectors.
Point to interesting code files.
Point to API docs.

etc.
