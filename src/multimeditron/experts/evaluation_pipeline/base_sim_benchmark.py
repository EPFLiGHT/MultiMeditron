"""
CLIP Image–Text Alignment Evaluation (Recall@1, 4-Way Forced Choice)

This script evaluates image–text alignment for CLIP-style vision–language models
using a controlled retrieval setting suitable for medical datasets.

Evaluation Protocol
-------------------
For each query image, the model is asked to rank one ground-truth caption
against three randomly sampled negative captions drawn from the same validation
split. A prediction is counted as correct if the ground-truth caption receives
the highest cosine similarity score.

This corresponds to Recall@1 under a 4-candidate forced-choice setup
(chance level = 25%), rather than global retrieval over the full dataset.
The metric is designed to probe fine-grained cross-modal alignment while
controlling difficulty and computational cost.

Procedure
---------
1. Load a CLIP-compatible vision–language model and processor.
2. Perform text-tower and image-tower sanity probes using diverse synthetic
   inputs to detect representation collapse.
3. For each evaluation sample:
   - Use the image as the query.
   - Compare cosine similarity against one positive and three negative captions.
   - Record whether the positive caption ranks first.
4. Report average Recall@1 over all valid samples.

Notes
-----
- Similarities are computed using L2-normalized embeddings and cosine similarity.
- Negatives are randomly sampled per query (seeded for reproducibility).
- Image paths are resolved relative to the JSONL manifest location.
- This evaluation is intended for comparative analysis between models and
  fine-tuned checkpoints.

CLI Usage
---------
Basic (uses defaults from the original script):
    python base_sim_benchmark.py

Evaluate a specific model + dataset and write logs to a custom directory:
    python base_sim_benchmark.py \
        --model finetuned /path/to/model_checkpoint_or_repo \
        --eval-datasets /path/to/eyepacs_val.jsonl \
        --log-dir ./logs

Evaluate multiple models:
    python base_sim_benchmark.py \
        --model base /path/to/base_model \
        --model finetuned /path/to/finetuned_model \
        --eval-datasets /path/to/eyepacs_val.jsonl \
        --log-dir ./logs

Control evaluation size, seed, and device:
    python base_sim_benchmark.py \
        --line-number 2000 \
        --seed 14 \
        --device cuda:0

Arguments
---------
--model NAME PATH
    Add a model to evaluate. Can be repeated. Example:
        --model base /path/to/base --model finetuned /path/to/finetuned

--eval-datasets PATH [PATH ...]
    One or more JSONL dataset manifests to evaluate.

--log-dir PATH
    Directory to write per-model logs.

--line-number N
    Maximum number of JSONL lines to evaluate (default: 1000).

--seed N
    Random seed for negative sampling and torch seed (default: 14).

--device DEVICE
    Force device: "cuda", "cuda:0", or "cpu". If omitted, auto-detect.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image
from transformers import VisionTextDualEncoderProcessor, AutoTokenizer

from load_from_clip import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate CLIP-style image-text alignment via 4-way forced-choice Recall@1."
    )

    p.add_argument(
        "--eval-datasets",
        nargs="+",
        default=[
            "/lightscratch/users/turan/datasets/opthalmology_expert_datasets/eyepacs/eyepacs_val.jsonl",
        ],
        help="One or more JSONL eval dataset paths.",
    )
    p.add_argument(
        "--log-dir",
        default="/lightscratch/users/turan/evaluation_clip/logs",
        help="Directory to write log files.",
    )

    # Repeatable: --model NAME PATH
    p.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        default=[
            ["finetuned_clip_2", "/lightscratch/users/turan/training/models_opthalmology/combined_dataset_opthalmology_fine_tuning_config_2"]
        ],
        help="Add a model as: --model <name> <model_path>. Can be repeated.",
    )

    p.add_argument(
        "--line-number",
        type=int,
        default=1000,
        help="Max number of JSONL lines to evaluate from each dataset.",
    )
    p.add_argument("--seed", type=int, default=14, help="Random seed.")
    p.add_argument(
        "--device",
        default=None,
        help='Device override, e.g. "cuda", "cuda:0", or "cpu". Default: auto.',
    )

    return p.parse_args()


@torch.no_grad()
def get_similarity(text: str, image_path: str, model, processor, device: str) -> float:
    """
    Return cosine similarity between an image and a text according to the given model.
    """
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=[text], images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    image_embeds = outputs.image_embeds  # (1, D)
    text_embeds = outputs.text_embeds    # (1, D)

    a_norm = torch.nn.functional.normalize(image_embeds, dim=1)
    b_norm = torch.nn.functional.normalize(text_embeds, dim=1)

    similarity = torch.matmul(a_norm, b_norm.T)  # (1,1)
    return float(similarity.item())


@torch.no_grad()
def evaluate_model(
    model_name_or_path: str,
    eval_dataset: str,
    *,
    line_number: int,
    device: str,
    seed: int,
    log_fp=None,
) -> float:
    """
    Evaluate a given model on a JSONL dataset using 4-way forced choice Recall@1.
    """
    # small helper to log to both stdout and file
    def log_print(msg: str):
        print(msg)
        if log_fp is not None:
            log_fp.write(msg + "\n")
            log_fp.flush()

    # ensure deterministic negative sampling per call (given seed)
    rng = random.Random(seed)

    clip_model = load_model(model_name_or_path).to(device).eval()
    processor = VisionTextDualEncoderProcessor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # ---- Text tower probe: test 6 diverse texts ----
    probe_texts = [
        "a cat sitting on a couch",
        "retinal fundus image with drusen",
        "an x-ray showing lung consolidation",
        "a photo of a dog running in a field",
        "a patient with diabetic retinopathy",
        "a wooden chair next to a table",
    ]

    probe_inputs = processor(
        text=probe_texts,
        images=[Image.new("RGB", (10, 10))] * 6,
        return_tensors="pt",
        padding=True,
    ).to(device)

    probe_out = clip_model(**probe_inputs)
    text_embs = probe_out.text_embeds  # (6, D)

    pairwise = torch.nn.functional.cosine_similarity(
        text_embs.unsqueeze(1), text_embs.unsqueeze(0), dim=-1
    )

    log_print(f"\n=== Text tower probe for {model_name_or_path} ===")
    log_print("pairwise cosine matrix:")
    for row in pairwise.float().cpu().numpy():
        log_print("   " + " ".join(f"{v:0.3f}" for v in row))

    unique = len(torch.unique(text_embs.cpu(), dim=0))
    log_print(f"unique text embeddings: {unique} / 6\n")

    # ---- Image tower probe: 6 simple synthetic images ----
    probe_images = []
    for i in range(6):
        img = Image.new(
            "RGB",
            (10, 10),
            color=(int((i * 40) % 256), int((i * 80) % 256), int((i * 120) % 256)),
        )
        probe_images.append(img)

    image_probe_inputs = processor(
        text=["dummy"] * 6,  # same text, different images
        images=probe_images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    image_probe_out = clip_model(**image_probe_inputs)
    image_embs = image_probe_out.image_embeds  # (6, D)

    image_pairwise = torch.nn.functional.cosine_similarity(
        image_embs.unsqueeze(1), image_embs.unsqueeze(0), dim=-1
    )

    log_print(f"\n=== Image tower probe for {model_name_or_path} ===")
    log_print("pairwise cosine matrix:")
    for row in image_pairwise.float().cpu().numpy():
        log_print("   " + " ".join(f"{v:0.3f}" for v in row))

    image_unique = len(torch.unique(image_embs.cpu(), dim=0))
    log_print(f"unique image embeddings: {image_unique} / 6\n")

    # ---- load jsonl lines for this dataset ----
    with open(eval_dataset, "r", encoding="utf-8") as file:
        all_lines = file.readlines()

    if not all_lines:
        raise ValueError(f"No lines found in {eval_dataset}")

    N = min(line_number, len(all_lines))
    lines = all_lines[:N]
    base_dir = Path(eval_dataset).parent

    good_guess = 0
    used = 0

    for i, line in enumerate(lines):
        # sample 3 distinct negatives != i
        candidates = list(range(N))
        candidates.remove(i)

        # if N is too small, skip
        if len(candidates) < 3:
            break

        a, b, c = rng.sample(candidates, 3)

        correct_line = json.loads(line)
        a_line = json.loads(lines[a])
        b_line = json.loads(lines[b])
        c_line = json.loads(lines[c])

        texts = [
            correct_line["text"],
            a_line["text"],
            b_line["text"],
            c_line["text"],
        ]

        # ---- resolve image path ----
        rel_path = correct_line["modalities"][0]["value"]
        p = Path(rel_path)
        if not p.is_absolute():
            p = (base_dir / p).resolve()

        if not p.exists():
            # Skip this sample if the image file doesn't exist
            continue

        image_path = str(p)

        model_similarities = []
        for t in texts:
            tokens = tokenizer.encode(t, truncation=True, max_length=500)
            text_value = tokenizer.decode(tokens, skip_special_tokens=True)

            sim = get_similarity(text_value, image_path, clip_model, processor, device)
            model_similarities.append(sim)

        pred_idx = int(torch.tensor(model_similarities).argmax().item())
        if pred_idx == 0:
            good_guess += 1

        used += 1
        if used >= N:
            break

    if used == 0:
        raise ValueError("No valid samples used (all images missing?).")

    acc = good_guess / used
    log_print(f"Used {used} samples, accuracy = {acc:.4f}")
    return acc


def main() -> None:
    args = parse_args()

    # seeds / device
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    clips: List[Tuple[str, str]] = [(name, path) for (name, path) in args.model]
    eval_datasets: List[str] = args.eval_datasets

    for name, model_id in clips:
        model_slug = Path(model_id).name
        log_path = log_dir / f"{model_slug}.txt"

        print(f"\n========== Logging for config: {name} ({model_id}) ==========")
        print(f"Log file: {log_path}")
        print(f"Device: {device}")
        print(f"Line number: {args.line_number}")
        print(f"Seed: {args.seed}")

        with open(log_path, "w", encoding="utf-8") as log_fp:
            for eval_dataset in eval_datasets:
                dataset_slug = Path(eval_dataset).stem
                header = (
                    f"\n\n===== Evaluating model {name} ({model_id}) "
                    f"on dataset {dataset_slug} ({eval_dataset}) ====="
                )
                print(header)
                log_fp.write(header + "\n")
                log_fp.flush()

                acc = evaluate_model(
                    model_id,
                    eval_dataset,
                    line_number=args.line_number,
                    device=device,
                    seed=args.seed,
                    log_fp=log_fp,
                )

                summary = f"{name} accuracy on {dataset_slug}: {acc:.4f}"
                print(summary)
                log_fp.write(summary + "\n")
                log_fp.flush()


if __name__ == "__main__":
    main()
