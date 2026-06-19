#!/usr/bin/env python
"""Merge compatible expert checkpoints by averaging their weights."""


import argparse
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


EXPERT_ROOT = Path("/lightscratch/users/nemo/models")
DEFAULT_EXPERTS = (
    "CT_expert",
    "MRI_expert",
    "Ophthalmology_expert",
    "Skin_expert",
    "US_expert",
    "XR_expert",
)
HF_WEIGHT_FILENAMES = {"model.safetensors", "pytorch_model.bin"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average weights from compatible expert checkpoints."
    )
    parser.add_argument(
        "--expert_root",
        type=Path,
        default=EXPERT_ROOT,
        help="Directory containing expert checkpoint folders.",
    )
    parser.add_argument(
        "--experts",
        nargs="+",
        default=list(DEFAULT_EXPERTS),
        help="Expert checkpoint folder names, relative to --expert_root.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional merge weights, one per expert. Defaults to uniform averaging.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the merged HuggingFace checkpoint will be written.",
    )
    parser.add_argument(
        "--reference_expert",
        default=None,
        help="Expert folder used as source for config/tokenizer/preprocessor files. Defaults to first expert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def normalize_weights(raw_weights, n_experts):
    if raw_weights is None:
        return torch.full((n_experts,), 1.0 / n_experts, dtype=torch.float64)
    if len(raw_weights) != n_experts:
        raise ValueError(
            f"--weights has {len(raw_weights)} values but {n_experts} experts were provided"
        )
    weights = torch.tensor(raw_weights, dtype=torch.float64)
    if torch.any(weights < 0):
        raise ValueError("--weights must be non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("--weights must sum to a positive value")
    return weights / total


def read_config(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_configs(expert_paths):
    def model_signature(config):
        text_config = config.get("text_config", {})
        vision_config = config.get("vision_config", {})
        return {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "projection_dim": config.get("projection_dim"),
            "text": {
                "model_type": text_config.get("model_type"),
                "hidden_size": text_config.get("hidden_size"),
                "intermediate_size": text_config.get("intermediate_size"),
                "num_attention_heads": text_config.get("num_attention_heads"),
                "num_hidden_layers": text_config.get("num_hidden_layers"),
                "type_vocab_size": text_config.get("type_vocab_size"),
                "vocab_size": text_config.get("vocab_size"),
            },
            "vision": {
                "model_type": vision_config.get("model_type"),
                "hidden_size": vision_config.get("hidden_size"),
                "intermediate_size": vision_config.get("intermediate_size"),
                "image_size": vision_config.get("image_size"),
                "num_attention_heads": vision_config.get("num_attention_heads"),
                "num_channels": vision_config.get("num_channels"),
                "num_hidden_layers": vision_config.get("num_hidden_layers"),
                "patch_size": vision_config.get("patch_size"),
            },
        }

    reference_subset = model_signature(read_config(expert_paths[0] / "config.json"))
    for path in expert_paths[1:]:
        subset = model_signature(read_config(path / "config.json"))
        if subset != reference_subset:
            raise ValueError(f"Config mismatch between {expert_paths[0]} and {path}")


def safetensor_metadata(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        return {key: tuple(f.get_tensor(key).shape) for key in f.keys()}


def validate_safetensors(weight_paths):
    reference = safetensor_metadata(weight_paths[0])
    reference_keys = set(reference)
    for path in weight_paths[1:]:
        metadata = safetensor_metadata(path)
        if set(metadata) != reference_keys:
            missing = sorted(reference_keys - set(metadata))
            extra = sorted(set(metadata) - reference_keys)
            raise ValueError(
                f"Tensor keys differ for {path}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        shape_mismatches = [
            key for key, shape in reference.items() if metadata[key] != shape
        ]
        if shape_mismatches:
            sample = ", ".join(shape_mismatches[:5])
            raise ValueError(f"Tensor shapes differ for {path}: {sample}")
    return sorted(reference)


def copy_hf_sidecar_files(reference_path, output_dir):
    for source in reference_path.iterdir():
        if source.name in HF_WEIGHT_FILENAMES:
            continue
        destination = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        elif source.is_file():
            shutil.copy2(source, destination)


def merge_safetensors(weight_paths, merge_weights, output_path):
    keys = validate_safetensors(weight_paths)
    merged = {}
    with ExitStack() as stack:
        handles = [
            stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for path in weight_paths
        ]
        for key in keys:
            tensors = [handle.get_tensor(key) for handle in handles]
            if tensors[0].is_floating_point() or tensors[0].is_complex():
                accumulator = torch.zeros_like(tensors[0], dtype=torch.float64)
                for weight, tensor in zip(merge_weights, tensors, strict=True):
                    accumulator += tensor.to(torch.float64) * float(weight)
                merged[key] = accumulator.to(dtype=tensors[0].dtype)
            else:
                merged[key] = tensors[0]
    save_file(merged, output_path)


def main():
    args = parse_args()
    expert_paths = [args.expert_root / expert for expert in args.experts]
    missing = [str(path) for path in expert_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expert checkpoint(s): " + ", ".join(missing))

    reference_name = args.reference_expert or args.experts[0]
    reference_path = args.expert_root / reference_name
    if reference_path not in expert_paths:
        raise ValueError("--reference_expert must be one of --experts")

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is not empty; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_configs(expert_paths)
    weight_paths = [path / "model.safetensors" for path in expert_paths]
    missing_weights = [str(path) for path in weight_paths if not path.exists()]
    if missing_weights:
        raise FileNotFoundError(
            "Missing model.safetensors file(s): " + ", ".join(missing_weights)
        )

    merge_weights = normalize_weights(args.weights, len(expert_paths))
    copy_hf_sidecar_files(reference_path, output_dir)
    merge_safetensors(weight_paths, merge_weights, output_dir / "model.safetensors")

    print("Merged experts:")
    for expert, weight in zip(args.experts, merge_weights.tolist(), strict=True):
        print(f"  {expert}: {weight:.6f}")
    print(f"Wrote merged checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
