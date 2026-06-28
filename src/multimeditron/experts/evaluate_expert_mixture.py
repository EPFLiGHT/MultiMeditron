#!/usr/bin/env python
"""Evaluate frozen mixtures of pretrained domain experts."""


import argparse
import csv
import os
from io import BytesIO
from pathlib import Path

from multimeditron.experts.evaluation_pipeline.build_benchmarks import build_benchmarks_from_names

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import VisionTextDualEncoderModel


EXPERT_ROOT = None
DEFAULT_RESULTS_PATH = Path("src/multimeditron/experts/logs/expert_mixture_results.csv")

DOMAIN_TO_EXPERT = {
    "ct": "CT_expert",
    "mri": "MRI_expert",
    "skin": "Skin_expert",
    "ophthalmology": "Ophthalmology_expert",
    "ultrasound": "US_expert",
    "xray": "XR_expert",
}

SMOKE_LIMIT_ENV = {
    "ct": ("CT_MAX_TRAIN_EXAMPLES", "CT_MAX_TEST_EXAMPLES"),
    "mri": ("MRI_MAX_TRAIN_EXAMPLES", "MRI_MAX_TEST_EXAMPLES"),
    "skin": ("SKIN_INTEGRATED_MAX_TRAIN_EXAMPLES", "SKIN_INTEGRATED_MAX_TEST_EXAMPLES"),
    "ophthalmology": ("OPHTH_MAX_TRAIN_EXAMPLES", "OPHTH_MAX_TEST_EXAMPLES"),
    "ultrasound": ("ULTRASOUND_MAX_TRAIN_EXAMPLES", "ULTRASOUND_MAX_TEST_EXAMPLES"),
    "xray": ("XRAY_MAX_TRAIN_EXAMPLES", "XRAY_MAX_TEST_EXAMPLES"),
}


class FrozenExpertMixture(nn.Module):
    """Image encoder that fuses normalized embeddings from several CLIP experts."""

    def __init__(self, expert_paths, fusion):
        super().__init__()
        if fusion not in {"concat", "mean"}:
            raise ValueError(f"Unsupported fusion: {fusion}")

        self.fusion = fusion
        self.expert_names = [path.name for path in expert_paths]
        self.experts = nn.ModuleList(
            VisionTextDualEncoderModel.from_pretrained(str(path))
            for path in expert_paths
        )
        for expert in self.experts:
            expert.eval()
            for parameter in expert.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _project(expert, pixel_values):
        vision_outputs = expert.vision_model(
            pixel_values=pixel_values, return_dict=True
        )
        pooled = vision_outputs.pooler_output
        if pooled is None:
            raise RuntimeError("vision_model returned no pooler_output")
        embedding = expert.visual_projection(pooled)
        return embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _fuse(self, embeddings):
        if self.fusion == "concat":
            fused = torch.cat(embeddings, dim=-1)
        else:
            fused = torch.stack(embeddings, dim=0).mean(dim=0)
        return fused / fused.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def encode_image_path_embeddings(self, img_path):
        from multimeditron.experts.evaluation_pipeline.load_from_clip import img_transform

        device = next(self.parameters()).device
        pixel_values = torch.stack([img_transform(img_path)]).to(device)
        return [self._project(expert, pixel_values) for expert in self.experts]

    @torch.no_grad()
    def encode_image_path(self, img_path):
        embeddings = self.encode_image_path_embeddings(img_path)
        return self._fuse(embeddings)[0].cpu()

    @torch.no_grad()
    def encode_image_bytes_embeddings(self, img_bytes):
        from multimeditron.experts.evaluation_pipeline.load_from_clip import image_processor

        device = next(self.parameters()).device
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        pixel_values = image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].to(device)
        return [self._project(expert, pixel_values) for expert in self.experts]

    @torch.no_grad()
    def encode_image_bytes(self, img_bytes):
        embeddings = self.encode_image_bytes_embeddings(img_bytes)
        return self._fuse(embeddings)[0].cpu()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen mixtures of pretrained domain experts."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DOMAIN_TO_EXPERT),
        choices=list(DOMAIN_TO_EXPERT),
        help="Benchmarks to evaluate.",
    )
    parser.add_argument(
        "--experts",
        nargs="+",
        default=list(DOMAIN_TO_EXPERT.values()),
        help="Expert checkpoint folder names, relative to --expert_root.",
    )
    parser.add_argument(
        "--expert_root",
        type=Path,
        default=None,
        required=True,
        help="Directory containing *_expert checkpoint folders.",
    )
    parser.add_argument(
        "--fusion",
        choices=["concat", "mean"],
        default="concat",
        help="How to fuse normalized expert embeddings.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Where to write scores.",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Optional per-domain train cap for a quick smoke test.",
    )
    parser.add_argument(
        "--max_test_examples",
        type=int,
        default=None,
        help="Optional per-domain test cap for a quick smoke test.",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Recompute embeddings instead of reusing benchmark cache files.",
    )
    parser.add_argument(
        "--mlp_folds",
        type=int,
        default=None,
        help="Optional number of CV folds for the MLP grid search.",
    )
    return parser.parse_args()


def apply_example_caps(domains, max_train, max_test):
    for domain in domains:
        train_env, test_env = SMOKE_LIMIT_ENV[domain]
        if max_train is not None:
            os.environ[train_env] = str(max_train)
        if max_test is not None:
            os.environ[test_env] = str(max_test)


def build_model_name(args, expert_paths):
    cap_parts = []
    if args.max_train_examples is not None:
        cap_parts.append(f"train{args.max_train_examples}")
    if args.max_test_examples is not None:
        cap_parts.append(f"test{args.max_test_examples}")
    if args.mlp_folds is not None:
        cap_parts.append(f"k{args.mlp_folds}")
    cap_suffix = "_".join(cap_parts) if cap_parts else "full"
    expert_suffix = "_".join(path.name for path in expert_paths)
    return f"expert_mixture_{args.fusion}_{cap_suffix}_{expert_suffix}"


def main():
    args = parse_args()
    apply_example_caps(args.domains, args.max_train_examples, args.max_test_examples)

    expert_paths = [args.expert_root / expert_name for expert_name in args.experts]
    missing = [str(path) for path in expert_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expert checkpoint(s): " + ", ".join(missing))

    model_name = build_model_name(args, expert_paths)
    mixture = FrozenExpertMixture(expert_paths=expert_paths, fusion=args.fusion)
    benchmarks = build_benchmarks_from_names(args.domains)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    mlp_kwargs = {"k": args.mlp_folds} if args.mlp_folds is not None else None
    for domain, benchmark in zip(args.domains, benchmarks):
        print(f"Evaluating {model_name} on {domain} ({benchmark.__class__.__name__})")
        score = benchmark.evaluate_model(
            model=mixture,
            model_name=model_name,
            use_cache=not args.no_cache,
            mlp_kwargs=mlp_kwargs,
        )
        print(f"{domain}: {score}")
        rows.append(
            {
                "domain": domain,
                "benchmark": benchmark.__class__.__name__,
                "fusion": args.fusion,
                "experts": "|".join(path.name for path in expert_paths),
                "score": score,
            }
        )

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["domain", "benchmark", "fusion", "experts", "score"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote results to {args.output_csv}")


if __name__ == "__main__":
    main()
