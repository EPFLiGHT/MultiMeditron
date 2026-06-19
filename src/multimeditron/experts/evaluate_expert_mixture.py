#!/usr/bin/env python
"""Evaluate frozen mixtures of pretrained domain experts."""


import argparse
import csv
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import VisionTextDualEncoderModel


EXPERT_ROOT = Path("/lightscratch/users/nemo/models")
DEFAULT_RESULTS_PATH = Path("src/multimeditron/experts/logs/expert_mixture_results.csv")
DEFAULT_MANIFEST_ROOT = Path("benchmark_splits/multimediset")

DOMAIN_TO_EXPERT = {
    "ct": "CT_expert",
    "mri": "MRI_expert",
    "skin": "Skin_expert",
    "ophthalmology": "Ophthalmology_expert",
    "ultrasound": "US_expert",
    "xray": "XR_expert",
}

ROUTER_DOMAIN_TO_EXPERT = {
    "ct": "CT_expert",
    "mri": "MRI_expert",
    "skin": "Skin_expert",
    "eye": "Ophthalmology_expert",
    "ultrasound": "US_expert",
    "xray": "XR_expert",
}

ROUTER_DOMAIN_TO_MANIFEST = {
    "ct": "ct",
    "mri": "mri",
    "skin": "skin",
    "eye": "eye",
    "ultrasound": "ultrasound",
    "xray": "xray",
}

SMOKE_LIMIT_ENV = {
    "ct": ("CT_MAX_TRAIN_EXAMPLES", "CT_MAX_TEST_EXAMPLES"),
    "mri": ("MRI_MAX_TRAIN_EXAMPLES", "MRI_MAX_TEST_EXAMPLES"),
    "skin": ("SKIN_INTEGRATED_MAX_TRAIN_EXAMPLES", "SKIN_INTEGRATED_MAX_TEST_EXAMPLES"),
    "ophthalmology": ("OPHTH_MAX_TRAIN_EXAMPLES", "OPHTH_MAX_TEST_EXAMPLES"),
    "ultrasound": ("ULTRASOUND_MAX_TRAIN_EXAMPLES", "ULTRASOUND_MAX_TEST_EXAMPLES"),
    "xray": ("XRAY_MAX_TRAIN_EXAMPLES", "XRAY_MAX_TEST_EXAMPLES"),
}


def _add_eval_pipeline_to_path():
    eval_dir = Path(__file__).resolve().parent / "evaluation_pipeline"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


class FrozenExpertMixture(nn.Module):
    """Image encoder that fuses normalized embeddings from several CLIP experts."""

    def __init__(self, expert_paths, fusion):
        super().__init__()
        if fusion not in {"concat", "mean", "routed_concat", "routed_mean"}:
            raise ValueError(f"Unsupported fusion: {fusion}")

        self.fusion = fusion
        self.expert_names = [path.name for path in expert_paths]
        self.router = None
        self.router_classes_ = None
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

    def _router_features(self, embeddings):
        return torch.cat(embeddings, dim=-1).detach().cpu().numpy()

    def _router_weights(self, embeddings):
        if self.router is None or self.router_classes_ is None:
            raise RuntimeError(f"{self.fusion} requires a trained router")

        probabilities = self.router.predict_proba(self._router_features(embeddings))[0]
        weights = torch.zeros(
            len(self.expert_names),
            device=embeddings[0].device,
            dtype=embeddings[0].dtype,
        )
        for router_class, probability in zip(self.router_classes_, probabilities):
            expert_name = ROUTER_DOMAIN_TO_EXPERT[str(router_class)]
            if expert_name in self.expert_names:
                weights[self.expert_names.index(expert_name)] = float(probability)
        if weights.sum() <= 0:
            weights.fill_(1.0 / len(weights))
        else:
            weights = weights / weights.sum()
        return weights

    def _fuse(self, embeddings):
        if self.fusion == "concat":
            fused = torch.cat(embeddings, dim=-1)
        elif self.fusion == "mean":
            fused = torch.stack(embeddings, dim=0).mean(dim=0)
        else:
            weights = self._router_weights(embeddings)
            if self.fusion == "routed_concat":
                fused = torch.cat(
                    [
                        weight * embedding
                        for weight, embedding in zip(weights, embeddings)
                    ],
                    dim=-1,
                )
            else:
                fused = torch.stack(
                    [
                        weight * embedding
                        for weight, embedding in zip(weights, embeddings)
                    ],
                    dim=0,
                ).sum(dim=0)
        return fused / fused.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def encode_image_path_embeddings(self, img_path):
        from load_from_clip import img_transform

        device = next(self.parameters()).device
        pixel_values = torch.stack([img_transform(img_path)]).to(device)
        return [self._project(expert, pixel_values) for expert in self.experts]

    @torch.no_grad()
    def encode_image_path(self, img_path):
        embeddings = self.encode_image_path_embeddings(img_path)
        return self._fuse(embeddings)[0].cpu()

    @torch.no_grad()
    def encode_image_bytes_embeddings(self, img_bytes):
        from load_from_clip import image_processor

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

    def fit_router(
        self,
        manifest_root,
        examples_per_domain,
    ):
        from evaluation_pipeline.benchmark_classification.multimediset_manifest import (
            _first_image_bytes,
            _first_image_path,
            _get_source_row,
            read_manifest,
            sample_records,
        )

        features = []
        labels = []
        for domain, manifest_domain in ROUTER_DOMAIN_TO_MANIFEST.items():
            manifest_path = manifest_root / manifest_domain / "train_model.jsonl"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing router manifest: {manifest_path}")

            records = sample_records(
                read_manifest(manifest_path), examples_per_domain, seed=42
            )
            for record in tqdm(records, desc=f"router-{domain}"):
                row = _get_source_row(record)
                image_bytes = _first_image_bytes(row)
                if image_bytes is not None:
                    embeddings = self.encode_image_bytes_embeddings(image_bytes)
                else:
                    image_path = _first_image_path(row, Path(record["source_root"]))
                    if image_path is None:
                        continue
                    embeddings = self.encode_image_path_embeddings(str(image_path))
                features.append(self._router_features(embeddings)[0])
                labels.append(domain)

        if len(set(labels)) < 2:
            raise ValueError("Router training needs at least two domains")

        X = np.stack(features)
        y = np.array(labels)
        router = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        router.fit(X, y)
        train_accuracy = accuracy_score(y, router.predict(X))
        print(f"Router train accuracy: {train_accuracy:.4f} on {len(y)} examples")
        self.router = router
        self.router_classes_ = router.classes_


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
        default=EXPERT_ROOT,
        help="Directory containing *_expert checkpoint folders.",
    )
    parser.add_argument(
        "--fusion",
        choices=["concat", "mean", "routed_concat", "routed_mean"],
        default="concat",
        help="How to fuse normalized expert embeddings.",
    )
    parser.add_argument(
        "--router_manifest_root",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT,
        help="MultiMediset manifest root used to train the domain router.",
    )
    parser.add_argument(
        "--router_train_examples_per_domain",
        type=int,
        default=500,
        help="Number of train_model manifest records sampled per domain to train the router.",
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
    if args.fusion.startswith("routed_"):
        cap_parts.append(f"router{args.router_train_examples_per_domain}")
    cap_suffix = "_".join(cap_parts) if cap_parts else "full"
    expert_suffix = "_".join(path.name for path in expert_paths)
    return f"expert_mixture_{args.fusion}_{cap_suffix}_{expert_suffix}"


def main():
    args = parse_args()
    _add_eval_pipeline_to_path()
    apply_example_caps(args.domains, args.max_train_examples, args.max_test_examples)

    from evaluation_pipeline.build_benchmarks import build_benchmarks_from_names

    expert_paths = [args.expert_root / expert_name for expert_name in args.experts]
    missing = [str(path) for path in expert_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expert checkpoint(s): " + ", ".join(missing))

    model_name = build_model_name(args, expert_paths)
    mixture = FrozenExpertMixture(expert_paths=expert_paths, fusion=args.fusion)
    if args.fusion.startswith("routed_"):
        mixture.fit_router(
            manifest_root=args.router_manifest_root,
            examples_per_domain=args.router_train_examples_per_domain,
        )
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
