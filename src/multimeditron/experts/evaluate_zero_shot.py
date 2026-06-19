#!/usr/bin/env python
"""Standalone zero-shot classification evaluation for CLIP expert models.

Complements evaluate_expert_baselines.py without replacing it.
Reuses precomputed image embeddings from the MLP benchmark cache when available.

Usage:
    python src/multimeditron/experts/evaluate_zero_shot.py \
        --model_path src/multimeditron/experts/models/Merged_expert_uniform \
        --output_csv src/multimeditron/experts/logs/zero_shot_results.csv

    # Single domain
    python ... --domains ct mri

    # Smoke test with size limits
    python ... --max_test_examples 100
"""


import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from transformers import AutoTokenizer, VisionTextDualEncoderModel


# ---------------------------------------------------------------------------
# Path setup (mirrors evaluate_expert_baselines.py)
# ---------------------------------------------------------------------------


def _add_eval_pipeline_to_path():
    eval_dir = Path(__file__).resolve().parent / "evaluation_pipeline"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


# ---------------------------------------------------------------------------
# Domain configuration: labels + text template
# Labels must match the index ordering used by each ClassificationBenchmark.
# ---------------------------------------------------------------------------

# X-ray labels (15 classes, multi-hot)
_XRAY_LABELS_15 = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "No Finding",
]

# X-ray labels when manifest-based benchmark is used (13 classes, multi-hot)
_XRAY_LABELS_13 = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "effusion",
    "emphysema",
    "fibrosis",
    "hernia",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "no finding",
]

DOMAIN_INFO = {
    "ct": {
        # Labels match CTBenchmark.labels (2 classes, index order must match)
        "labels": ["covid-19 infection", "right lung"],
        "template": "a CT scan showing {}",
        "multilabel": False,
    },
    "mri": {
        # Labels match MRIBenchmark.labels (4 classes, index order must match)
        "labels": ["glioma", "meningioma", "no tumor", "pituitary"],
        "template": "an MRI scan showing {}",
        "multilabel": False,
    },
    "skin": {
        # Labels match SkinBenchmark.labels (10 classes, index order must match)
        "labels": [
            "atopic dermatitis",
            "basal cell carcinoma",
            "benign keratosis like lesions",
            "eczema",
            "melanocytic nevi",
            "melanoma",
            "psoriasis",
            "seborrheic keratoses",
            "tinea ringworm candidiasis",
            "warts molluscum viral infections",
        ],
        "template": "a dermatology photograph showing {}",
        "multilabel": False,
    },
    "ophthalmology": {
        # label_id 0 = diabetic_retinopathy, 1 = normal in the eye manifest.
        # (OphthalmologyBenchmark.labels has the order reversed — the manifest is authoritative.)
        "labels": ["diabetic retinopathy", "normal"],
        "template": "a fundus photograph showing {}",
        "multilabel": False,
    },
    "ultrasound": {
        # 13 classes from 3 mixed datasets (label_ids from benchmark_splits manifest):
        #   0–3: COVID-US-2026 lung findings (COVID, Normal, Other, Pneumonia)
        #   4–6: BUSI breast findings (benign, malignant, normal)
        #   7–12: DDTI thyroid TIRADS grades (tirads_2 … tirads_5)
        "labels": [
            "COVID-19 lung infection",   # 0
            "normal lung",               # 1
            "other lung finding",        # 2
            "pneumonia",                 # 3
            "benign breast tumor",       # 4
            "malignant breast tumor",    # 5
            "normal breast",             # 6
            "TIRADS 2 thyroid nodule",   # 7
            "TIRADS 3 thyroid nodule",   # 8
            "TIRADS 4A thyroid nodule",  # 9
            "TIRADS 4B thyroid nodule",  # 10
            "TIRADS 4C thyroid nodule",  # 11
            "TIRADS 5 thyroid nodule",   # 12
        ],
        "template": "an ultrasound image showing {}",
        "multilabel": False,
    },
    "xray": {
        "labels": _XRAY_LABELS_15,
        "template": "a chest X-ray showing {}",
        "multilabel": True,
    },
}

SMOKE_LIMIT_ENV = {
    "ct": ("CT_MAX_TRAIN_EXAMPLES", "CT_MAX_TEST_EXAMPLES"),
    "mri": ("MRI_MAX_TRAIN_EXAMPLES", "MRI_MAX_TEST_EXAMPLES"),
    "skin": ("SKIN_INTEGRATED_MAX_TRAIN_EXAMPLES", "SKIN_INTEGRATED_MAX_TEST_EXAMPLES"),
    "ophthalmology": ("OPHTH_MAX_TRAIN_EXAMPLES", "OPHTH_MAX_TEST_EXAMPLES"),
    "ultrasound": ("ULTRASOUND_MAX_TRAIN_EXAMPLES", "ULTRASOUND_MAX_TEST_EXAMPLES"),
    "xray": ("XRAY_MAX_TRAIN_EXAMPLES", "XRAY_MAX_TEST_EXAMPLES"),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_path,
    device,
):
    model = VisionTextDualEncoderModel.from_pretrained(model_path)
    model = model.to(device).eval()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (ValueError, OSError):
        text_model_name = model.config.text_config._name_or_path
        print(f"Tokenizer not found in {model_path}, loading from {text_model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        except (ValueError, OSError):
            # Infer a compatible public fallback from the text model architecture
            model_type = getattr(model.config.text_config, "model_type", "")
            fallback = (
                "bert-base-uncased"
                if "bert" in model_type
                else "FacebookAI/roberta-base"
            )
            print(
                f"Could not load tokenizer from {text_model_name}, falling back to {fallback}"
            )
            tokenizer = AutoTokenizer.from_pretrained(fallback)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts,
    device,
    batch_size=64,
):
    """Return L2-normalised text embeddings, shape [n_texts, embed_dim]."""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        embeds = model.get_text_features(**inputs)
        if not isinstance(embeds, torch.Tensor):
            embeds = embeds.pooler_output
        embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def _zero_shot_single_label(
    image_embeds,
    text_embeds,
):
    """Argmax cosine similarity → predicted class index."""
    sims = image_embeds @ text_embeds.T  # [n, n_classes]
    return sims.argmax(dim=-1).numpy()


def _zero_shot_multilabel(
    image_embeds,
    text_embeds,
):
    """Per-class threshold at per-image mean similarity → binary predictions."""
    sims = image_embeds @ text_embeds.T  # [n, n_classes]
    threshold = sims.mean(dim=-1, keepdim=True)
    return (sims > threshold).numpy().astype(int)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metrics_single(preds, labels):
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(labels, preds))
    acc = float(accuracy_score(labels, preds))
    return {
        "score": macro_f1,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "accuracy": acc,
    }


def _metrics_multilabel(preds, labels):
    micro_f1 = float(f1_score(labels, preds, average="micro", zero_division=0))
    hamming_acc = float((preds == labels).mean())
    return {
        "score": micro_f1,
        "micro_f1": micro_f1,
        "hamming_accuracy": hamming_acc,
    }


# ---------------------------------------------------------------------------
# Per-domain evaluation
# ---------------------------------------------------------------------------


def evaluate_domain(
    domain,
    benchmark,
    model,
    tokenizer,
    device,
    model_path,
    use_cache=True,
):
    info = DOMAIN_INFO[domain]
    labels_list = info["labels"]
    template = info["template"]
    multilabel = info["multilabel"]

    # Build text prompts and encode them
    prompts = [template.format(lbl) for lbl in labels_list]
    text_embeds = encode_texts(model, tokenizer, prompts, device)  # [n_classes, d]

    # Reuse the benchmark's existing image embedding infrastructure
    model_name = Path(model_path).name
    test_dataset = benchmark.build_test_dataset(
        model=model,
        model_name=model_name,
        use_cache=use_cache,
    )

    if len(test_dataset) == 0:
        raise RuntimeError(f"[{domain}] test dataset is empty")

    # Collect embeddings and labels from the dataset
    all_embeds = []
    all_labels = []
    for i in range(len(test_dataset)):
        emb, lbl = test_dataset[i]
        all_embeds.append(emb if isinstance(emb, torch.Tensor) else torch.tensor(emb))
        all_labels.append(lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl))

    image_embeds = torch.stack(all_embeds)  # [n, d]
    labels_tensor = torch.stack(all_labels)  # [n] or [n, n_classes]

    # Normalise image embeddings (may already be normalised but harmless)
    image_embeds = F.normalize(image_embeds.float(), dim=-1)

    # For X-ray, detect whether manifest (13 labels) or CSV (15 labels) is used
    if multilabel and labels_tensor.dim() == 2:
        n_label_cols = labels_tensor.shape[1]
        if n_label_cols != len(labels_list):
            if n_label_cols == 13:
                alt_labels = _XRAY_LABELS_13
            else:
                raise ValueError(
                    f"[{domain}] label tensor has {n_label_cols} columns "
                    f"but DOMAIN_INFO defines {len(labels_list)} labels"
                )
            prompts = [template.format(lbl) for lbl in alt_labels]
            text_embeds = encode_texts(model, tokenizer, prompts, device)

    text_embeds = text_embeds.float()

    if multilabel:
        preds = _zero_shot_multilabel(image_embeds, text_embeds)
        gt = labels_tensor.numpy().astype(int)
        return _metrics_multilabel(preds, gt)
    else:
        preds = _zero_shot_single_label(image_embeds, text_embeds)
        gt = labels_tensor.numpy().astype(int)
        return _metrics_single(preds, gt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot classification evaluation for CLIP expert models."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to a VisionTextDualEncoderModel checkpoint.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DOMAIN_INFO),
        choices=list(DOMAIN_INFO),
        help="Domains to evaluate (default: all).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("src/multimeditron/experts/logs/zero_shot_results.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max_test_examples",
        type=int,
        default=None,
        help="Cap test set size per domain (quick smoke test).",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Recompute image embeddings instead of reusing cache.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _add_eval_pipeline_to_path()

    # Apply test size caps via env vars (same mechanism as the MLP eval)
    if args.max_test_examples is not None:
        for domain in args.domains:
            _, test_env = SMOKE_LIMIT_ENV[domain]
            os.environ[test_env] = str(args.max_test_examples)

    from evaluation_pipeline.build_benchmarks import build_benchmarks_from_names

    benchmarks = build_benchmarks_from_names(args.domains)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(str(args.model_path), device)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for domain, benchmark in zip(args.domains, benchmarks):
        print(f"\n[{domain}] running zero-shot evaluation …")
        try:
            result = evaluate_domain(
                domain=domain,
                benchmark=benchmark,
                model=model,
                tokenizer=tokenizer,
                device=device,
                model_path=str(args.model_path),
                use_cache=not args.no_cache,
            )
        except Exception as exc:
            print(f"[{domain}] FAILED: {exc}")
            continue

        print(f"[{domain}] {result}")
        rows.append(
            {
                "domain": domain,
                "benchmark": benchmark.__class__.__name__,
                "model_path": str(args.model_path),
                **result,
            }
        )

    if not rows:
        print("No results — nothing written.")
        return

    fixed_fields = ["domain", "benchmark", "model_path"]
    metric_fields = list(
        dict.fromkeys(k for row in rows for k in row if k not in fixed_fields)
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fixed_fields + metric_fields,
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote results to {args.output_csv}")


if __name__ == "__main__":
    main()
