"""
Confusion-Matrix Evaluation for Skin Disease CLIP Benchmark.

This script evaluates a trained CLIP-style vision–language model on a
multi-class skin disease classification benchmark and visualizes the
resulting confusion matrix. The evaluation is performed using the
SkinDiseaseBenchmark helper, which computes overall accuracy, per-class
accuracy, and the raw confusion matrix based on image–text matching.

The confusion matrix is row-normalized (by true class) to highlight
systematic confusions between disease categories. Long clinical labels
are optionally mapped to shorter aliases for improved readability in
figures. The resulting plot is saved as a publication-ready PNG and is
intended for diagnostic analysis and qualitative comparison between
model variants.

CLI Usage
---------
Basic (defaults reproduce your current hard-coded paths):
    python eval_skin_confusion_matrix.py

Evaluate a specific model + dataset paths and write to a custom output file:
    python eval_skin_confusion_matrix.py \
        --model-dir /path/to/model_dir \
        --train-jsonl /path/to/train.jsonl \
        --test-jsonl /path/to/val.jsonl \
        --image-root /path/to/images \
        --out confusion_matrix.png

Disable short-label mapping (use full labels on axes):
    python eval_skin_confusion_matrix.py \
        --model-dir /path/to/model_dir \
        --no-short-labels

Control figure size:
    python eval_skin_confusion_matrix.py \
        --figsize 7 7 \
        --out cm.png

Arguments
---------
--model-dir PATH
    Path to a CLIP-style model directory or HF repo used by SkinDiseaseBenchmark.

--train-jsonl PATH
    Training manifest (used by SkinDiseaseBenchmark; kept for parity with your helper).

--test-jsonl PATH
    Validation/test manifest to evaluate.

--image-root PATH
    Root directory that contains the images referenced by the JSONL files.

--out PATH
    Output PNG path.

--figsize W H
    Matplotlib figure size in inches.

--short-labels / --no-short-labels
    Enable/disable mapping long clinical labels to short aliases for readability.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from skin_benchmark import SkinDiseaseBenchmark


DEFAULT_MODEL_DIR = (
    "/mloscratch/users/turan/evaluation_clip/models/"
    "combined_dataset_skin_aggressive_training_config_1_"
    "lr5.418484333396616e-05_wd0.20568011432383415_nfrz2"
)
DEFAULT_TRAIN_JSONL = (
    "/mloscratch/users/turan/datasets/skin_diseases_10/train_raw.jsonl"
)
DEFAULT_TEST_JSONL = (
    "/mloscratch/users/turan/datasets/skin_diseases_10/skin10_val_raw.jsonl"
)
DEFAULT_IMAGE_ROOT = "/mloscratch/users/turan/datasets/skin_diseases_10"
DEFAULT_OUT = "confusion_matrix_10_diseases_short_labels.png"
DEFAULT_FIGSIZE = (6.0, 6.0)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SkinDiseaseBenchmark and save a row-normalized confusion matrix PNG."
    )

    p.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR, help="Model directory or repo id."
    )
    p.add_argument(
        "--train-jsonl", default=DEFAULT_TRAIN_JSONL, help="Train JSONL manifest."
    )
    p.add_argument(
        "--test-jsonl", default=DEFAULT_TEST_JSONL, help="Test/val JSONL manifest."
    )
    p.add_argument(
        "--image-root", default=DEFAULT_IMAGE_ROOT, help="Root directory for images."
    )

    p.add_argument("--out", default=DEFAULT_OUT, help="Output PNG path.")
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=list(DEFAULT_FIGSIZE),
        metavar=("W", "H"),
        help="Figure size in inches (width height).",
    )

    p.add_argument(
        "--short-labels",
        dest="short_labels",
        action="store_true",
        help="Use short aliases for long clinical labels (default).",
    )
    p.add_argument(
        "--no-short-labels",
        dest="short_labels",
        action="store_false",
        help="Disable short-label mapping and use full labels.",
    )
    p.set_defaults(short_labels=True)

    return p.parse_args()


def get_short_label_map():
    # Map long labels -> short labels (edit names if your strings differ)
    return {
        "Eczema": "Eczema",
        "Warts Molluscum and other Viral Infections": "Warts",
        "Melanoma": "Melanoma",
        "Atopic Dermatitis": "Dermatitis",
        "Basal Cell Carcinoma (BCC)": "BCC",
        "Melanocytic Nevi (NV)": "Nevi",
        "Benign Keratosis-like Lesions": "BKL",
        "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis",
        "Seborrheic Keratoses and other Benign Tumors": "Seb. Keratoses",
        "Tinea Ringworm Candidiasis and other Fungal Infections": "Fungal",
    }


def row_normalize(cm):
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    # avoid divide-by-zero if a class has 0 samples
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def main():
    args = parse_args()

    model_dir = args.model_dir
    out_path = Path(args.out)

    skin_bench = SkinDiseaseBenchmark(
        train_jsonl=args.train_jsonl,
        test_jsonl=args.test_jsonl,
        image_root=args.image_root,
    )

    acc, per_class_acc, per_class_total, cm, id2label = (
        skin_bench.evaluate_with_confusion(model_dir)
    )

    labels = [id2label[i] for i in range(len(id2label))]

    if args.short_labels:
        short_map = get_short_label_map()
        display_labels = [short_map.get(l, l) for l in labels]
    else:
        display_labels = labels

    cm_norm = row_normalize(np.array(cm))

    plt.figure(figsize=(float(args.figsize[0]), float(args.figsize[1])))
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar()

    tick_marks = np.arange(len(display_labels))
    plt.xticks(tick_marks, display_labels, rotation=45, ha="right")
    plt.yticks(tick_marks, display_labels)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion matrix (accuracy={acc:.2f})")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved confusion matrix to {out_path}")

    # Optional: print a short summary for logs
    print(f"Accuracy: {acc:.4f}")
    print("Per-class accuracy (label: acc, total):")
    for i, lab in enumerate(labels):
        print(f"  {lab}: {per_class_acc[i]:.4f} (n={per_class_total[i]})")


if __name__ == "__main__":
    main()
