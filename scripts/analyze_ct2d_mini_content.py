#!/usr/bin/env python3
"""Detailed content analysis of the CT2D-glob-mini HuggingFace dataset.

Streams through the dataset in one pass and reports:
  - Actual imaging modalities present (CT, histology, X-ray, MRI, ...)
  - Tissue types / organs covered (for histology)
  - Anatomical regions (for radiology)
  - Pathologies mentioned
  - Caption length distribution
  - Sample captions per modality and per organ/tissue type

Usage:
    python scripts/analyze_ct2d_mini_content.py
    python scripts/analyze_ct2d_mini_content.py --max_examples 20000
    python scripts/analyze_ct2d_mini_content.py --split test
    python scripts/analyze_ct2d_mini_content.py --output_json testing_logs/ct2d_mini_analysis.json
"""


import argparse
import json
from collections import Counter
from pathlib import Path


DATASET_PATH = "/lightscratch/datasets/MultiMediset/general_purpose/CT2D-glob-mini"

# ---------------------------------------------------------------------------
# Keyword taxonomy
# ---------------------------------------------------------------------------

MODALITIES = {
    "CT / scanner": (
        "ct scan",
        "computed tomography",
        "hounsfield",
        "axial ct",
        "coronal ct",
        "sagittal ct",
        "ct image",
        "ct-scan",
    ),
    "histology / pathology": (
        "histolog",
        "hematoxylin",
        "eosin",
        "h&e",
        "microscop",
        "patholog",
        "biopsy",
        "tissue section",
        "slide",
        "stain",
        "cellular",
        "cytolog",
        "cell morpholog",
    ),
    "X-ray / radiograph": (
        "x-ray",
        "radiograph",
        "chest x",
        "plain film",
        "plain radiograph",
        "anteroposterior",
        "posteroanterior",
    ),
    "MRI": (
        "magnetic resonance",
        " mri ",
        "t1-weighted",
        "t2-weighted",
        "flair",
        " t1 ",
        " t2 ",
    ),
    "ultrasound": ("ultrasound", "echograph", "sonograph", "echogenicit"),
    "fundus / ophthalmology": ("fundus", "retina", "optic disc", "macula", "choroid"),
    "skin / dermoscopy": (
        "dermoscop",
        "skin lesion",
        "melanoma",
        "nevus",
        "pigmented lesion",
    ),
    "endoscopy": ("endoscop", "colonoscop", "gastroscop", "laparoscop"),
}

TISSUES = {
    "lung / pulmonary": ("lung", "pulmonar", "bronch", "alveol", "pleura", "airway"),
    "colon / colorectal": (
        "colon",
        "colorectal",
        "rectum",
        "colonic",
        "sigmoid",
        "adenocarcinoma of the colon",
    ),
    "stomach / gastric": ("stomach", "gastric", "gastro", "mucosa of the stomach"),
    "liver / hepatic": (
        "liver",
        "hepat",
        "hepatocellular",
        "bile duct",
        "cholangiocarc",
    ),
    "kidney / renal": ("kidney", "renal", "nephro", "glomerul", "tubular"),
    "prostate": ("prostate", "prostatic"),
    "breast": ("breast", "mammary", "ductal carcinoma", "lobular"),
    "skin": ("skin", "dermis", "epidermis", "melanocyt", "squamous cell"),
    "brain / neural": (
        "brain",
        "cerebral",
        "neuron",
        "glia",
        "astrocyt",
        "oligodendro",
        "neural tissue",
    ),
    "thyroid": ("thyroid", "follicular", "papillary thyroid"),
    "pancreas": ("pancrea", "islet", "acinar"),
    "lymph node / lymphoma": ("lymph node", "lymphoma", "lymphocyt", "germinal center"),
    "bone / marrow": ("bone marrow", "osteocyt", "trabecular bone", "osseous"),
    "cervix / uterus": ("cervix", "cervical", "uterus", "endometri", "squamo-columnar"),
    "bladder": ("bladder", "urothelial", "transitional cell"),
}

PATHOLOGIES = {
    "carcinoma / adenocarcinoma": (
        "carcinoma",
        "adenocarcinoma",
        "malignant",
        "metastasis",
        "metastatic",
    ),
    "tumor (generic)": ("tumor", "neoplasm", "neoplastic", "mass"),
    "glioblastoma / glioma": ("glioblastoma", "glioma", "gbm"),
    "lymphoma": ("lymphoma",),
    "inflammation / infection": (
        "inflammation",
        "inflammatory",
        "infect",
        "abscess",
        "pneumonia",
    ),
    "COVID-19": ("covid", "sars-cov", "coronavirus"),
    "fibrosis": ("fibrosis", "fibrotic"),
    "healthy / normal": ("healthy", "normal", "benign", "unremarkable", "no evidence"),
    "atherosclerosis": ("atherosclerosis", "atherosoma", "atheroma", "plaque"),
    "necrosis": ("necrosis", "necrotic"),
    "dysplasia": ("dysplasia", "dysplastic"),
}

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def first_modality_value(example):
    mods = example.get("modalities") or []
    if not mods:
        return ""
    first = mods[0]
    return str(first.get("value", "") if isinstance(first, dict) else first)


def match_keywords(text, taxonomy):
    return [label for label, kws in taxonomy.items() if any(kw in text for kw in kws)]


def analyze(dataset, max_examples):
    modality_counts = Counter()
    tissue_counts = Counter()
    pathology_counts = Counter()
    caption_lengths = []
    no_text_count = 0
    total = 0

    modality_samples = {}
    tissue_samples = {}

    iterable = (
        dataset
        if max_examples is None
        else dataset.select(range(min(max_examples, len(dataset))))
    )

    for example in iterable:
        text = str(example.get("text") or "")
        text_lower = text.lower()
        total += 1

        if not text_lower.strip():
            no_text_count += 1
            continue

        caption_lengths.append(len(text.split()))

        matched_mod = match_keywords(text_lower, MODALITIES)
        for label in matched_mod:
            modality_counts[label] += 1
            if label not in modality_samples:
                modality_samples[label] = text[:300]
        if not matched_mod:
            modality_counts["(no modality keyword)"] += 1

        for label in match_keywords(text_lower, TISSUES):
            tissue_counts[label] += 1
            if label not in tissue_samples:
                tissue_samples[label] = text[:300]

        for label in match_keywords(text_lower, PATHOLOGIES):
            pathology_counts[label] += 1

    n = len(caption_lengths)
    lengths_sorted = sorted(caption_lengths)
    avg_len = sum(lengths_sorted) / n if n else 0

    return {
        "total_examples": total,
        "no_text_count": no_text_count,
        "caption_length": {
            "mean": round(avg_len, 1),
            "p25": lengths_sorted[n // 4] if n else 0,
            "p50": lengths_sorted[n // 2] if n else 0,
            "p75": lengths_sorted[3 * n // 4] if n else 0,
            "min": lengths_sorted[0] if lengths_sorted else 0,
            "max": lengths_sorted[-1] if lengths_sorted else 0,
        },
        "modalities": dict(modality_counts.most_common()),
        "tissues": dict(tissue_counts.most_common()),
        "pathologies": dict(pathology_counts.most_common()),
        "modality_samples": modality_samples,
        "tissue_samples": tissue_samples,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def pct(count, total):
    return f"{100 * count / total:.1f}%" if total else "0%"


def print_counter(title, counts, total, top_n=20):
    print(f"\n{title}")
    print("-" * 60)
    for label, count in list(counts.items())[:top_n]:
        print(f"  {label:<45} {count:>7}  ({pct(count, total)})")


def print_samples(title, samples):
    print(f"\n{title}")
    print("-" * 60)
    for label, text in samples.items():
        print(f"\n[{label}]")
        print(f"  {text}")


def print_report(result):
    total = result["total_examples"]
    print("=" * 60)
    print("  CT2D-glob-mini Content Analysis")
    print("=" * 60)
    print(f"  Total examples  : {total:,}")
    print(f"  No text field   : {result['no_text_count']:,}")
    cl = result["caption_length"]
    print(
        f"  Caption length  : mean={cl['mean']} words  "
        f"p25={cl['p25']}  p50={cl['p50']}  p75={cl['p75']}  max={cl['max']}"
    )

    print_counter("IMAGING MODALITIES", result["modalities"], total)
    print_counter("TISSUE / ORGAN TYPES", result["tissues"], total)
    print_counter("PATHOLOGIES MENTIONED", result["pathologies"], total)
    print_samples("SAMPLE CAPTIONS BY MODALITY", result["modality_samples"])
    print_samples("SAMPLE CAPTIONS BY TISSUE TYPE", result["tissue_samples"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze CT2D-glob-mini dataset content."
    )
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Dataset split to analyze (default: train)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Cap examples for quick sampling (default: all ~67k)",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Also write raw counts to a JSON file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from datasets import load_from_disk
    except ImportError as e:
        raise RuntimeError("Please install datasets: pip install datasets") from e

    print(f"Loading {args.dataset_path} (split={args.split})...")
    ds = load_from_disk(args.dataset_path)
    split_ds = ds[args.split]

    scope = (
        f"{args.max_examples:,} examples"
        if args.max_examples
        else f"all {len(split_ds):,} examples"
    )
    print(f"Analyzing {scope}...")

    result = analyze(split_ds, args.max_examples)
    print_report(result)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        json_result = {
            k: v
            for k, v in result.items()
            if k not in ("modality_samples", "tissue_samples")
        }
        args.output_json.write_text(json.dumps(json_result, indent=2), encoding="utf-8")
        print(f"\nWrote raw counts to {args.output_json}")


if __name__ == "__main__":
    main()
