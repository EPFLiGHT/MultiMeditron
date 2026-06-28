#!/usr/bin/env python3
"""Detailed content analysis of MRI-glob.jsonl.

Streams through the full JSONL in one pass and reports:
  - Anatomical regions covered
  - MRI sequences and planes
  - Pathologies mentioned
  - Image path structure (subfolder patterns)
  - Caption length distribution
  - Sample captions per anatomical region

Usage:
    python scripts/analyze_mri_glob_content.py
    python scripts/analyze_mri_glob_content.py --max_lines 100000  # quick sample
    python scripts/analyze_mri_glob_content.py --output_json results/mri_glob_analysis.json
"""


import argparse
import json
from collections import Counter
from pathlib import Path


JSONL_PATH = Path("/lightscratch/users/nemo/datasets/MRI_data/MRI-glob/MRI-glob.jsonl")

# ---------------------------------------------------------------------------
# Keyword taxonomy
# ---------------------------------------------------------------------------

ANATOMY = {
    "brain / cerebral": (
        "brain",
        "cerebral",
        "cerebrum",
        "cerebellum",
        "brainstem",
        "intracranial",
        "cranial",
    ),
    "spine / vertebral": (
        "spine",
        "spinal",
        "vertebral",
        "lumbar",
        "cervical",
        "thoracic",
        "sacral",
        "disc",
        "cord",
    ),
    "knee": ("knee", "meniscus", "meniscal", "ligament", "acl", "pcl"),
    "shoulder": ("shoulder", "rotator", "cuff", "glenohumeral"),
    "hip": ("hip", "femoral", "acetabul"),
    "abdomen / liver": (
        "abdomen",
        "abdominal",
        "liver",
        "hepat",
        "pancrea",
        "spleen",
        "kidney",
        "renal",
        "bowel",
    ),
    "pelvis / prostate": (
        "pelvis",
        "pelvic",
        "prostate",
        "uterus",
        "uterine",
        "ovary",
        "ovarian",
        "endometri",
    ),
    "cardiac / heart": ("cardiac", "heart", "myocard", "ventricle", "atrium", "aorta"),
    "breast": ("breast", "mammary"),
    "neck / thyroid": ("neck", "thyroid", "parathyroid", "salivary", "parotid"),
    "orbit / eye": ("orbit", "orbital", "optic", "retina"),
}

MRI_SEQUENCES = {
    "T1": ("t1-weighted", "t1 weighted", " t1 ", "t1w"),
    "T2": ("t2-weighted", "t2 weighted", " t2 ", "t2w"),
    "FLAIR": ("flair",),
    "DWI / ADC": ("diffusion", "dwi", " adc "),
    "Contrast / Gd": (
        "contrast",
        "gadolinium",
        "contrast-enhanced",
        "post-contrast",
        "enhancement",
    ),
    "MRA": ("angiograph", "mra", "magnetic resonance angiograph"),
    "Perfusion": ("perfusion", "dsc", "dce"),
}

MRI_PLANES = {
    "axial / transverse": ("axial", "transverse", "transaxial"),
    "sagittal": ("sagittal",),
    "coronal": ("coronal",),
}

PATHOLOGIES = {
    "brain tumor (generic)": ("brain tumor",),
    "glioma / glioblastoma": ("glioma", "glioblastoma", "gbm"),
    "meningioma": ("meningioma",),
    "metastasis": ("metastasis", "metastatic", "metastases"),
    "pituitary": ("pituitary",),
    "stroke / infarct": ("stroke", "infarct", "ischemi", "cerebrovascular"),
    "hemorrhage": ("hemorrhage", "haematoma", "hematoma", "bleeding"),
    "multiple sclerosis": ("multiple sclerosis", "demyelinat", "white matter lesion"),
    "alzheimer / dementia": ("alzheimer", "dementia", "atrophy"),
    "epilepsy": ("epilepsy", "epileptic", "seizure"),
    "edema": ("edema", "oedema"),
    "lesion (generic)": ("lesion",),
    "abscess / infection": ("abscess", "infection", "osteomyelitis"),
    "cyst": ("cyst",),
    "hernia / disc": ("hernia", "herniat", "disc protrusion", "radiculopathy"),
    "fracture": ("fracture", "fractur"),
    "crohn": ("crohn",),
    "healthy / normal": ("healthy", "normal", "unremarkable", "no findings"),
}


def first_modality_value(example):
    modalities = example.get("modalities") or []
    if not modalities:
        return ""
    first = modalities[0]
    return str(first.get("value", "") if isinstance(first, dict) else first)


def match_keywords(text, taxonomy):
    matched = []
    for label, keywords in taxonomy.items():
        if any(kw in text for kw in keywords):
            matched.append(label)
    return matched


def path_subfolder_pattern(image_value, depth=3):
    parts = Path(image_value.replace("\\", "/")).parts
    meaningful = [p for p in parts if p not in (".", "/", "")]
    return "/".join(meaningful[:depth]) if meaningful else "(empty)"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze(jsonl_path, max_lines):
    anatomy_counts = Counter()
    sequence_counts = Counter()
    plane_counts = Counter()
    pathology_counts = Counter()
    path_pattern_counts = Counter()
    caption_lengths = []
    no_text_count = 0
    total = 0

    # Collect one sample caption per anatomy category
    anatomy_samples = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_lines is not None and total >= max_lines:
                break
            if not line.strip():
                continue

            example = json.loads(line)
            text = str(example.get("text") or "")
            text_lower = text.lower()
            total += 1

            if not text_lower:
                no_text_count += 1
                continue

            caption_lengths.append(len(text.split()))

            matched_anatomy = match_keywords(text_lower, ANATOMY)
            for label in matched_anatomy:
                anatomy_counts[label] += 1
                if label not in anatomy_samples:
                    anatomy_samples[label] = text[:300]
            if not matched_anatomy:
                anatomy_counts["(no anatomy keyword)"] += 1

            for label in match_keywords(text_lower, MRI_SEQUENCES):
                sequence_counts[label] += 1

            for label in match_keywords(text_lower, MRI_PLANES):
                plane_counts[label] += 1

            for label in match_keywords(text_lower, PATHOLOGIES):
                pathology_counts[label] += 1

            image_value = first_modality_value(example)
            if image_value:
                pattern = path_subfolder_pattern(image_value, depth=3)
                path_pattern_counts[pattern] += 1

    lengths_arr = sorted(caption_lengths)
    n = len(lengths_arr)
    avg_len = sum(lengths_arr) / n if n else 0
    p25 = lengths_arr[n // 4] if n else 0
    p50 = lengths_arr[n // 2] if n else 0
    p75 = lengths_arr[3 * n // 4] if n else 0

    return {
        "total_lines": total,
        "no_text_count": no_text_count,
        "caption_length": {
            "mean": round(avg_len, 1),
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "min": lengths_arr[0] if lengths_arr else 0,
            "max": lengths_arr[-1] if lengths_arr else 0,
        },
        "anatomy": dict(anatomy_counts.most_common()),
        "sequences": dict(sequence_counts.most_common()),
        "planes": dict(plane_counts.most_common()),
        "pathologies": dict(pathology_counts.most_common()),
        "top_path_patterns": dict(path_pattern_counts.most_common(30)),
        "anatomy_samples": anatomy_samples,
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
        print(f"  {label:<40} {count:>8}  ({pct(count, total)})")


def print_report(result):
    total = result["total_lines"]
    print("=" * 60)
    print("  MRI-glob Content Analysis")
    print("=" * 60)
    print(f"  Total lines        : {total:,}")
    print(f"  No text field      : {result['no_text_count']:,}")
    cl = result["caption_length"]
    print(
        f"  Caption length     : mean={cl['mean']} words  p25={cl['p25']}  p50={cl['p50']}  p75={cl['p75']}  max={cl['max']}"
    )

    print_counter("ANATOMICAL REGIONS", result["anatomy"], total)
    print_counter("MRI SEQUENCES", result["sequences"], total)
    print_counter("IMAGING PLANES", result["planes"], total)
    print_counter("PATHOLOGIES MENTIONED", result["pathologies"], total)
    print_counter(
        "TOP IMAGE PATH PATTERNS", result["top_path_patterns"], total, top_n=20
    )

    print("\nSAMPLE CAPTIONS BY ANATOMICAL REGION")
    print("-" * 60)
    for region, sample in result["anatomy_samples"].items():
        print(f"\n[{region}]")
        print(f"  {sample}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MRI-glob.jsonl content.")
    parser.add_argument("--jsonl", type=Path, default=JSONL_PATH)
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Cap lines for quick sampling (default: all 2.6M)",
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
    if not args.jsonl.exists():
        print(f"JSONL not found: {args.jsonl}")
        return

    scope = f"{args.max_lines:,} lines" if args.max_lines else "all lines"
    print(f"Analyzing {args.jsonl} ({scope})...")

    result = analyze(args.jsonl, args.max_lines)
    print_report(result)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        # Remove sample captions from JSON output to keep it clean
        json_result = {k: v for k, v in result.items() if k != "anatomy_samples"}
        args.output_json.write_text(json.dumps(json_result, indent=2), encoding="utf-8")
        print(f"\nWrote raw counts to {args.output_json}")


if __name__ == "__main__":
    main()
