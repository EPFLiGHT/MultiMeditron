#!/usr/bin/env python3
"""Analyze the true source composition of CT2D-glob by parsing raw file paths.

CT2D-glob captions are AI-generated and unreliable for identifying image modality.
The raw path JSONL (CT2D-glob-rawpath2905.jsonl) retains original absolute paths
from before anonymization, which encode the source dataset and category in the
filename. This script uses filename regex patterns to recover the real breakdown.

Usage:
    python scripts/analyze_ct2d_glob_sources.py
    python scripts/analyze_ct2d_glob_sources.py --output_json docs/source/ct2d_glob_source_analysis.json
"""


import argparse
import json
import re
from collections import Counter
from pathlib import Path


RAW_PATH_JSONL = Path(
    "/lightscratch/users/nemo/datasets/CT_data/CT2D-glob/old/CT2D-glob-rawpath2905.jsonl"
)


def classify_filename(fname):
    """Map a raw filename to a source category using naming conventions."""
    if fname.startswith("brain-"):
        return "histology__brain"

    if re.search(r"seg_train", fname, re.IGNORECASE):
        if re.search(r"covid", fname, re.IGNORECASE):
            return "ct__covid_lung"
        if re.search(r"right.lung", fname, re.IGNORECASE):
            return "ct__right_lung"
        if re.search(r"left.lung", fname, re.IGNORECASE):
            return "ct__left_lung"
        if re.search(r"atherosoma|atherosclerosis", fname, re.IGNORECASE):
            return "ct__atherosoma"
        m = re.search(r"-([^-]+?)(?:\.nii|-\d)", fname)
        label = m.group(1).strip().replace(" ", "_").lower() if m else "unknown"
        return f"ct__{label}"

    if fname.startswith("IHC4BC"):
        return "histo__IHC4BC_breast"

    # TCGA cancer-specific: CancerType-CancerType-HASH.jpg (two identical words)
    m = re.match(r"([A-Za-z][A-Za-z_]+(?:_[A-Za-z]+)+)-\1-[A-Za-z0-9]{4}", fname)
    if m:
        cancer = m.group(1).lower()
        return f"tcga_cancer__{cancer}"

    # Generic organ histology: organname-HASH.jpg  (all lowercase, no underscore)
    m = re.match(r"([a-z][a-z_]+)-[a-f0-9]{6,}\.jpg", fname)
    if m:
        organ = m.group(1)
        return f"tcga_histo__{organ}"

    # TCGA cancer with explicit cancer name prefix: CancerName-HASH.jpg
    m = re.match(r"([A-Za-z][A-Za-z_]+(?:-[A-Za-z_]+)*)-[a-f0-9]{6,}\.jpg", fname)
    if m:
        name = m.group(1).lower().replace("-", "_")
        return f"tcga_cancer__{name}"

    return f"other__{fname[:40]}"


def aggregate_super_categories(counts):
    """Collapse fine-grained categories into human-readable super-categories."""
    super_counts = Counter()
    for cat, n in counts.items():
        if cat.startswith("ct__"):
            super_counts["Real CT (NIfTI slices)"] += n
        elif cat == "histology__brain":
            super_counts["Brain histology (non-TCGA)"] += n
        elif cat.startswith("tcga_histo__"):
            super_counts["TCGA histology – generic organ"] += n
        elif cat.startswith("tcga_cancer__"):
            super_counts["TCGA histology – named cancer"] += n
        elif cat.startswith("histo__"):
            super_counts["Other histology (IHC, etc.)"] += n
        else:
            super_counts["Miscellaneous / unclassified"] += n
    return super_counts


def analyze(jsonl_path):
    fine = Counter()
    examples = {}

    with jsonl_path.open() as f:
        for line in f:
            d = json.loads(line)
            fname = d["modalities"][0]["value"].split("/")[-1]
            text = d.get("text", "")[:120].replace("\n", " ")
            cat = classify_filename(fname)
            fine[cat] += 1
            if cat not in examples:
                examples[cat] = text

    super_cats = aggregate_super_categories(fine)
    return fine, super_cats, examples


def print_report(fine, super_cats, examples):
    total = sum(fine.values())

    def pct(n):
        return f"{100 * n / total:.1f}%"

    print("=" * 70)
    print("  CT2D-glob — actual composition by file source")
    print(f"  Total: {total:,} examples")
    print("=" * 70)

    print("\n── SUPER-CATEGORIES ──────────────────────────────────────────────")
    for cat, n in super_cats.most_common():
        print(f"  {cat:<45} {n:>9,}  ({pct(n)})")

    print("\n── DETAILED CATEGORIES (top 60) ──────────────────────────────────")
    for cat, n in fine.most_common(60):
        ex = examples.get(cat, "")[:80]
        print(f"  {cat:<55} {n:>8,}  ({pct(n)})")
        print(f'    ex: "{ex}"')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=RAW_PATH_JSONL)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    print(f"Reading {args.jsonl} ...")
    fine, super_cats, examples = analyze(args.jsonl)
    print_report(fine, super_cats, examples)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "total": sum(fine.values()),
            "super_categories": dict(super_cats.most_common()),
            "fine_categories": dict(fine.most_common()),
        }
        args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
