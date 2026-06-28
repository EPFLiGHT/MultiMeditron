#!/usr/bin/env python3
"""
Audit of labeling quality in the legacy MRI benchmark (evaluation_clip_mri.py).

This script reproduces and quantifies five systematic problems identified in the
original find_label() function used by tagemoua/processing-scripts/experts/evaluation_clip_mri.py:

  1. Label/function mismatch (Crohn bug) — find_label() returns "crohn" (lowercase)
                                            but the labels list contains "Crohn" (capital C).
                                            The label "Crohn" is therefore never assigned.
  2. Implicit healthy fallback            — any example not matching the other three keywords
                                            is silently labeled "healthy", including histology
                                            slides and non-MRI images.
  3. Case inconsistency on Bone infection — "Bone infection" (capital B and I) is searched
                                            exactly; variants with different casing are missed.
  4. Incoherent task definition           — the 4 classes (brain tumor, Crohn disease, bone
                                            infection, healthy) span unrelated body parts on
                                            a brain-MRI dataset, making the task medically
                                            ill-defined.
  5. 50% random subsetting               — the script discards half the data via random
                                            sampling before splitting, inflating variance.

Usage:
    python scripts/audit_legacy_mri_labels.py
    python scripts/audit_legacy_mri_labels.py --jsonl /path/to/MRI-5.jsonl
    python scripts/audit_legacy_mri_labels.py --max-lines 20000   # quick smoke-test

Output:
    Structured report to stdout. Deterministic (no randomness) — fully reproducible.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_JSONL = Path(
    "/lightscratch/users/tagemoua/processing-scripts/experts/MRI-5.jsonl"
)

# ---------------------------------------------------------------------------
# Legacy definitions (verbatim from evaluation_clip_mri.py:33-43)
# ---------------------------------------------------------------------------
LEGACY_LABELS_LIST = ["brain tumor", "Crohn", "healthy", "Bone infection"]


def legacy_find_label(example: dict) -> str:
    text = example.get("text", "")
    if "brain tumor" in text:
        return "brain tumor"
    if "crohn" in text:          # BUG: lowercase — never matches "Crohn" in labels list
        return "crohn"
    if "Bone infection" in text:
        return "Bone infection"
    return "healthy"             # fallback


# ---------------------------------------------------------------------------
# Fixed version — what the function *should* have returned to match labels list
# ---------------------------------------------------------------------------
def fixed_find_label(example: dict) -> str:
    text = example.get("text", "")
    tl = text.lower()
    if "brain tumor" in tl:
        return "brain tumor"
    if "crohn" in tl:
        return "Crohn"           # matches the labels list
    if "bone infection" in tl:
        return "Bone infection"
    return "healthy"


# ---------------------------------------------------------------------------
# Histology / modality detection helpers
# ---------------------------------------------------------------------------
HISTOLOGY_KEYWORDS = ["histolog", "epithelial", "nuclei", "neoplastic", "patholog",
                       "esophagus", "lymphoma", "mucosal", "stratum"]
MRI_KEYWORDS       = ["magnetic resonance", "mri scan", "mri of", "brain mri",
                       "cerebral", "cranial", "ventricle"]
OTHER_MODALITY_KW  = ["chest radiograph", "x-ray", "ultrasound", "echocardiograph",
                       "computed tomography", " ct "]


def classify_content(text: str) -> str:
    tl = text.lower()
    if any(k in tl for k in HISTOLOGY_KEYWORDS):
        return "histology_slide"
    if any(k in tl for k in MRI_KEYWORDS):
        return "mri_scan"
    if any(k in tl for k in OTHER_MODALITY_KW):
        return "other_modality"
    return "unlabeled_other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL,
                        help="Path to MRI-5.jsonl (default: %(default)s)")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Stop after N lines (for quick tests)")
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"[ERROR] JSONL not found: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Single-pass scan
    # -----------------------------------------------------------------------
    total = 0
    legacy_label_counts  = Counter()
    fixed_label_counts   = Counter()
    healthy_breakdown    = Counter()   # what's inside "healthy" (legacy)
    crohn_case_variants  = Counter()   # case forms of "crohn" in the file
    bone_case_variants   = Counter()   # case forms of "bone infection" in the file
    crohn_mislabeled     = 0           # crohn examples that fall into healthy (legacy)
    multi_keyword        = 0           # examples matching >1 non-healthy keyword

    with args.jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            if args.max_lines and total > args.max_lines:
                break

            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = ex.get("text", "")
            tl = text.lower()

            legacy_label = legacy_find_label(ex)
            fixed_label  = fixed_find_label(ex)

            legacy_label_counts[legacy_label] += 1
            fixed_label_counts[fixed_label]   += 1

            # --- Problem 1: Crohn mislabeling ---
            if "crohn" in tl:
                crohn_case_variants["any_crohn_variant"] += 1
                if "Crohn" in text:
                    crohn_case_variants["Crohn_capital"] += 1
                if "crohn" in text and "Crohn" not in text:
                    crohn_case_variants["crohn_lowercase_only"] += 1
                if legacy_label == "healthy":
                    crohn_mislabeled += 1

            # --- Problem 3: Bone infection casing ---
            if "bone infection" in tl:
                bone_case_variants["any_bone_infection"] += 1
                if "Bone infection" in text:
                    bone_case_variants["Bone_infection_matched"] += 1
                if "bone infection" in text and "Bone infection" not in text:
                    bone_case_variants["bone_infection_missed"] += 1
                if "BONE INFECTION" in text:
                    bone_case_variants["BONE_INFECTION_missed"] += 1

            # --- Problem 2: healthy fallback content ---
            if legacy_label == "healthy":
                healthy_breakdown[classify_content(text)] += 1

            # --- Multi-keyword ambiguity ---
            kws = sum([
                "brain tumor" in text,
                "crohn" in tl,
                "Bone infection" in text,
            ])
            if kws > 1:
                multi_keyword += 1

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    sep = "=" * 70

    print(sep)
    print("AUDIT: Legacy MRI Benchmark Label Quality")
    print(f"File : {args.jsonl}")
    print(f"Lines scanned: {total:,}")
    print(sep)

    # --- [0] Overall distributions ---
    print("\n[0] LABEL DISTRIBUTION COMPARISON")
    all_keys = sorted(set(list(legacy_label_counts) + list(fixed_label_counts)))
    print(f"    {'Label':<22} {'Legacy':>10}  {'Fixed':>10}  {'Δ':>10}")
    print(f"    {'-'*22} {'-'*10}  {'-'*10}  {'-'*10}")
    for label in all_keys:
        lg = legacy_label_counts.get(label, 0)
        fx = fixed_label_counts.get(label, 0)
        print(f"    {label:<22} {lg:>10,}  {fx:>10,}  {fx-lg:>+10,}")
    print(f"    {'TOTAL':<22} {total:>10,}  {total:>10,}")
    print()
    print(f"    Note: 'crohn' (lowercase) appears in legacy counts because")
    print(f"    find_label() returns 'crohn' — a value absent from the labels list.")
    print(f"    This means label_to_idx['crohn'] would raise a KeyError at runtime.")

    # --- [1] Crohn bug ---
    n_any_crohn = crohn_case_variants["any_crohn_variant"]
    n_cap       = crohn_case_variants["Crohn_capital"]
    n_lower     = crohn_case_variants["crohn_lowercase_only"]
    print(f"\n[1] LABEL/FUNCTION MISMATCH — 'Crohn' vs 'crohn'")
    print(f"    Labels list contains : 'Crohn'  (capital C)")
    print(f"    find_label() returns : 'crohn'  (lowercase)")
    print(f"    => label_to_idx lookup would raise KeyError for every Crohn example.")
    print(f"    Examples containing any crohn variant : {n_any_crohn:,}")
    print(f"      'Crohn' (capital, as in labels list) : {n_cap:,}")
    print(f"      'crohn' (lowercase only)             : {n_lower:,}")
    print(f"    Crohn examples silently re-labeled 'healthy' (legacy): {crohn_mislabeled:,}")

    # --- [2] Healthy fallback ---
    n_healthy = legacy_label_counts["healthy"]
    print(f"\n[2] IMPLICIT 'healthy' FALLBACK")
    print(f"    Examples labeled 'healthy' (legacy) : {n_healthy:,}  ({n_healthy/total*100:.1f}%)")
    print(f"    Content breakdown:")
    for bucket, n in healthy_breakdown.most_common():
        print(f"      {bucket:<30} {n:>8,}  ({n/n_healthy*100:.1f}% of healthy)")
    print(f"    => 'healthy' mixes genuine healthy MRI scans with histology slides")
    print(f"       and other modalities that share no semantic relationship.")

    # --- [3] Bone infection casing ---
    n_any_bone     = bone_case_variants["any_bone_infection"]
    n_bone_matched = bone_case_variants["Bone_infection_matched"]
    n_bone_missed  = bone_case_variants["bone_infection_missed"] + bone_case_variants["BONE_INFECTION_missed"]
    print(f"\n[3] CASE INCONSISTENCY — 'Bone infection'")
    print(f"    Examples with any 'bone infection' variant : {n_any_bone:,}")
    if n_any_bone > 0:
        print(f"      Matched ('Bone infection', exact)        : {n_bone_matched:,}  ({n_bone_matched/n_any_bone*100:.1f}%)")
        print(f"      Missed (other casing)                    : {n_bone_missed:,}  ({n_bone_missed/n_any_bone*100:.1f}%)")
    else:
        print(f"      No 'bone infection' variants found in this file.")

    # --- [4] Incoherent task ---
    n_brain_tumor   = legacy_label_counts["brain tumor"]
    n_crohn_legacy  = legacy_label_counts.get("crohn", 0)
    n_bone_legacy   = legacy_label_counts["Bone infection"]
    print(f"\n[4] INCOHERENT TASK DEFINITION")
    print(f"    The dataset is labeled as MRI-glob (brain MRI scans), yet the labels")
    print(f"    include 'Crohn' (gastrointestinal disease) and 'Bone infection'")
    print(f"    (orthopedic condition) — pathologies not diagnosable from brain MRI.")
    print(f"    Actual distribution of non-healthy labels (legacy):")
    print(f"      brain tumor   : {n_brain_tumor:>8,}")
    print(f"      crohn         : {n_crohn_legacy:>8,}  (lowercase — would cause KeyError)")
    print(f"      Bone infection: {n_bone_legacy:>8,}")

    # --- [5] 50% subsetting ---
    print(f"\n[5] ARBITRARY 50% RANDOM SUBSETTING")
    subset = total // 2
    train  = int(0.8 * subset)
    test   = subset - train
    print(f"    The script samples 50% of the data before splitting:")
    print(f"      Total examples available : {total:,}")
    print(f"      Used for train+test      : {subset:,}  (50%)")
    print(f"      Train set (80%)          : {train:,}")
    print(f"      Test set  (20%)          : {test:,}")
    print(f"    => Half the data is discarded without justification,")
    print(f"       inflating variance and reducing benchmark reliability.")

    # --- Summary ---
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    n_crohn_affected = n_any_crohn
    print(f"  Dataset size              : {total:,} examples")
    print(f"  'healthy' mislabels       : {n_healthy:,} ({n_healthy/total*100:.1f}%) — includes histology and non-MRI")
    print(f"  Crohn examples (KeyError) : {n_crohn_affected:,} — 'crohn'≠'Crohn' causes runtime crash")
    print(f"  Bone infection missed     : {n_bone_missed:,} ({n_bone_missed/max(n_any_bone,1)*100:.1f}% of variants)")
    print(f"  Data discarded (50% sub.) : {total - subset:,} examples dropped arbitrarily")
    print()
    print(f"  Conclusion: the legacy MRI benchmark has a fatal runtime bug (Crohn")
    print(f"  KeyError), an incoherent class set spanning unrelated body parts,")
    print(f"  and discards half the data without justification.")
    print(sep)


if __name__ == "__main__":
    main()
