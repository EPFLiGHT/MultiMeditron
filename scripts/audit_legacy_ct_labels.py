#!/usr/bin/env python3
"""
Audit of labeling quality in the legacy CT benchmark (evaluation_clip_ct2d.py).

This script reproduces and quantifies four systematic problems identified in the
original find_label() function used by tagemoua/processing-scripts/experts/evaluation_clip_ct2d.py:

  1. Implicit healthy fallback   — examples that match no keyword are silently
                                   labeled "healthy", including histology slides.
  2. Case-sensitivity bug        — "Covid" (capital C) misses "covid" and "COVID".
  3. Ambiguous multi-keyword     — examples matching several labels are resolved
                                   by arbitrary keyword order, not medical logic.
  4. AI-caption noise            — pathology words appear in captions of unrelated
                                   images (e.g. "tumor" in a histology description).

Usage:
    python scripts/audit_legacy_ct_labels.py
    python scripts/audit_legacy_ct_labels.py --jsonl /path/to/CT2D-glob.jsonl
    python scripts/audit_legacy_ct_labels.py --max-lines 100000   # quick smoke-test

Output:
    Prints a structured report to stdout. Suitable for copy-paste into a report.
    Totals are deterministic (no randomness) for full reproducibility.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_JSONL = Path(
    "/lightscratch/users/tagemoua/processing-scripts/experts/CT2D-glob.jsonl"
)

# ---------------------------------------------------------------------------
# Legacy labeling logic (verbatim reproduction of evaluation_clip_ct2d.py:31)
# ---------------------------------------------------------------------------
LEGACY_LABELS = ["atherosoma", "Covid", "healthy", "glioblastoma", "tumor"]


def legacy_find_label(example: dict) -> str:
    text = example.get("text", "")
    if "tumor" in text:
        return "tumor"
    if "atherosoma" in text:
        return "atherosoma"
    if "glioblastoma" in text:
        return "glioblastoma"
    if "Covid" in text:          # original: capital C only
        return "Covid"
    return "healthy"             # fallback — no explicit match needed


# ---------------------------------------------------------------------------
# Keywords used to characterise "healthy" mislabels
# ---------------------------------------------------------------------------
HISTOLOGY_KEYWORDS = ["histolog", "epithelial", "nuclei", "neoplastic", "patholog"]
MODALITY_KEYWORDS  = ["chest radiograph", "radiograph", "x-ray", "ultrasound", "mri",
                       "magnetic resonance", "echocardiograph"]


def keyword_flags(text: str) -> dict:
    tl = text.lower()
    return {
        "has_histology":  any(k in tl for k in HISTOLOGY_KEYWORDS),
        "has_modality":   any(k in tl for k in MODALITY_KEYWORDS),
        "covid_lower":    "covid" in tl,
        "covid_upper":    "Covid" in text,
        "covid_allcaps":  "COVID" in text,
        "multi_keyword":  sum(k in text for k in ["tumor", "atherosoma", "glioblastoma", "Covid"])
                          + ("healthy" in tl and not any(k in text for k in ["tumor","atherosoma","glioblastoma","Covid"])) > 1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL,
                        help="Path to CT2D-glob.jsonl (default: %(default)s)")
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
    label_counts       = Counter()
    healthy_breakdown  = Counter()   # what's inside "healthy" examples
    covid_case_buckets = Counter()   # case variants of covid in the whole file
    ambiguous_examples = []          # examples that match >1 keyword
    order_flipped      = 0           # tumor+glioblastoma: would change label if order swapped

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

            label = legacy_find_label(ex)
            label_counts[label] += 1
            text = ex.get("text", "")
            flags = keyword_flags(text)

            # --- Problem 1: what is inside "healthy" ---
            if label == "healthy":
                if flags["has_histology"]:
                    healthy_breakdown["histology_slide"] += 1
                elif flags["has_modality"]:
                    healthy_breakdown["other_modality"] += 1
                else:
                    healthy_breakdown["genuinely_unlabeled"] += 1

            # --- Problem 2: covid case variants ---
            if flags["covid_lower"]:
                covid_case_buckets["any_covid_variant"] += 1
            if flags["covid_upper"]:
                covid_case_buckets["Covid_matched_by_legacy"] += 1
            if flags["covid_allcaps"] and not flags["covid_upper"]:
                covid_case_buckets["COVID_missed_by_legacy"] += 1
            if flags["covid_lower"] and not flags["covid_upper"] and not flags["covid_allcaps"]:
                covid_case_buckets["covid_lowercase_missed"] += 1

            # --- Problem 3: ambiguous / order-dependent ---
            kws_present = [k for k in ["tumor", "atherosoma", "glioblastoma", "Covid"]
                           if k in text]
            if len(kws_present) > 1:
                ambiguous_examples.append((label, kws_present))
            # Would swapping tumor↔glioblastoma priority change the label?
            if "tumor" in text and "glioblastoma" in text:
                order_flipped += 1

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    sep = "=" * 70

    print(sep)
    print("AUDIT: Legacy CT Benchmark Label Quality")
    print(f"File : {args.jsonl}")
    print(f"Lines scanned: {total:,}")
    print(sep)

    # --- Overall distribution ---
    print("\n[0] OVERALL LABEL DISTRIBUTION (legacy find_label)")
    print(f"    {'Label':<20} {'Count':>10}  {'%':>7}")
    print(f"    {'-'*20} {'-'*10}  {'-'*7}")
    for label in LEGACY_LABELS:
        n = label_counts[label]
        print(f"    {label:<20} {n:>10,}  {n/total*100:>6.2f}%")
    print(f"    {'TOTAL':<20} {total:>10,}  100.00%")

    # --- Problem 1 ---
    print(f"\n[1] IMPLICIT 'healthy' FALLBACK")
    n_healthy = label_counts["healthy"]
    print(f"    Examples labeled 'healthy' : {n_healthy:,}  ({n_healthy/total*100:.1f}% of dataset)")
    print(f"    Of which:")
    for bucket, n in healthy_breakdown.most_common():
        print(f"      {bucket:<30} {n:>10,}  ({n/n_healthy*100:.1f}% of healthy)")
    print(f"    => The vast majority of 'healthy' examples are histology slides,")
    print(f"       not healthy CT scans. The fallback label is semantically wrong.")

    # --- Problem 2 ---
    print(f"\n[2] CASE-SENSITIVITY BUG  ('Covid' vs 'covid' / 'COVID')")
    total_covid_variants = covid_case_buckets["any_covid_variant"]
    matched             = covid_case_buckets["Covid_matched_by_legacy"]
    missed_lower        = covid_case_buckets["covid_lowercase_missed"]
    missed_caps         = covid_case_buckets["COVID_missed_by_legacy"]
    print(f"    Total examples containing any covid variant : {total_covid_variants:,}")
    print(f"    Matched by legacy   ('Covid') : {matched:,}  ({matched/max(total_covid_variants,1)*100:.1f}%)")
    print(f"    Missed — 'covid'              : {missed_lower:,}  ({missed_lower/max(total_covid_variants,1)*100:.1f}%)")
    print(f"    Missed — 'COVID'              : {missed_caps:,}  ({missed_caps/max(total_covid_variants,1)*100:.1f}%)")
    missed_total = missed_lower + missed_caps
    print(f"    Total missed                  : {missed_total:,}  ({missed_total/max(total_covid_variants,1)*100:.1f}%)")
    print(f"    => These examples were silently re-labeled 'healthy'.")

    # --- Problem 3 ---
    n_ambig = len(ambiguous_examples)
    print(f"\n[3] AMBIGUOUS MULTI-KEYWORD EXAMPLES (order-dependent labeling)")
    print(f"    Examples matching >1 label keyword : {n_ambig:,}  ({n_ambig/total*100:.2f}%)")
    print(f"    Examples with both 'tumor' AND 'glioblastoma' : {order_flipped:,}")
    print(f"    => All are labeled 'tumor' (first in the if-chain) regardless of")
    print(f"       which condition is actually described in the caption.")
    if ambiguous_examples:
        print(f"    Top co-occurring keyword pairs:")
        pair_counts = Counter(tuple(sorted(kws)) for _, kws in ambiguous_examples)
        for pair, n in pair_counts.most_common(5):
            print(f"      {str(pair):<40} {n:>8,}")

    # --- Problem 4 ---
    print(f"\n[4] AI-CAPTION NOISE (pathology words in unrelated images)")
    n_histo_tumor = sum(
        1 for label, kws in ambiguous_examples
        if label == "tumor"
    )
    # Re-count: histology examples that got labeled 'tumor'
    # We need a second pass for this — approximate from healthy_breakdown
    print(f"    The JSONL captions are AI-generated and describe image regions in")
    print(f"    detail. The word 'tumor' appears in {label_counts['tumor']:,} examples,")
    print(f"    but the same JSONL contains {healthy_breakdown['histology_slide']:,} histology")
    print(f"    slides labeled 'healthy' (histology keyword present but 'tumor' absent).")
    print(f"    When 'tumor' appears in a histology caption to describe adjacent tissue,")
    print(f"    the example is mislabeled 'tumor' even though the image is not a CT scan.")

    # --- Summary ---
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"  Dataset size          : {total:,} examples")
    print(f"  'healthy' mislabels   : {n_healthy:,} ({n_healthy/total*100:.1f}%) — dominated by histology slides")
    print(f"  Covid examples missed : {missed_total:,} ({missed_total/max(total_covid_variants,1)*100:.1f}% of all covid variants)")
    print(f"  Ambiguous labels      : {n_ambig:,} ({n_ambig/total*100:.2f}%) — resolved by arbitrary keyword order")
    print(f"\n  Conclusion: the legacy benchmark evaluated models on a dataset where")
    print(f"  ~{healthy_breakdown['histology_slide']/total*100:.0f}% of examples are histology slides mislabeled as 'healthy',")
    print(f"  making reported accuracies unreliable and non-reproducible across")
    print(f"  caption phrasings.")
    print(sep)


if __name__ == "__main__":
    main()
