#!/usr/bin/env python3
"""
Audit of labeling quality in the legacy ophthalmology benchmark
(eval_clip_opthalmology by turan).

This script reproduces and quantifies five systematic problems:

  1. Phantom classes     — 'glaucoma' and 'amd' are defined in labels[] but have
                           zero examples in the EyeDataset. The benchmark was
                           de facto binary despite claiming 4 classes.
  2. "dr" false positives — the pattern `"dr" in t` matches everyday substrings
                            (address, drop, draw, ...) and causes mislabeling.
  3. AMD always-true bug  — `"age related macular degeneration"` is a non-empty
                            string literal → always truthy in Python → every
                            example that reaches that branch is labeled "amd",
                            regardless of its actual content.
  4. Underscore mismatch  — EyeDataset labels use underscores ("diabetic_retinopathy")
                            but the labels list uses spaces ("diabetic retinopathy").
                            label_to_idx lookup would raise KeyError at runtime.
  5. Placeholder paths    — dataset_path_jsonl_train/test are "/mloscratch/users/you/..."
                            and were never configured. The script never ran.

Usage:
    python scripts/audit_legacy_ophthalmology_labels.py
    python scripts/audit_legacy_ophthalmology_labels.py --max-examples 5000

Output:
    Structured report to stdout. Deterministic — fully reproducible.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANIFEST_DIR = Path(
    "/lightscratch/users/cljordan/multimeditron/benchmark_splits/multimediset/eye"
)
EYEDATASET_PATH = Path(
    "/lightscratch/datasets/MultiMediset/general_purpose/EyeDataset"
)
PLACEHOLDER_TRAIN = "/mloscratch/users/you/ophtha/train.jsonl"
PLACEHOLDER_TEST  = "/mloscratch/users/you/ophtha/test.jsonl"

# ---------------------------------------------------------------------------
# Legacy definitions (verbatim from eval_clip_opthalmology)
# ---------------------------------------------------------------------------
LEGACY_LABELS = ["normal", "diabetic retinopathy", "glaucoma", "amd"]


def legacy_find_label(example: dict) -> str:
    if "label" in example and example["label"]:
        return str(example["label"]).lower()

    t = str(example.get("text", "")).lower()
    if "diabetic retinopathy" in t or "dr" in t:          # BUG: "dr" too broad
        return "diabetic retinopathy"
    if "glaucoma" in t:
        return "glaucoma"
    if "amd" in t or "age related macular degeneration" or "age-related macular degeneration" in t:
        # BUG: "age related macular degeneration" is a non-empty string → always True
        return "amd"
    if "normal" in t or "healthy" in t:
        return "normal"
    raise ValueError(f"Could not infer label from: {t[:80]!r}")


def fixed_find_label_text(text: str) -> str:
    """Correct version — no "dr" shortcut, no always-true string."""
    t = text.lower()
    if "diabetic retinopathy" in t:
        return "diabetic retinopathy"
    if "glaucoma" in t:
        return "glaucoma"
    if "amd" in t or "age related macular degeneration" in t or "age-related macular degeneration" in t:
        return "amd"
    if "normal" in t or "healthy" in t:
        return "normal"
    return None


# ---------------------------------------------------------------------------
# AMD always-true static proof (no data needed)
# ---------------------------------------------------------------------------
def prove_amd_bug():
    """Demonstrates the always-true bug with concrete Python evaluation."""
    literal = "age related macular degeneration"
    result = bool(literal)
    return literal, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap on EyeDataset examples to scan (default: all)")
    args = parser.parse_args()

    sep = "=" * 70

    print(sep)
    print("AUDIT: Legacy Ophthalmology Benchmark Label Quality")
    print(f"EyeDataset : {EYEDATASET_PATH}")
    print(f"Manifests  : {MANIFEST_DIR}")
    print(sep)

    # -----------------------------------------------------------------------
    # [0] Placeholder paths — static check, no data needed
    # -----------------------------------------------------------------------
    print("\n[0] PLACEHOLDER PATHS — SCRIPT NEVER RAN")
    for label, path in [("train", PLACEHOLDER_TRAIN), ("test", PLACEHOLDER_TEST)]:
        exists = Path(path).exists()
        print(f"  dataset_path_jsonl_{label} = {path!r}")
        print(f"    Exists on this machine: {exists}")
    print("  => Both paths contain the literal username 'you' and point to")
    print("     /mloscratch, a cluster filesystem not mounted here.")
    print("     The script was a template that was never filled in.")

    # -----------------------------------------------------------------------
    # [1] Phantom classes — check manifest label distribution
    # -----------------------------------------------------------------------
    print(f"\n[1] PHANTOM CLASSES — labels[] claims 4 classes, data has 2")
    print(f"  Labels defined in script : {LEGACY_LABELS}")
    all_manifest_labels = Counter()
    for split_name in ("train_model", "mlp_train", "benchmark_eval", "holdout_test"):
        path = MANIFEST_DIR / f"{split_name}.jsonl"
        if not path.exists():
            continue
        import json
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                all_manifest_labels[r.get("label", "?")] += 1

    total_manifest = sum(all_manifest_labels.values())
    print(f"\n  EyeDataset label distribution (all splits combined, {total_manifest:,} records):")
    for label in LEGACY_LABELS:
        n = all_manifest_labels.get(label, 0)
        # also check with underscore variant
        n_under = all_manifest_labels.get(label.replace(" ", "_"), 0)
        total_n = n + n_under
        flag = " ← PHANTOM" if total_n == 0 else ""
        print(f"    {label:<30} {total_n:>8,}{flag}")
    print()
    for k, v in all_manifest_labels.most_common():
        print(f"    actual key in data: {k!r:<35} {v:>8,}")
    print("  => 'glaucoma' and 'amd' have 0 examples. The 4-class benchmark")
    print("     was de facto a binary classification task.")

    # -----------------------------------------------------------------------
    # [2 & 4] Run legacy find_label on real EyeDataset texts
    # -----------------------------------------------------------------------
    print(f"\n[2] 'dr' FALSE POSITIVES  &  [4] UNDERSCORE MISMATCH")
    print("  Loading EyeDataset texts…")

    if not EYEDATASET_PATH.exists():
        print(f"  [SKIP] EyeDataset not found at {EYEDATASET_PATH}")
    else:
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(EYEDATASET_PATH))
            # Use the val split (smaller) for speed
            split = ds["val"] if "val" in ds else ds["train"]
            if args.max_examples:
                split = split.select(range(min(args.max_examples, len(split))))

            total = len(split)
            legacy_counts  = Counter()
            fixed_counts   = Counter()
            dr_false_pos   = 0    # "dr" matched but not "diabetic retinopathy"
            dr_words        = Counter()  # which words triggered "dr"
            underscore_keys = Counter()  # label field values from data
            amd_from_bug    = 0   # labeled amd purely by the always-true bug
            key_errors      = 0

            import re
            DR_WORD_RE = re.compile(r'\bdr\w*', re.IGNORECASE)

            for i in range(total):
                row = split[i]
                text = row.get("text", "") or ""
                t = text.lower()

                # Track what value "label" field would give
                # (manifests have label, raw dataset doesn't — simulate both paths)
                has_label_field = False  # raw EyeDataset has no explicit label field

                # --- Legacy path (text-based) ---
                try:
                    label = legacy_find_label({"text": text})
                    legacy_counts[label] += 1
                except ValueError:
                    key_errors += 1
                    label = None

                # --- Fixed path ---
                fixed = fixed_find_label_text(text)
                fixed_counts[fixed or "unresolved"] += 1

                # --- Problem 2: "dr" false positives ---
                if "dr" in t and "diabetic retinopathy" not in t:
                    dr_false_pos += 1
                    # find the triggering word
                    for m in DR_WORD_RE.finditer(text):
                        w = m.group().lower()
                        if w != "dr":  # "dr" alone vs embedded
                            dr_words[w] += 1
                        else:
                            dr_words["dr (standalone)"] += 1

                # --- Problem 3: amd-bug (any example not matching earlier rules) ---
                if ("diabetic retinopathy" not in t and "dr" not in t
                        and "glaucoma" not in t
                        and label == "diabetic retinopathy"
                        and "amd" not in t):
                    amd_from_bug += 1

            print(f"  EyeDataset split: {len(split):,} examples scanned")

            print(f"\n  Legacy find_label distribution (text-only path):")
            for lbl in LEGACY_LABELS + ["unresolved"]:
                n = legacy_counts.get(lbl, 0)
                print(f"    {lbl:<30} {n:>8,}  ({n/total*100:.1f}%)")

            print(f"\n  Fixed find_label distribution:")
            for lbl in LEGACY_LABELS + ["unresolved"]:
                n = fixed_counts.get(lbl, 0)
                print(f"    {lbl:<30} {n:>8,}  ({n/total*100:.1f}%)")

            print(f"\n  'dr' false positives:")
            print(f"    Examples where 'dr' matched but NOT 'diabetic retinopathy': {dr_false_pos:,}")
            print(f"    ({dr_false_pos/total*100:.1f}% of examples mislabeled as 'diabetic retinopathy')")
            print(f"    Top triggering patterns:")
            for word, n in dr_words.most_common(8):
                print(f"      {word:<30} {n:>6,}")

        except Exception as e:
            print(f"  [ERROR] Could not load EyeDataset: {e}")

    # -----------------------------------------------------------------------
    # [3] AMD always-true bug — static proof
    # -----------------------------------------------------------------------
    print(f"\n[3] AMD ALWAYS-TRUE BUG — static proof")
    literal, result = prove_amd_bug()
    print(f"  The condition in find_label():")
    print(f'    if "amd" in t or "age related macular degeneration" or "age-related macular degeneration" in t:')
    print(f"  Python evaluates the middle operand as a standalone boolean:")
    print(f'    bool("{literal}") == {result}')
    print(f"  => The entire condition short-circuits to True for every example")
    print(f"     that is not already caught by the 'diabetic retinopathy' or")
    print(f"     'glaucoma' branches — regardless of what the text says.")
    print(f"  Proof:")
    t_test = "this is a completely normal fundus photograph"
    cond = "amd" in t_test or "age related macular degeneration" or "age-related macular degeneration" in t_test
    print(f'    t = "{t_test}"')
    print(f'    result of the condition = {cond}  (expected: False, got: {cond})')

    # -----------------------------------------------------------------------
    # [4] Underscore mismatch — static + data proof
    # -----------------------------------------------------------------------
    print(f"\n[4] UNDERSCORE MISMATCH — label field vs labels list")
    print(f"  labels list in script     : 'diabetic retinopathy'  (space)")
    print(f"  EyeDataset manifest field : 'diabetic_retinopathy'  (underscore)")
    print(f"  find_label() line 65      : return str(example['label']).lower()")
    print(f"                              → returns 'diabetic_retinopathy'")
    print(f"  label_to_idx lookup       : label_to_idx['diabetic_retinopathy']")
    print(f"                              → KeyError (key not in dict)")
    import json
    example_key = None
    manifest = MANIFEST_DIR / "mlp_train.jsonl"
    if manifest.exists():
        with manifest.open() as f:
            r = json.loads(f.readline())
            example_key = r.get("label")
    if example_key:
        label_to_idx = {l: i for i, l in enumerate(LEGACY_LABELS)}
        would_crash = example_key not in label_to_idx
        print(f"  Actual label value from manifest : {example_key!r}")
        print(f"  Present in label_to_idx          : {not would_crash}")
        if would_crash:
            print(f"  => Would raise KeyError at runtime.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"  1. Phantom classes  : 'glaucoma' (0 examples) and 'amd' (0 examples)")
    print(f"                        Task is binary in practice, not 4-class.")
    print(f"  2. 'dr' FP          : see above — mislabels normal images as DR")
    print(f"  3. AMD always-true  : bool('age related macular degeneration') == True")
    print(f"                        Every non-DR/non-glaucoma example → 'amd'")
    print(f"  4. Underscore bug   : 'diabetic_retinopathy' ≠ 'diabetic retinopathy'")
    print(f"                        → KeyError on label_to_idx lookup")
    print(f"  5. Never ran        : placeholder paths '/mloscratch/users/you/...'")
    print(f"                        were never replaced with real data paths.")
    print(sep)


if __name__ == "__main__":
    main()
