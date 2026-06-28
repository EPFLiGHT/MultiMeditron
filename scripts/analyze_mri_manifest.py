#!/usr/bin/env python3
"""Analyze the MRI MultiMediSet manifests.

Inspects all 4 splits (mlp_train, benchmark_eval, holdout_test, train_model),
reports label distributions, source datasets, and samples captions from the
source JSONL to verify label quality.

Usage:
    python scripts/analyze_mri_manifest.py
    python scripts/analyze_mri_manifest.py --manifest_root benchmark_splits/multimediset/mri
    python scripts/analyze_mri_manifest.py --samples 5   # captions per label
"""


import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_ROOT = REPO_ROOT / "benchmark_splits" / "multimediset" / "mri"
SOURCE_JSONL = Path(
    "/lightscratch/users/nemo/datasets/MRI_data/MRI-glob/MRI-glob.jsonl"
)

SPLITS = ["mlp_train", "benchmark_eval", "holdout_test", "train_model"]


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Source JSONL random-access index (built once, keyed by line number)
# ---------------------------------------------------------------------------


def build_line_index(jsonl_path, indices):
    """Return {line_index: parsed_record} for the requested line indices."""
    result = {}
    if not jsonl_path.exists():
        print(f"  [warn] Source JSONL not found: {jsonl_path}")
        return result

    target = set(indices)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in target:
                result[i] = json.loads(line)
                target.discard(i)
                if not target:
                    break
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_split(split_name, records):
    if not records:
        return {"total": 0}

    label_counts = Counter(r["label"] for r in records)
    label_id_counts = Counter(r["label_id"] for r in records)
    dataset_counts = Counter(r.get("dataset", "unknown") for r in records)
    source_root_counts = Counter(r.get("source_root", "unknown") for r in records)

    return {
        "total": len(records),
        "label_counts": dict(label_counts.most_common()),
        "label_id_counts": dict(label_id_counts.most_common()),
        "datasets": dict(dataset_counts.most_common()),
        "source_roots": dict(source_root_counts.most_common()),
    }


def sample_captions(records, n_per_label):
    """Load n_per_label source captions for each label to inspect text quality."""
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    needed_indices = set()
    samples_by_label = {}

    for label, recs in by_label.items():
        chosen = recs[:n_per_label]
        indices = [r["source_index"] for r in chosen]
        needed_indices.update(indices)
        samples_by_label[label] = indices

    index_to_row = build_line_index(SOURCE_JSONL, needed_indices)

    result = {}
    for label, indices in samples_by_label.items():
        texts = []
        for idx in indices:
            row = index_to_row.get(idx)
            if row is None:
                texts.append(f"[index {idx} not found in JSONL]")
            else:
                texts.append(row.get("text", "[no text field]")[:300])
        result[label] = texts

    return result


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_analysis(split_name, stats):
    print(f"\n--- {split_name} ---")
    if stats["total"] == 0:
        print("  (no records)")
        return

    print(f"  Total records : {stats['total']}")

    print("  Label distribution:")
    for label, count in stats["label_counts"].items():
        pct = 100 * count / stats["total"]
        label_id = next(
            (
                lid
                for lid, lbl in zip(
                    stats["label_id_counts"].keys(), stats["label_counts"].keys()
                )
                if lbl == label
            ),
            "?",
        )
        print(f"    [{label_id}] {label:<40} {count:>6}  ({pct:.1f}%)")

    if len(stats["datasets"]) > 1 or list(stats["datasets"].keys()) != ["MRI-glob"]:
        print("  Source datasets:")
        for ds, count in stats["datasets"].items():
            print(f"    {ds}: {count}")


def print_captions(split_name, captions):
    print(f"\n  Caption samples from '{split_name}':")
    for label, texts in captions.items():
        print(f"\n  [{label}]")
        for i, text in enumerate(texts, 1):
            print(f"    {i}. {text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MRI MultiMediSet manifests.")
    parser.add_argument(
        "--manifest_root",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT,
        help=f"Path to MRI manifest directory (default: {DEFAULT_MANIFEST_ROOT})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of source captions to sample per label (default: 3)",
    )
    parser.add_argument(
        "--caption_split",
        default="benchmark_eval",
        choices=SPLITS,
        help="Which split to use for caption sampling (default: benchmark_eval)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_root = args.manifest_root

    print_section("MRI Manifest Analysis")
    print(f"Manifest root : {manifest_root}")
    print(f"Source JSONL  : {SOURCE_JSONL}")
    print(f"JSONL exists  : {SOURCE_JSONL.exists()}")

    # --- Per-split statistics ---
    all_records = {}
    print_section("Split statistics")

    for split in SPLITS:
        path = manifest_root / f"{split}.jsonl"
        records = load_manifest(path)
        all_records[split] = records
        stats = analyze_split(split, records)
        exists = "✓" if path.exists() else "✗"
        print_analysis(f"{exists} {split}", stats)

    # --- Caption quality check ---
    caption_records = all_records.get(args.caption_split, [])
    if caption_records and SOURCE_JSONL.exists():
        print_section(
            f"Caption samples ({args.caption_split}, {args.samples} per label)"
        )
        captions = sample_captions(caption_records, args.samples)
        print_captions(args.caption_split, captions)
    elif not SOURCE_JSONL.exists():
        print(f"\n[skip] Source JSONL not accessible — cannot sample captions")

    # --- Cross-split label consistency ---
    print_section("Cross-split label consistency")
    all_labels = set()
    for split, records in all_records.items():
        labels = set(r["label"] for r in records)
        all_labels |= labels
        print(f"  {split:<20} labels: {sorted(labels)}")

    print(f"\n  Union of all labels: {sorted(all_labels)}")

    # --- Image accessibility spot-check ---
    print_section("Image accessibility spot-check (first 20 records of benchmark_eval)")
    eval_records = all_records.get("benchmark_eval", [])
    if eval_records and SOURCE_JSONL.exists():
        check_indices = {r["source_index"] for r in eval_records[:20]}
        index_to_row = build_line_index(SOURCE_JSONL, check_indices)
        found = missing = 0
        for r in eval_records[:20]:
            row = index_to_row.get(r["source_index"])
            if row is None:
                missing += 1
                continue
            modalities = row.get("modalities") or []
            if not modalities:
                missing += 1
                continue
            image_value = modalities[0].get("value", "")
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = Path(r["source_root"]) / image_value
            if image_path.exists():
                found += 1
            else:
                missing += 1
                print(f"  [missing] {image_path}")
        print(f"  Accessible: {found}/20   Missing: {missing}/20")
    else:
        print("  [skip] No eval records or JSONL not accessible")


if __name__ == "__main__":
    main()
