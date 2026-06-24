#!/usr/bin/env python3
"""Build benchmark manifests for CT2D-glob-derived benchmarks.

Uses CT2D-glob-rawpath2905.jsonl to extract reliable labels from original
filenames, then matches entries in the HuggingFace CT2D-glob dataset by text
content to recover source_index values.

Two benchmarks are built:

  histopathology — TCGA cancer-type classification (33 classes)
    Files named {CancerType}-{CancerType}-{digit}-TCGA-...jpg
    Output: benchmark_splits/multimediset/histopathology/

  ct — COVID-19 vs healthy lung, binary (2 classes, clean rawpath labels)
    Files named seg_train_...-{pathology}.nii-{slice}.jpg
    Replaces the unreliable caption-based ct manifests.
    Output: benchmark_splits/multimediset/ct/

Each benchmark produces four JSONL split files:
  train_model.jsonl    80 % of labelled HF train split
  mlp_train.jsonl      10 % of labelled HF train split
  benchmark_eval.jsonl 10 % of labelled HF train split
  holdout_test.jsonl  100 % of labelled HF test split
"""


import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

RAWPATH_JSONL = Path(
    os.environ.get(
        "CT2D_RAWPATH_JSONL",
        "/lightscratch/users/nemo/datasets/CT_data/CT2D-glob/old/CT2D-glob-rawpath2905.jsonl",
    )
)
HF_DATASET_ROOT = Path(
    os.environ.get(
        "CT2D_HF_DATASET_ROOT",
        "/lightscratch/datasets/MultiMediset/general_purpose/CT2D-glob",
    )
)
DEFAULT_HISTO_OUTPUT_DIR = Path("benchmark_splits/multimediset/histopathology")
DEFAULT_CT_OUTPUT_DIR = Path("benchmark_splits/multimediset/ct")

SPLIT_NAMES = ("train_model", "mlp_train", "benchmark_eval")
SPLIT_NAMES_WITH_HOLDOUT = SPLIT_NAMES + ("holdout_test",)


# ---------------------------------------------------------------------------
# Label extractors
# ---------------------------------------------------------------------------


def extract_tcga_label(rawpath):
    """Cancer type from a TCGA filename, or None.

    Matches:  CancerType-CancerType-digit-TCGA-...jpg  (type repeated twice)
    Rejects:  .nii- (real CT slice)
              MSS_JPEG-blk-...-TCGA- (no type repetition)
    """
    filename = rawpath.split("/")[-1]
    if ".nii-" in filename:
        return None
    parts = filename.split("-")
    if (
        len(parts) >= 4
        and parts[0] == parts[1]
        and parts[2].isdigit()
        and "TCGA" in filename
    ):
        return parts[0].replace("_", " ")
    return None


def extract_tcga_patient_id(rawpath):
    """TCGA patient barcode (TCGA-{TSS}-{Patient}) from a TCGA filename, or None.

    Filename format: CancerType-CancerType-digit-TCGA-{TSS}-{Patient}-...jpg
    Example: BLCA-BLCA-1-TCGA-ZF-AA56-01A-... → TCGA-ZF-AA56
    """
    filename = rawpath.split("/")[-1]
    parts = filename.split("-")
    try:
        tcga_idx = parts.index("TCGA")
    except ValueError:
        return None
    if tcga_idx + 2 < len(parts):
        return f"TCGA-{parts[tcga_idx + 1]}-{parts[tcga_idx + 2]}"
    return None


def extract_ct_label(rawpath):
    """CT pathology label from a NIfTI-slice filename, or None.

    Matches:  seg_train_{id}-{pathology}.nii-{slice}.jpg
    Returns the pathology string verbatim, e.g. "covid-19 infection" or "right lung".
    """
    filename = rawpath.split("/")[-1]
    if ".nii-" not in filename:
        return None
    # everything between the first '-' and '.nii-'
    before_nii = filename.split(".nii-")[0]  # seg_train_9336_b_1-covid-19 infection
    dash_idx = before_nii.find("-")
    if dash_idx == -1:
        return None
    return before_nii[dash_idx + 1 :]  # covid-19 infection  /  right lung


def extract_ct_volume_id(rawpath):
    """Volume ID from a NIfTI-slice CT filename, or None.

    Matches:  seg_train_{id}-{pathology}.nii-{slice}.jpg
    Returns:  seg_train_{id}  (all slices of one 3-D scan share this prefix)
    """
    filename = rawpath.split("/")[-1]
    if ".nii-" not in filename:
        return None
    before_nii = filename.split(".nii-")[0]  # seg_train_9336_b_1-covid-19 infection
    dash_idx = before_nii.find("-")
    if dash_idx == -1:
        return None
    return before_nii[:dash_idx]  # seg_train_9336_b_1


# ---------------------------------------------------------------------------
# Index building (text → label, streaming)
# ---------------------------------------------------------------------------


def build_text_label_index(
    rawpath_jsonl,
    extractor,
    desc,
    patient_id_extractor=None,
):
    """Stream rawpath JSONL, apply extractor to each rawpath.

    Returns (label_index, patient_index) where:
      label_index   : {text → label}
      patient_index : {text → patient_id}  (empty dict when patient_id_extractor is None)

    Texts that map to two different labels are dropped (conflict resolution).
    """
    label_index = {}
    patient_index = {}
    conflicts = set()

    with rawpath_jsonl.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc=desc, unit="lines"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            rawpath = (row.get("modalities") or [{}])[0].get("value", "")
            label = extractor(rawpath)
            if label is None:
                continue

            text = row.get("text", "")
            if not text:
                continue

            if text in label_index:
                if label_index[text] != label:
                    conflicts.add(text)
            else:
                label_index[text] = label
                if patient_id_extractor is not None:
                    pid = patient_id_extractor(rawpath)
                    if pid is not None:
                        patient_index[text] = pid

    for text in conflicts:
        del label_index[text]
        patient_index.pop(text, None)

    return label_index, patient_index


# ---------------------------------------------------------------------------
# Manifest record + split helpers
# ---------------------------------------------------------------------------


def make_record(
    *,
    benchmark,
    source_root,
    source_split,
    source_index,
    label,
    label_id,
    group_id=None,
):
    return {
        "benchmark": benchmark,
        "dataset": "CT2D-glob",
        "group_id": group_id,
        "label": label,
        "label_id": label_id,
        "source_index": source_index,
        "source_root": str(source_root),
        "source_split": source_split,
    }


def ratio_counts(total, plan):
    raw = {split: total * ratio for split, ratio in plan}
    counts = {split: int(v) for split, v in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw.items(), key=lambda x: x[1] - int(x[1]), reverse=True)
    for split, _ in order[:remainder]:
        counts[split] += 1
    return counts


def assign_records(
    records,
    plan,
    rng,
):
    shuffled = records.copy()
    rng.shuffle(shuffled)
    counts = ratio_counts(len(shuffled), plan)
    assigned = {split: [] for split, _ in plan}
    offset = 0
    for split, _ in plan:
        count = counts[split]
        assigned[split] = shuffled[offset : offset + count]
        offset += count
    return assigned


def assign_grouped_records(records, plan, rng):
    """Split records by group_id so every image from one patient goes to the same split."""
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record.get("group_id"))].append(record)

    groups = list(grouped.items())
    rng.shuffle(groups)
    target_counts = ratio_counts(len(records), plan)
    assigned = {split: [] for split, _ in plan}
    split_order = [split for split, _ in plan]
    split_idx = 0
    for _group_id, group_records in groups:
        split = split_order[min(split_idx, len(split_order) - 1)]
        assigned[split].extend(group_records)
        if (
            split_idx < len(split_order) - 1
            and len(assigned[split]) >= target_counts[split]
        ):
            split_idx += 1
    return assigned


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Core split builder (shared by both benchmarks)
# ---------------------------------------------------------------------------


def build_splits(
    *,
    benchmark_name,
    text_to_label,
    hf_root,
    output_dir,
    seed,
    max_per_class,
    patient_index=None,
    holdout=False,
):
    all_labels = sorted(set(text_to_label.values()))
    label_to_id = {label: idx for idx, label in enumerate(all_labels)}
    print(f"  {len(text_to_label):,} unique text entries  /  {len(all_labels)} classes")
    for label, idx in label_to_id.items():
        print(f"    [{idx:02d}] {label}")

    split_names = SPLIT_NAMES_WITH_HOLDOUT if holdout else SPLIT_NAMES

    ds = load_from_disk(str(hf_root))

    source_keys = set(ds.keys())
    if {"train", "test"}.issubset(source_keys):
        if holdout:
            plan_by_source = {
                "train": [("train_model", 0.8), ("mlp_train", 0.1), ("benchmark_eval", 0.1)],
                "test": [("holdout_test", 1.0)],
            }
        else:
            plan_by_source = {
                "train": [("train_model", 0.8), ("mlp_train", 0.1)],
                "test": [("benchmark_eval", 1.0)],
            }
    else:
        if holdout:
            source_plan = [
                ("train_model", 0.7),
                ("mlp_train", 0.1),
                ("benchmark_eval", 0.1),
                ("holdout_test", 0.1),
            ]
        else:
            source_plan = [
                ("train_model", 0.7),
                ("mlp_train", 0.15),
                ("benchmark_eval", 0.15),
            ]
        plan_by_source = {s: source_plan for s in sorted(source_keys)}

    target_records = {split: [] for split in split_names}

    for source_split in sorted(ds.keys()):
        split_data = ds[source_split].select_columns(["text", "modalities"])
        total = len(split_data)
        records = []

        print(f"\n  Scanning '{source_split}' ({total:,} rows) …")
        offset = 0
        for batch in tqdm(
            split_data.iter(batch_size=50_000),
            total=(total + 49_999) // 50_000,
            desc=f"  {source_split}",
            unit="batch",
        ):
            for local_idx, text in enumerate(batch["text"]):
                label = text_to_label.get(text)
                if label is None:
                    continue
                records.append(
                    make_record(
                        benchmark=benchmark_name,
                        source_root=hf_root,
                        source_split=source_split,
                        source_index=offset + local_idx,
                        label=label,
                        label_id=label_to_id[label],
                        group_id=patient_index.get(text) if patient_index else None,
                    )
                )
            offset += len(batch["text"])

        label_counts = Counter(r["label"] for r in records)
        print(
            f"  → {len(records):,} labelled examples  {dict(sorted(label_counts.items()))}"
        )

        plan = plan_by_source.get(source_split)
        if plan is None:
            continue

        by_label = defaultdict(list)
        for record in records:
            by_label[record["label"]].append(record)

        rng = random.Random(f"{seed}:{benchmark_name}:CT2D-glob:{source_split}")
        balanced = []
        for label_records in by_label.values():
            shuffled = label_records.copy()
            rng.shuffle(shuffled)
            if max_per_class is not None:
                shuffled = shuffled[:max_per_class]
            balanced.extend(shuffled)

        has_groups = patient_index and any(
            r["group_id"] is not None for r in balanced
        )
        assigned = (
            assign_grouped_records(balanced, plan, rng)
            if has_groups
            else assign_records(balanced, plan, rng)
        )
        for target_split, split_records in assigned.items():
            target_records[target_split].extend(split_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in split_names:
        path = output_dir / f"{split_name}.jsonl"
        write_jsonl(path, target_records[split_name])
        print(
            f"  {split_name:20s}  {len(target_records[split_name]):>8,} records  →  {path}"
        )

    summary = {
        "benchmark": benchmark_name,
        "seed": seed,
        "source_rawpath_jsonl": str(RAWPATH_JSONL),
        "source_hf_dataset": str(hf_root),
        "num_labels": len(all_labels),
        "labels": label_to_id,
        "splits": {split: len(records) for split, records in target_records.items()},
        "label_distribution": {
            split: dict(Counter(r["label"] for r in records))
            for split, records in target_records.items()
        },
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rawpath-jsonl", type=Path, default=RAWPATH_JSONL)
    parser.add_argument("--hf-dataset-root", type=Path, default=HF_DATASET_ROOT)
    parser.add_argument(
        "--histo-output-dir", type=Path, default=DEFAULT_HISTO_OUTPUT_DIR
    )
    parser.add_argument("--ct-output-dir", type=Path, default=DEFAULT_CT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap examples per class per source split (None = use all).",
    )
    parser.add_argument(
        "--benchmark",
        choices=["histopathology", "ct", "all"],
        default="all",
        help="Which benchmark(s) to build (default: all).",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        default=False,
        help=(
            "Generate a holdout_test.jsonl split in addition to the standard three. "
            "When the source dataset has a dedicated test split it is used as-is; "
            "otherwise 10 %% is carved from train."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for path in (args.rawpath_jsonl, args.hf_dataset_root):
        if not path.exists():
            print(f"ERROR: not found: {path}", file=sys.stderr)
            sys.exit(1)

    build_histo = args.benchmark in ("histopathology", "all")
    build_ct = args.benchmark in ("ct", "all")

    # Build rawpath indices (one pass per benchmark to keep memory bounded)
    if build_histo:
        print("\n=== Histopathology benchmark ===")
        print("Building text→label index (TCGA cancer types) …")
        histo_index, histo_patient_index = build_text_label_index(
            args.rawpath_jsonl,
            extract_tcga_label,
            desc="indexing TCGA labels",
            patient_id_extractor=extract_tcga_patient_id,
        )
        print(f"  patient IDs found: {len(histo_patient_index):,}")
        build_splits(
            benchmark_name="histopathology",
            text_to_label=histo_index,
            hf_root=args.hf_dataset_root,
            output_dir=args.histo_output_dir,
            seed=args.seed,
            max_per_class=args.max_per_class,
            patient_index=histo_patient_index,
            holdout=args.holdout,
        )
        del histo_index, histo_patient_index

    if build_ct:
        print("\n=== CT benchmark (rawpath, binary) ===")
        print("Building text→label index (CT NIfTI slices) …")
        ct_index, ct_volume_index = build_text_label_index(
            args.rawpath_jsonl,
            extract_ct_label,
            desc="indexing CT labels",
            patient_id_extractor=extract_ct_volume_id,
        )
        print(f"  volume IDs found: {len(ct_volume_index):,}")
        build_splits(
            benchmark_name="ct",
            text_to_label=ct_index,
            hf_root=args.hf_dataset_root,
            output_dir=args.ct_output_dir,
            seed=args.seed,
            max_per_class=args.max_per_class,
            patient_index=ct_volume_index,
            holdout=args.holdout,
        )
        del ct_index, ct_volume_index

    print("\nDone.")


if __name__ == "__main__":
    main()
