#!/usr/bin/env python3

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_multimediset_labels import (  # noqa: E402
    DEFAULT_BASE_ROOT,
    DEFAULT_MRI_ROOT,
    TARGET_BENCHMARKS,
    load_dataset_like,
    make_specs,
    row_at,
    split_size,
)


DEFAULT_RULES_PATH = Path("config/multimediset_label_rules.json")
DEFAULT_OUTPUT_DIR = Path("benchmark_splits/multimediset")
SPLIT_NAMES = ("train_model", "mlp_train", "benchmark_eval")


def load_rules(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def benchmark_label_maps(rules):
    labels_by_benchmark = defaultdict(set)
    for dataset_name, config in rules["datasets"].items():
        benchmark = config["benchmark"]
        for label in config["labels"]:
            labels_by_benchmark[benchmark].add(label)

    return {
        benchmark: {label: idx for idx, label in enumerate(sorted(labels))}
        for benchmark, labels in sorted(labels_by_benchmark.items())
    }


def source_split_plan(source_splits, holdout=False):
    if {"train", "test"}.issubset(source_splits):
        if holdout:
            # test becomes holdout; benchmark_eval carved from train
            return {
                "train": (
                    ("train_model", 0.8),
                    ("mlp_train", 0.1),
                    ("benchmark_eval", 0.1),
                ),
                "test": (("holdout_test", 1.0),),
            }
        # test is split evenly between mlp_train and benchmark_eval so that
        # both sets have comparable sizes when the source test split is large.
        return {
            "train": (
                ("train_model", 0.8),
                ("mlp_train", 0.1),
                ("benchmark_eval", 0.1),
            ),
            "test": (("mlp_train", 0.7), ("benchmark_eval", 0.3)),
        }
    if {"train", "val"}.issubset(source_splits):
        if holdout:
            # val becomes the held-out set; carve benchmark_eval from train
            return {
                "train": (
                    ("train_model", 0.8),
                    ("mlp_train", 0.1),
                    ("benchmark_eval", 0.1),
                ),
                "val": (("holdout_test", 1.0),),
            }
        return {
            "train": (("train_model", 0.9), ("mlp_train", 0.1)),
            "val": (("benchmark_eval", 1.0),),
        }
    if holdout:
        base_ratios = (
            ("train_model", 0.7),
            ("mlp_train", 0.1),
            ("benchmark_eval", 0.1),
            ("holdout_test", 0.1),
        )
    else:
        base_ratios = (
            ("train_model", 0.7),
            ("mlp_train", 0.15),
            ("benchmark_eval", 0.15),
        )
    if "train" in source_splits:
        return {"train": base_ratios}
    return {split: base_ratios for split in sorted(source_splits)}


def ratio_counts(total, plan):
    if not plan:
        return {}
    raw_counts = [(split, total * ratio) for split, ratio in plan]
    counts = {split: int(raw) for split, raw in raw_counts}
    remainder = total - sum(counts.values())
    order = sorted(raw_counts, key=lambda item: item[1] - int(item[1]), reverse=True)
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


def assign_grouped_records(
    records,
    plan,
    rng,
):
    grouped = defaultdict(list)
    for record in records:
        group_id = record.get("group_id")
        grouped[str(group_id)].append(record)

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


def assign_mixed_records(records, plan, rng):
    """Variant of assign_grouped_records for datasets with a partial group_key.

    Records with a non-None group_id are split by group (patient-level).
    Records with group_id=None are split randomly.
    Both results are combined into the same target splits.
    """
    grouped = [r for r in records if r.get("group_id") is not None]
    ungrouped = [r for r in records if r.get("group_id") is None]

    assigned = {split: [] for split, _ in plan}

    if grouped:
        for split, recs in assign_grouped_records(grouped, plan, rng).items():
            assigned[split].extend(recs)
    if ungrouped:
        for split, recs in assign_records(ungrouped, plan, rng).items():
            assigned[split].extend(recs)

    return assigned


def make_record(
    *,
    benchmark,
    dataset_name,
    source_root,
    source_split,
    source_index,
    label,
    label_id,
    group_id,
    subdataset=None,
):
    return {
        "benchmark": benchmark,
        "dataset": dataset_name,
        "source_root": str(source_root),
        "source_split": source_split,
        "source_index": source_index,
        "label": label,
        "label_id": label_id,
        "group_id": group_id,
        "subdataset": subdataset,
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def summarize_records(records_by_split):
    summary = {}
    for split_name, records in records_by_split.items():
        label_counts = Counter(record["label"] for record in records)
        dataset_counts = Counter(record["dataset"] for record in records)
        benchmark_counts = Counter(record["benchmark"] for record in records)
        summary[split_name] = {
            "num_examples": len(records),
            "by_label": dict(sorted(label_counts.items())),
            "by_dataset": dict(sorted(dataset_counts.items())),
            "by_benchmark": dict(sorted(benchmark_counts.items())),
        }
    return summary


def build_splits(args):
    rules = load_rules(args.rules)
    label_maps = benchmark_label_maps(rules)
    specs = make_specs(args.base_root, args.mri_root)
    specs_by_name = {spec.name: spec for spec in specs}

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = SPLIT_NAMES + ("holdout_test",) if args.holdout else SPLIT_NAMES
    benchmark_records = {
        benchmark: {split: [] for split in split_names}
        for benchmark in TARGET_BENCHMARKS
    }
    skipped_summary = {}

    for dataset_name, dataset_config in sorted(rules["datasets"].items()):
        if dataset_name not in specs_by_name:
            raise KeyError(f"No DatasetSpec found for {dataset_name}")
        spec = specs_by_name[dataset_name]
        benchmark = dataset_config["benchmark"]
        if args.benchmarks is not None and benchmark not in args.benchmarks:
            print(f"Skipping {dataset_name} (benchmark={benchmark}, not in --benchmarks filter)")
            continue
        label_to_id = label_maps[benchmark]
        print(f"Loading {dataset_name} for benchmark {benchmark}...")
        dataset = load_dataset_like(spec.root)
        plan_by_source_split = source_split_plan(set(dataset.keys()), holdout=args.holdout)

        skipped_by_split = {}
        total_by_source_split = {}
        kept_by_source_split = {}
        seen_group_ids = set()

        for source_split, split_data in dataset.items():
            if source_split not in plan_by_source_split:
                continue
            records = []
            skipped_missing_label = 0
            total = split_size(split_data)
            scan_total = total
            if args.max_source_examples_per_split is not None:
                scan_total = min(total, args.max_source_examples_per_split)
            print(f"  scanning {source_split}: {scan_total}/{total} example(s)")
            for idx in range(scan_total):
                row = row_at(split_data, idx)
                label = spec.extractor(row)
                if label is None:
                    skipped_missing_label += 1
                    continue
                group_id = None
                if spec.group_key is not None:
                    if callable(spec.group_key):
                        gid = spec.group_key(row)
                        if gid is not None:
                            group_id = str(gid)
                    elif row.get(spec.group_key) is not None:
                        group_id = str(row.get(spec.group_key))
                subdataset = (
                    spec.subdataset_extractor(row)
                    if spec.subdataset_extractor is not None
                    else None
                )
                records.append(
                    make_record(
                        benchmark=benchmark,
                        dataset_name=dataset_name,
                        source_root=spec.root,
                        source_split=source_split,
                        source_index=idx,
                        label=label,
                        label_id=label_to_id[label],
                        group_id=group_id,
                        subdataset=subdataset,
                    )
                )

            if spec.group_key is not None and seen_group_ids:
                n_before = len(records)
                records = [
                    r for r in records
                    if r["group_id"] is None or r["group_id"] not in seen_group_ids
                ]
                n_removed = n_before - len(records)
                if n_removed:
                    print(f"  [dedup] {n_removed} record(s) excluded ({source_split}): same patient already in a previous split")
            if spec.group_key is not None:
                seen_group_ids.update(r["group_id"] for r in records if r["group_id"] is not None)

            seed_material = f"{args.seed}:{benchmark}:{dataset_name}:{source_split}"
            rng = random.Random(seed_material)
            plan = plan_by_source_split[source_split]
            if spec.group_key is not None:
                assigned = assign_mixed_records(records, plan, rng)
            else:
                assigned = assign_records(records, plan, rng)

            for target_split, assigned_records in assigned.items():
                benchmark_records[benchmark][target_split].extend(assigned_records)

            total_by_source_split[source_split] = scan_total
            kept_by_source_split[source_split] = len(records)
            skipped_by_split[source_split] = skipped_missing_label

        skipped_summary[dataset_name] = {
            "benchmark": benchmark,
            "source_root": str(spec.root),
            "source_totals": total_by_source_split,
            "kept_labeled": kept_by_source_split,
            "skipped_missing_label": skipped_by_split,
            "split_plan": {
                source_split: dict(plan)
                for source_split, plan in plan_by_source_split.items()
            },
        }

    for benchmark, records_by_split in benchmark_records.items():
        if args.benchmarks is not None and benchmark not in args.benchmarks:
            continue
        benchmark_dir = output_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        for split_name in split_names:
            write_jsonl(
                benchmark_dir / f"{split_name}.jsonl", records_by_split[split_name]
            )

    split_summary = {
        "seed": args.seed,
        "rules": str(args.rules),
        "output_dir": str(output_dir),
        "target_benchmarks": TARGET_BENCHMARKS,
        "label_maps": label_maps,
        "datasets": skipped_summary,
        "benchmarks": {
            benchmark: summarize_records(records_by_split)
            for benchmark, records_by_split in benchmark_records.items()
        },
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return split_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build clean JSONL manifests for MultiMediset benchmark splits."
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--mri-root", type=Path, default=DEFAULT_MRI_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        metavar="BENCHMARK",
        help="Only rebuild splits for these benchmarks (e.g. --benchmarks skin eye ultrasound).",
    )
    parser.add_argument(
        "--max-source-examples-per-split",
        type=int,
        default=None,
        help="Limit scanned source examples per dataset split. Intended for smoke tests.",
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
    summary = build_splits(args)
    print(f"Wrote manifests under {args.output_dir}")
    for benchmark, splits in summary["benchmarks"].items():
        counts = {
            split_name: split_summary["num_examples"]
            for split_name, split_summary in splits.items()
        }
        print(f"{benchmark}: {counts}")


if __name__ == "__main__":
    main()
