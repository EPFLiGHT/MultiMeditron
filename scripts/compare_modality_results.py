#!/usr/bin/env python3
"""Compare per-modality GMAI evaluation results between two MultiMeditron models.

Reads the latest *_results.json from each model's eval output directory and
produces a side-by-side Markdown table of accuracy per modality sub-task.

Usage:
    python scripts/compare_modality_results.py [--results-root DIR]

The script auto-discovers the two model directories under --results-root:
  - MultiMeditron-8B-attn-pep-end2end__checkpoint-3063   (5-expert baseline)
  - MultiMeditron-8B-attn-pep-end2end-7exp__checkpoint-800 (7-expert)

You can also pass explicit paths:
    python scripts/compare_modality_results.py --model-a /path/to/dir --model-b /path/to/dir
"""
import argparse
import json
import os
import sys
from pathlib import Path

RESULTS_ROOT = "/users/surech/meditron/reports/lmms_eval_results"
MODEL_A_DIR = "MultiMeditron-8B-attn-pep-end2end__checkpoint-3063"
MODEL_B_DIR = "MultiMeditron-8B-attn-pep-end2end-7exp__checkpoint-800"

# Modality tasks in display order (matches expert domains)
MODALITY_TASKS = [
    "gmai_ct",
    "gmai_mri",
    "gmai_xray",
    "gmai_ultrasound",
    "gmai_endoscopy",
    "gmai_histopathology",
    "gmai_fundus",
    "gmai_microscopy",
    "gmai_dermoscopy",
    "gmai_oct",
]

# Department tasks
DEPARTMENT_TASKS = [
    "gmai_ophthalmology",
    "gmai_dermatology",
]

# Known sample counts per modality (from GMAI-MMBench-val analysis)
SAMPLE_COUNTS = {
    "gmai_ct": 934,
    "gmai_mri": 784,
    "gmai_endoscopy": 654,
    "gmai_histopathology": 624,
    "gmai_fundus": 409,
    "gmai_xray": 395,
    "gmai_microscopy": 218,
    "gmai_dermoscopy": 210,
    "gmai_ultrasound": 105,
    "gmai_oct": 95,
    "gmai_ophthalmology": 569,
    "gmai_dermatology": 185,
}


def find_latest_results(directory: str) -> dict | None:
    """Find and load the most recent *_results.json in a directory."""
    result_dir = Path(directory)
    if not result_dir.exists():
        return None

    result_files = sorted(result_dir.glob("*_results.json"))
    if not result_files:
        return None

    # Take the latest by filename (timestamp-based naming)
    latest = result_files[-1]
    with open(latest) as f:
        data = json.load(f)
    return data.get("results", {})


def collect_all_results(directory: str) -> dict:
    """Collect results from all *_results.json files, merging task results.

    When multiple result files exist (from separate eval runs), this merges
    them all, with later files overriding earlier ones for the same task.
    """
    result_dir = Path(directory)
    if not result_dir.exists():
        return {}

    merged = {}
    for result_file in sorted(result_dir.glob("*_results.json")):
        with open(result_file) as f:
            data = json.load(f)
        for task_name, task_data in data.get("results", {}).items():
            merged[task_name] = task_data

    return merged


def format_accuracy(results: dict, task: str) -> str:
    """Extract accuracy for a task, return formatted string or '—'."""
    if task not in results:
        return "—"
    acc = results[task].get("accuracy,none")
    if acc is None:
        return "—"
    return f"{acc:.1f}%"


def raw_accuracy(results: dict, task: str) -> float | None:
    """Extract raw accuracy float for a task."""
    if task not in results:
        return None
    return results[task].get("accuracy,none")


def format_delta(a_val: float | None, b_val: float | None) -> str:
    """Format the delta between two accuracy values."""
    if a_val is None or b_val is None:
        return "—"
    delta = b_val - a_val
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}"


def print_table(results_a: dict, results_b: dict, label_a: str, label_b: str):
    """Print a Markdown comparison table."""
    all_tasks = MODALITY_TASKS + DEPARTMENT_TASKS

    # Header
    print(f"\n## Per-Modality GMAI Accuracy: {label_a} vs {label_b}\n")
    print(f"| Task | Samples | {label_a} | {label_b} | Delta |")
    print("|---|---:|---:|---:|---:|")

    for task in all_tasks:
        samples = SAMPLE_COUNTS.get(task, "?")
        a_str = format_accuracy(results_a, task)
        b_str = format_accuracy(results_b, task)
        a_raw = raw_accuracy(results_a, task)
        b_raw = raw_accuracy(results_b, task)
        delta = format_delta(a_raw, b_raw)
        display_name = task.replace("gmai_", "").replace("_", " ").title()
        print(f"| {display_name} | {samples} | {a_str} | {b_str} | {delta} |")

    # Weighted average across modalities that have results in both
    a_total_correct = 0
    a_total_samples = 0
    b_total_correct = 0
    b_total_samples = 0
    for task in MODALITY_TASKS:
        a_raw = raw_accuracy(results_a, task)
        b_raw = raw_accuracy(results_b, task)
        n = SAMPLE_COUNTS.get(task, 0)
        if a_raw is not None:
            a_total_correct += a_raw * n / 100
            a_total_samples += n
        if b_raw is not None:
            b_total_correct += b_raw * n / 100
            b_total_samples += n

    if a_total_samples > 0 and b_total_samples > 0:
        a_avg = a_total_correct / a_total_samples * 100
        b_avg = b_total_correct / b_total_samples * 100
        delta_avg = b_avg - a_avg
        sign = "+" if delta_avg >= 0 else ""
        print(f"| **Weighted Avg (modality)** | **{a_total_samples}** | **{a_avg:.1f}%** | **{b_avg:.1f}%** | **{sign}{delta_avg:.1f}** |")

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare per-modality GMAI results")
    parser.add_argument(
        "--results-root",
        default=RESULTS_ROOT,
        help="Root directory containing model result subdirectories",
    )
    parser.add_argument("--model-a", default=None, help="Explicit path to model A results dir")
    parser.add_argument("--model-b", default=None, help="Explicit path to model B results dir")
    parser.add_argument(
        "--label-a", default="5-exp (ckpt-3063)", help="Display label for model A"
    )
    parser.add_argument(
        "--label-b", default="7-exp (ckpt-800)", help="Display label for model B"
    )
    args = parser.parse_args()

    dir_a = args.model_a or os.path.join(args.results_root, MODEL_A_DIR)
    dir_b = args.model_b or os.path.join(args.results_root, MODEL_B_DIR)

    print(f"Model A: {dir_a}")
    print(f"Model B: {dir_b}")

    results_a = collect_all_results(dir_a)
    results_b = collect_all_results(dir_b)

    if not results_a and not results_b:
        print("\nNo results found for either model. Jobs may still be running.", file=sys.stderr)
        print("Check with: squeue --me", file=sys.stderr)
        sys.exit(1)

    available_a = [t for t in MODALITY_TASKS + DEPARTMENT_TASKS if t in results_a]
    available_b = [t for t in MODALITY_TASKS + DEPARTMENT_TASKS if t in results_b]
    print(f"\nModel A tasks available: {len(available_a)}/{len(MODALITY_TASKS) + len(DEPARTMENT_TASKS)}")
    print(f"Model B tasks available: {len(available_b)}/{len(MODALITY_TASKS) + len(DEPARTMENT_TASKS)}")

    print_table(results_a, results_b, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
