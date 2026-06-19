#!/usr/bin/env python
"""Evaluate pretrained domain experts on the existing classification benchmarks."""


import argparse
import csv
import os
import sys
from pathlib import Path


EXPERT_ROOT = Path("/lightscratch/users/nemo/models")
DEFAULT_RESULTS_PATH = Path(
    "src/multimeditron/experts/logs/expert_baseline_results.csv"
)

DOMAIN_TO_EXPERT = {
    "ct": "CT_expert",
    "mri": "MRI_expert",
    "skin": "Skin_expert",
    "ophthalmology": "Ophthalmology_expert",
    "ultrasound": "US_expert",
    "xray": "XR_expert",
}

SMOKE_LIMIT_ENV = {
    "ct": ("CT_MAX_TRAIN_EXAMPLES", "CT_MAX_TEST_EXAMPLES"),
    "mri": ("MRI_MAX_TRAIN_EXAMPLES", "MRI_MAX_TEST_EXAMPLES"),
    "skin": ("SKIN_INTEGRATED_MAX_TRAIN_EXAMPLES", "SKIN_INTEGRATED_MAX_TEST_EXAMPLES"),
    "ophthalmology": ("OPHTH_MAX_TRAIN_EXAMPLES", "OPHTH_MAX_TEST_EXAMPLES"),
    "ultrasound": ("ULTRASOUND_MAX_TRAIN_EXAMPLES", "ULTRASOUND_MAX_TEST_EXAMPLES"),
    "xray": ("XRAY_MAX_TRAIN_EXAMPLES", "XRAY_MAX_TEST_EXAMPLES"),
}


def _add_eval_pipeline_to_path():
    eval_dir = Path(__file__).resolve().parent / "evaluation_pipeline"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained domain experts on matching benchmarks."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DOMAIN_TO_EXPERT),
        choices=list(DOMAIN_TO_EXPERT),
        help="Benchmarks to evaluate.",
    )
    parser.add_argument(
        "--expert_root",
        type=Path,
        default=EXPERT_ROOT,
        help="Directory containing *_expert checkpoint folders.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Single checkpoint path evaluated on all domains (overrides --expert_root + per-domain mapping).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Where to write scores.",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Optional per-domain train cap for a quick smoke test.",
    )
    parser.add_argument(
        "--max_test_examples",
        type=int,
        default=None,
        help="Optional per-domain test cap for a quick smoke test.",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Recompute embeddings instead of reusing benchmark cache files.",
    )
    return parser.parse_args()


def apply_example_caps(domains, max_train, max_test):
    for domain in domains:
        train_env, test_env = SMOKE_LIMIT_ENV[domain]
        if max_train is not None:
            os.environ[train_env] = str(max_train)
        if max_test is not None:
            os.environ[test_env] = str(max_test)


def main():
    args = parse_args()
    _add_eval_pipeline_to_path()
    apply_example_caps(args.domains, args.max_train_examples, args.max_test_examples)

    from evaluation_pipeline.build_benchmarks import build_benchmarks_from_names

    benchmarks = build_benchmarks_from_names(args.domains)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for domain, benchmark in zip(args.domains, benchmarks):
        if args.model_path is not None:
            expert_path = args.model_path
        else:
            expert_path = args.expert_root / DOMAIN_TO_EXPERT[domain]
        if not expert_path.exists():
            raise FileNotFoundError(f"Missing expert checkpoint: {expert_path}")

        print(f"Evaluating {expert_path} on {domain} ({benchmark.__class__.__name__})")
        result = benchmark.evaluate(str(expert_path), use_cache=not args.no_cache)
        print(f"{domain}: {result}")
        rows.append(
            {
                "domain": domain,
                "benchmark": benchmark.__class__.__name__,
                "expert_path": str(expert_path),
                **result,
            }
        )

    fixed_fields = ["domain", "benchmark", "expert_path"]
    seen = dict.fromkeys(k for row in rows for k in row if k not in fixed_fields)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fixed_fields + list(seen), extrasaction="ignore", restval=""
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote results to {args.output_csv}")


if __name__ == "__main__":
    main()
