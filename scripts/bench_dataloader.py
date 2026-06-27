"""Dataloader throughput benchmark.

Measures how fast the data pipeline can produce batches — without any GPU
compute — to determine whether training is CPU/IO-bound or GPU-bound.

Single-worker mode (recommended first run):
    python scripts/bench_dataloader.py \
        --config cookbook/sft/moe/attn/pep/stage2_sanitycheck_zero2.yaml \
        --num_workers 16 \
        --prefetch_factor 4 \
        --num_batches 100

Sweep mode (compare multiple worker counts):
    python scripts/bench_dataloader.py \
        --config cookbook/sft/moe/attn/pep/stage2_sanitycheck_zero2.yaml \
        --num_workers 2 4 8 16 \
        --prefetch_factor 4

Single mode reports: throughput (samples/s) + per-batch wait mean/p50/p95/p99 in ms.
Sweep mode prints a compact table.

Rule of thumb: if DataLoader throughput (samples/sec) < training throughput
at num_workers=production_value (16), the pipeline is IO/CPU-bound.
"""

import argparse
import os
import sys
import time
import yaml
import multiprocessing
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_raw_dataset(config: dict):
    """Load and concatenate all datasets from config, same logic as train.py."""
    from datasets import concatenate_datasets, load_from_disk, load_dataset, config as ds_cfg

    packed_datasets = []
    for ds_entry in config.get("datasets", []):
        path = ds_entry["packed_path"]
        info_file = os.path.join(path, ds_cfg.DATASET_INFO_FILENAME)
        state_file = os.path.join(path, ds_cfg.DATASET_STATE_JSON_FILENAME)
        if os.path.exists(info_file) and os.path.exists(state_file):
            ds = load_from_disk(path)
        else:
            _, ext = os.path.splitext(path)
            if ext in (".jsonl", ".json"):
                ds = load_dataset("json", data_files=path)["train"]
            elif ext == ".parquet":
                ds = load_dataset("parquet", data_files=path)["train"]
            else:
                ds = load_dataset(path)["train"]
        packed_datasets.append(ds)
        logger.info("  loaded %s  (%d samples)", path, len(ds))

    combined = concatenate_datasets(packed_datasets).shuffle(seed=42)
    logger.info("Total combined dataset: %d samples", len(combined))
    return combined


def dummy_collate(batch):
    """Minimal collate: return the raw list so we measure only IO, not collation."""
    return batch


def bench_workers(
    dataset,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    prefetch_factor: int = 4,
) -> dict:
    """Benchmark a DataLoader and return timing statistics.

    Returns a dict with keys:
        samples_per_sec  – total throughput
        mean_ms          – mean batch wait time in ms
        p50_ms           – median batch wait time in ms
        p95_ms           – 95th-percentile batch wait time in ms
        p99_ms           – 99th-percentile batch wait time in ms
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=False,
        collate_fn=dummy_collate,
        drop_last=True,
    )

    # warm-up: 5 batches
    warm_batches = 5
    it = iter(loader)
    for _ in range(warm_batches):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)

    # timed run — measure wall time of each individual batch fetch
    batch_times_ms: list = []
    total_samples = 0
    t_run_start = time.perf_counter()
    t_prev = t_run_start
    for i, batch in enumerate(loader):
        t_now = time.perf_counter()
        batch_times_ms.append((t_now - t_prev) * 1000.0)
        total_samples += len(batch)
        t_prev = t_now
        if i + 1 >= num_batches:
            break
    elapsed = time.perf_counter() - t_run_start

    arr = np.array(batch_times_ms)
    return {
        "samples_per_sec": total_samples / elapsed,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DataLoader throughput and per-batch latency")
    parser.add_argument("--config", required=True, help="Training YAML config path")
    parser.add_argument(
        "--num_workers",
        nargs="+",
        type=int,
        default=[16],
        help="num_workers value(s) to test. Pass multiple values for a sweep (e.g. --num_workers 2 4 8 16).",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="DataLoader prefetch_factor (default: 4). Applied to all workers > 0.",
    )
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8)
    parser.add_argument(
        "--num_batches", "--num-batches", type=int, default=100,
        help="Number of batches to time per measurement (default: 100)",
    )
    args = parser.parse_args()

    logger.info("Loading config: %s", args.config)
    config = load_config(args.config)

    logger.info("Building dataset...")
    dataset = build_raw_dataset(config)
    total = len(dataset)

    is_sweep = len(args.num_workers) > 1

    print(f"\n{'='*72}")
    print(f"Dataset:          {total:,} samples")
    print(f"Batch size:       {args.batch_size}")
    print(f"Batches measured: {args.num_batches}")
    print(f"Prefetch factor:  {args.prefetch_factor}")
    print(f"{'='*72}")

    if is_sweep:
        # Sweep mode: compact table showing throughput across worker counts
        print(f"{'num_workers':>12}  {'samples/sec':>14}  {'mean_ms':>10}  {'p50_ms':>10}  {'p95_ms':>10}  {'p99_ms':>10}")
        print(f"{'-'*72}")
        for nw in args.num_workers:
            try:
                stats = bench_workers(dataset, args.batch_size, nw, args.num_batches, args.prefetch_factor)
                flag = "  <-- possible bottleneck" if stats["samples_per_sec"] < 15.0 else ""
                print(
                    f"{nw:>12}  {stats['samples_per_sec']:>14.2f}  "
                    f"{stats['mean_ms']:>10.1f}  {stats['p50_ms']:>10.1f}  "
                    f"{stats['p95_ms']:>10.1f}  {stats['p99_ms']:>10.1f}{flag}"
                )
            except Exception as e:
                print(f"{nw:>12}  ERROR: {e}")
    else:
        # Single-value mode: detailed per-batch stats
        nw = args.num_workers[0]
        logger.info("Running single benchmark: num_workers=%d, prefetch_factor=%d", nw, args.prefetch_factor)
        try:
            stats = bench_workers(dataset, args.batch_size, nw, args.num_batches, args.prefetch_factor)
            print(f"\nResults (num_workers={nw}, prefetch_factor={args.prefetch_factor}):")
            print(f"  Throughput:       {stats['samples_per_sec']:.2f} samples/s")
            print(f"  Batch wait mean:  {stats['mean_ms']:.1f} ms")
            print(f"  Batch wait p50:   {stats['p50_ms']:.1f} ms")
            print(f"  Batch wait p95:   {stats['p95_ms']:.1f} ms")
            print(f"  Batch wait p99:   {stats['p99_ms']:.1f} ms")
            if stats["samples_per_sec"] < 15.0:
                print("\n  WARNING: throughput below 15 samples/s — DataLoader may be the bottleneck.")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n{'='*72}")
    print(
        "\nInterpretation:\n"
        "  If best samples/sec < training samples/sec, the DataLoader IS the bottleneck.\n"
        "  If samples/sec grows with num_workers, increase dataloader_num_workers.\n"
        "  If samples/sec plateaus early, IO (disk/capstor) is the limit.\n"
        "  Production training typically achieves ~7-15 samples/sec at bs=8.\n"
        "  Target: p95 batch wait << GPU step time (~1350ms on GH200).\n"
    )


if __name__ == "__main__":
    main()
