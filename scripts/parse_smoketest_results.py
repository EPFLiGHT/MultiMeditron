#!/usr/bin/env python3
"""Parse smoke-test job logs and produce a GPU usage analysis table.

Usage (run after all smoke jobs finish):
    python scripts/parse_smoketest_results.py --job-ids 2069820 2069821 2069822 2069823 2069824 2069825

Or let it auto-discover by matching job names:
    python scripts/parse_smoketest_results.py --auto

Output columns:
  - Job / config name
  - ZeRO stage / strategy
  - Nodes
  - samples/sec  (from Trainer log)
  - steps/sec
  - Mean SM%     (from nvidia-smi dmon)
  - Mean mem-bw% (from nvidia-smi dmon)
  - Mean power W (from nvidia-smi dmon)
  - Peak VRAM GB (from nvidia-smi dmon)
  - Mean temp C  (from nvidia-smi dmon)
"""

import argparse
import os
import re
import sys
import glob
from pathlib import Path
from statistics import mean

REPORTS_DIR = Path("/users/surech/meditron/reports")
GPU_UTIL_DIR = REPORTS_DIR  # gpu-util-<jobid>/ lives here too


def find_log_files(job_id: str):
    out = list(REPORTS_DIR.glob(f"R-*.{job_id}.out"))
    err = list(REPORTS_DIR.glob(f"R-*.{job_id}.err"))
    return out[0] if out else None, err[0] if err else None


def parse_trainer_metrics(out_path: Path) -> dict:
    """Extract samples/sec, steps/sec from HuggingFace Trainer JSON log lines."""
    metrics = {"samples_per_sec": [], "steps_per_sec": [], "loss": []}
    if out_path is None or not out_path.exists():
        return {}
    with open(out_path) as f:
        for line in f:
            # Trainer prints: {"loss": 1.234, "learning_rate": ..., "train_samples_per_second": 7.5, ...}
            if "train_samples_per_second" in line or "samples_per_second" in line:
                try:
                    import json
                    # find JSON blob in line
                    m = re.search(r'\{.*\}', line)
                    if m:
                        d = json.loads(m.group())
                        if "train_samples_per_second" in d:
                            metrics["samples_per_sec"].append(d["train_samples_per_second"])
                        if "train_steps_per_second" in d:
                            metrics["steps_per_sec"].append(d["train_steps_per_second"])
                        if "loss" in d:
                            metrics["loss"].append(d["loss"])
                except Exception:
                    pass
    return {
        "samples_per_sec": mean(metrics["samples_per_sec"]) if metrics["samples_per_sec"] else None,
        "steps_per_sec": mean(metrics["steps_per_sec"]) if metrics["steps_per_sec"] else None,
        "final_loss": metrics["loss"][-1] if metrics["loss"] else None,
    }


def parse_dmon_log(job_id: str) -> dict:
    """Parse nvidia-smi dmon -s pumt log.

    dmon column order with -s pumt:
      #gpu  pwr  gtemp  mtemp  sm   mem   fb   bar1
    """
    log_dir = REPORTS_DIR / f"gpu-util-{job_id}"
    log_file = log_dir / "node-0.log"
    if not log_file.exists():
        return {}

    sm_vals, mem_vals, pwr_vals, fb_vals, temp_vals = [], [], [], [], []

    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                # columns: gpu pwr gtemp mtemp sm mem fb bar1
                # indices:  0    1    2      3    4   5   6   7
                pwr = float(parts[1])
                gtemp = float(parts[2])
                sm = float(parts[4])
                mem = float(parts[5])
                fb = float(parts[6])
                sm_vals.append(sm)
                mem_vals.append(mem)
                pwr_vals.append(pwr)
                fb_vals.append(fb)
                temp_vals.append(gtemp)
            except (ValueError, IndexError):
                continue

    if not sm_vals:
        return {}

    return {
        "mean_sm_pct": mean(sm_vals),
        "mean_mem_bw_pct": mean(mem_vals),
        "mean_power_w": mean(pwr_vals),
        "peak_vram_mb": max(fb_vals),
        "mean_temp_c": mean(temp_vals),
    }


# Config → human-readable description
CONFIG_MAP = {
    "smoke-s1-ddp":  ("Stage 1", "DDP (no DS)", 1),
    "smoke-s1-z1":   ("Stage 1", "ZeRO-1", 1),
    "smoke-s1-z2":   ("Stage 1", "ZeRO-2", 1),
    "smoke-s2-z1":   ("Stage 2", "ZeRO-1", 2),
    "smoke-s2-z2":   ("Stage 2", "ZeRO-2", 2),
    "smoke-s2-z3":   ("Stage 2", "ZeRO-3 (prod)", 2),
}


def detect_job_name(out_path: Path) -> str:
    if out_path is None:
        return "unknown"
    # R-<jobname>.<jobid>.out
    stem = out_path.stem  # e.g. R-smoke-s1-ddp.2069820
    parts = stem.split(".")
    # parts[-1] is job_id, rest after "R-" is job name
    return ".".join(parts[:-1]).removeprefix("R-")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-ids", nargs="+", help="SLURM job IDs to analyse")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect by scanning reports/ for smoke test logs")
    args = parser.parse_args()

    job_ids = args.job_ids or []

    if args.auto or not job_ids:
        # find all smoke log files
        auto_found = list(REPORTS_DIR.glob("R-smoke-*.*.out"))
        for f in auto_found:
            jid = f.stem.split(".")[-1]
            if jid not in job_ids:
                job_ids.append(jid)

    if not job_ids:
        print("No job IDs found. Pass --job-ids or --auto after jobs complete.")
        sys.exit(1)

    rows = []
    for jid in sorted(job_ids):
        out, err = find_log_files(jid)
        job_name = detect_job_name(out)
        stage, strategy, nodes = CONFIG_MAP.get(job_name, ("?", job_name, "?"))

        trainer = parse_trainer_metrics(out)
        dmon = parse_dmon_log(jid)

        rows.append({
            "job_id": jid,
            "name": job_name,
            "stage": stage,
            "strategy": strategy,
            "nodes": nodes,
            **trainer,
            **dmon,
        })

    # Print table
    hdr = (
        f"{'Job ID':>10}  {'Strategy':>16}  {'Stage':>7}  {'N':>2}  "
        f"{'samp/s':>8}  {'step/s':>7}  "
        f"{'SM%':>5}  {'MemBW%':>7}  {'Pwr(W)':>7}  {'VRAM(GB)':>9}  {'Temp(C)':>7}"
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        def fmt(v, fmt_str):
            return f"{v:{fmt_str}}" if v is not None else f"{'N/A':>{fmt_str.split('>')[0] if '>' in fmt_str else '6'}}"

        sps = f"{r['samples_per_sec']:>8.2f}" if r.get("samples_per_sec") else f"{'N/A':>8}"
        stps = f"{r['steps_per_sec']:>7.3f}" if r.get("steps_per_sec") else f"{'N/A':>7}"
        sm = f"{r['mean_sm_pct']:>5.1f}" if r.get("mean_sm_pct") else f"{'N/A':>5}"
        mbw = f"{r['mean_mem_bw_pct']:>7.1f}" if r.get("mean_mem_bw_pct") else f"{'N/A':>7}"
        pwr = f"{r['mean_power_w']:>7.0f}" if r.get("mean_power_w") else f"{'N/A':>7}"
        vram = f"{r['peak_vram_mb']/1024:>9.1f}" if r.get("peak_vram_mb") else f"{'N/A':>9}"
        temp = f"{r['mean_temp_c']:>7.1f}" if r.get("mean_temp_c") else f"{'N/A':>7}"

        print(
            f"{r['job_id']:>10}  {r['strategy']:>16}  {r['stage']:>7}  {r['nodes']:>2}  "
            f"{sps}  {stps}  "
            f"{sm}  {mbw}  {pwr}  {vram}  {temp}"
        )

    print("=" * len(hdr))
    print("\nInterpretation guide:")
    print("  Low SM% (<30%) + Low MemBW%    → GPU starved, likely DataLoader/CPU bottleneck")
    print("  High SM% + High MemBW%          → GPU compute/memory-bandwidth bound (good)")
    print("  Low SM% + High Pwr              → MFU is low, lots of idle cycles")
    print("  ZeRO-1 vs ZeRO-2 vs ZeRO-3     → higher samp/s = less comm overhead")
    print("  DDP vs ZeRO-1 (Stage1)          → measures pure ZeRO comm cost\n")


if __name__ == "__main__":
    main()
