#!/usr/bin/env python3
"""Build clean CT-only manifests from CT2D-glob.

Two-pass approach to avoid memory accumulation:
  Pass 1 — scan all shards, write matching CT records to a temp JSONL (only
            text + metadata, no image bytes).
  Pass 2 — shuffle the temp file, assign splits, write final manifests.

Output: benchmark_splits/multimediset/ct/
  train_model.jsonl, mlp_train.jsonl, benchmark_eval.jsonl, holdout_test.jsonl
"""

import json
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CT2D_GLOB_ROOT = Path("/lightscratch/datasets/MultiMediset/general_purpose/CT2D-glob")
OUTPUT_DIR = Path("benchmark_splits/multimediset/ct")
SEED = 42

CT_PATTERN = re.compile(
    r"\bct\b|ct scan|computed tomography|ct image|ct of|axial ct|coronal ct|ct slice|ct chest|ct abdomen",
    re.IGNORECASE,
)

LABELS = ["atherosoma", "Covid", "healthy", "glioblastoma", "tumor"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}

# train source: 70/10/10/10
TRAIN_PLAN = [
    ("train_model", 0.7),
    ("mlp_train", 0.1),
    ("benchmark_eval", 0.1),
    ("holdout_test", 0.1),
]
# test source: all goes to holdout
TEST_PLAN = [("holdout_test", 1.0)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_label(text):
    if "tumor" in text:
        return "tumor"
    if "atherosoma" in text:
        return "atherosoma"
    if "glioblastoma" in text:
        return "glioblastoma"
    if "Covid" in text:
        return "Covid"
    return "healthy"


def scan_to_tmp(split_dir, source_split, tmp_path):
    """Pass 1: scan shards, write CT records to tmp_path. Returns count."""
    shards = sorted(split_dir.glob("data-*.arrow"))
    global_idx = 0
    ct_count = 0
    with tmp_path.open("w", encoding="utf-8") as out:
        for shard in shards:
            with pa.memory_map(str(shard), "r") as src:
                table = pa.ipc.open_stream(src).read_all()
            texts = table["text"]
            table["modalities"] if "modalities" in table.schema.names else None
            for i in range(len(table)):
                text = texts[i].as_py() or ""
                if CT_PATTERN.search(text):
                    label = find_label(text)
                    record = {
                        "benchmark": "ct",
                        "dataset": "CT2D-glob",
                        "source_root": str(CT2D_GLOB_ROOT),
                        "source_split": source_split,
                        "source_index": global_idx,
                        "label": label,
                        "label_id": LABEL_TO_ID[label],
                        "group_id": None,
                    }
                    out.write(json.dumps(record, sort_keys=True) + "\n")
                    ct_count += 1
                global_idx += 1
            print(
                f"  {shard.name}: {global_idx} scanned, {ct_count} CT so far",
                flush=True,
            )
    return ct_count


def assign_from_tmp(tmp_path, plan, rng):
    """Pass 2: read lines from tmp, shuffle, assign to splits. Returns lines per split."""
    lines = tmp_path.read_text(encoding="utf-8").splitlines()
    rng.shuffle(lines)
    total = len(lines)
    raw = [(split, total * ratio) for split, ratio in plan]
    counts = {split: int(r) for split, r in raw}
    remainder = total - sum(counts.values())
    for split, _ in sorted(raw, key=lambda x: x[1] - int(x[1]), reverse=True)[
        :remainder
    ]:
        counts[split] += 1
    assigned = {}
    offset = 0
    for split, _ in plan:
        assigned[split] = lines[offset : offset + counts[split]]
        offset += counts[split]
    return assigned


def append_jsonl(path, lines):
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear existing output files
    for split_name in ("train_model", "mlp_train", "benchmark_eval", "holdout_test"):
        (OUTPUT_DIR / f"{split_name}.jsonl").unlink(missing_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        for source_split, plan in [("train", TRAIN_PLAN), ("test", TEST_PLAN)]:
            split_dir = CT2D_GLOB_ROOT / source_split
            if not split_dir.exists():
                print(f"Skipping {source_split} (not found)")
                continue
            print(f"\n=== Pass 1: scanning {source_split} ===")
            tmp_path = Path(tmpdir) / f"{source_split}.jsonl"
            ct_count = scan_to_tmp(split_dir, source_split, tmp_path)
            print(f"  -> {ct_count} CT examples found")

            print(f"=== Pass 2: assigning splits for {source_split} ===")
            rng = random.Random(f"{SEED}:{source_split}")
            assigned = assign_from_tmp(tmp_path, plan, rng)
            for split_name, lines in assigned.items():
                append_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", lines)
                print(f"  {split_name}: {len(lines)} records")

    print("\nFinal counts:")
    for split_name in ("train_model", "mlp_train", "benchmark_eval", "holdout_test"):
        path = OUTPUT_DIR / f"{split_name}.jsonl"
        lines = path.read_text().splitlines()
        label_counts = Counter(json.loads(l)["label"] for l in lines if l)
        print(
            f"  {split_name}.jsonl: {len(lines)} — {dict(sorted(label_counts.items()))}"
        )

    print(f"\nDone. Manifests written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
