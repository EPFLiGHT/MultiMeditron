"""
submit_batches.py

Submit JSONL batch request files to the OpenAI Batch API.
This script is safe to re-run and will not re-submit completed batches.
"""

from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import os
import sys
import config

ESTIMATE_ONLY = os.getenv("ESTIMATE_ONLY", "false").lower() == "true"

BATCHES_DIR = Path(config.BATCHES_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

BATCHES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------

if not config.OPENAI_API_KEY:
    raise SystemExit(
        "OPENAI_API_KEY is not set. Please export it before running."
    )

client = OpenAI(api_key=config.OPENAI_API_KEY)

# ---------------------------------------------------------------------
# Load batch parts
# ---------------------------------------------------------------------

parts = sorted(BATCHES_DIR.glob("*.jsonl"))

if not parts:
    raise SystemExit(f"No batch files found in {BATCHES_DIR}")

if ESTIMATE_ONLY:
    parts = parts[:1]
    print("[INFO] ESTIMATE_ONLY enabled: submitting first batch only")

# ---------------------------------------------------------------------
# Submit batches
# ---------------------------------------------------------------------

for part in tqdm(parts, desc="Submitting batches"):
    batch_id_file = OUTPUT_DIR / f"batch_id_{part.name}.txt"

    # Skip if already submitted
    if batch_id_file.exists():
        print(f"[SKIP] {part.name} already submitted")
        continue

    # Upload batch input file
    with part.open("rb") as fh:
        uploaded = client.files.create(
            file=fh,
            purpose="batch",
        )

    # Create batch job
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Dataset augmentation - {part.name}",
        },
    )

    # Persist batch ID (critical for reproducibility)
    batch_id_file.write_text(batch.id)

    print(f"[SUBMITTED] {part.name} → batch_id={batch.id}")

print("All batches submitted.")
