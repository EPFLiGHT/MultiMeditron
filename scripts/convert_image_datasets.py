"""
Convert eye_dataset and skin_dataset from their raw format to the format expected
by the MultiMeditron training pipeline.

Raw format (per row):
  - text:        plain string description
  - modalities:  [{"type": "image", "value": "<filepath_string>"}]
  - image:       HF Image feature (decode=False) → {"bytes": b"...", "path": "..."}

Target format (same as llava_pretrain_cleaned):
  - conversations: [
        {"role": "user",      "content": "Describe this medical image. <|reserved_special_token_0|>"},
        {"role": "assistant", "content": "<description text>"}
    ]
  - modalities:  [{"type": "image", "value": {"bytes": b"...", "path": None}}]

Usage (run inside the container):
    python scripts/convert_image_datasets.py
"""

from datasets import Dataset, DatasetDict, load_from_disk, Features, Sequence, Value
import os

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"
USER_PROMPT = f"Describe this medical image. {ATTACHMENT_TOKEN}"

DATASETS = {
    "eye_dataset":  "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/eye_dataset",
    "skin_dataset": "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/skin_dataset",
}

OUT_BASE = "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow"


def convert_row(row):
    # The image column (decode=False) already contains {"bytes": ..., "path": ...}
    image_data = row["image"]
    image_value = {"bytes": image_data["bytes"], "path": None}

    # Rebuild modalities with bytes instead of filepath string
    new_modalities = []
    for mod in row["modalities"]:
        new_modalities.append({"type": mod["type"], "value": image_value})

    # Build a simple two-turn conversation
    conversations = [
        {"role": "user",      "content": USER_PROMPT},
        {"role": "assistant", "content": row["text"]},
    ]

    return {"conversations": conversations, "modalities": new_modalities}


def convert_dataset(src_path: str, dst_path: str, name: str):
    print(f"\n{'='*60}")
    print(f"Converting: {name}")
    print(f"  source : {src_path}")
    print(f"  output : {dst_path}")

    raw = load_from_disk(src_path)
    if isinstance(raw, DatasetDict):
        print(f"  splits : {list(raw.keys())} → using 'train'")
        ds = raw["train"]
    else:
        ds = raw
    print(f"  rows   : {len(ds):,}")

    converted = ds.map(
        convert_row,
        remove_columns=ds.column_names,
        desc=f"Converting {name}",
        num_proc=16,
    )

    print(f"  saving to {dst_path} …")
    converted.save_to_disk(dst_path)
    print(f"  done. rows={len(converted):,}, columns={converted.column_names}")


if __name__ == "__main__":
    for name, src in DATASETS.items():
        dst = os.path.join(OUT_BASE, f"{name}_converted")
        if os.path.exists(dst):
            print(f"[SKIP] Output already exists: {dst}")
            continue
        convert_dataset(src, dst, name)
    print("\nAll conversions complete.")
