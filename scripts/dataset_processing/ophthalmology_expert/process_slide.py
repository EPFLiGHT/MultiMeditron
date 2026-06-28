"""
Prepare the SLID-E dataset into a JSONL manifest compatible with MultiMeditron.

- Downloads SLID-E metadata CSV from Figshare
- Matches images across train/val/test folders
- Builds a single combined JSONL
- No datasets are committed; paths are provided via CLI/config
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm
from transformers import HfArgumentParser

logger = logging.getLogger(__name__)

# -------------------------
# Constants
# -------------------------

FIGSHARE_CSV_URL = (
    "https://figshare.com/ndownloader/files/46618959"
)

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF",
}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "train set": "train",
    "val": "val",
    "validation": "val",
    "validation set": "val",
    "test": "test",
    "testing": "test",
    "test set": "test",
}

# -------------------------
# Arguments
# -------------------------

@dataclass
class SLIDPrepArguments:
    dataset_root: Path = field(
        metadata={"help": "Root directory containing train/, val/, test/ image folders"}
    )
    output_jsonl: Path = field(
        metadata={"help": "Path to write combined JSONL manifest"}
    )
    csv_path: Optional[Path] = field(
        default=None,
        metadata={"help": "Optional path to SLID-E CSV (downloaded if not provided)"},
    )

# -------------------------
# Utilities
# -------------------------

def normalize_split(val: str) -> Optional[str]:
    if not isinstance(val, str):
        return None
    key = val.strip().lower()
    return SPLIT_ALIASES.get(key)

def safe_str(cell) -> str:
    if pd.isna(cell):
        return ""
    return str(cell).strip()

def download_csv(dst: Path) -> Path:
    logger.info(f"Downloading SLID-E CSV from Figshare → {dst}")
    r = requests.get(FIGSHARE_CSV_URL, timeout=60)
    r.raise_for_status()
    dst.write_bytes(r.content)
    return dst

def require_columns(df: pd.DataFrame) -> Dict[str, str]:
    canon = {
        re.sub(r"[-_ ]+", " ", c.lower().strip()): c
        for c in df.columns
    }
    required = {
        "filename": None,
        "epiphora stage": None,
        "partition group": None,
    }
    for key in required:
        for k, orig in canon.items():
            if k == key:
                required[key] = orig
                break
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing required CSV columns: {missing}")
    return required

def find_image(split_dir: Path, filename: str) -> Optional[Path]:
    cand = split_dir / filename
    if cand.exists():
        return cand

    stem = Path(filename).stem.lower()
    for p in split_dir.rglob("*"):
        if p.is_file() and p.suffix in IMAGE_EXTS and p.stem.lower() == stem:
            return p
    return None

def find_image_any(root: Path, filename: str) -> Optional[Path]:
    for d in ["train", "val", "test"]:
        p = find_image(root / d, filename)
        if p is not None:
            return p
    return None

def build_text(stage: str) -> str:
    return f"SLID-E slit-lamp image with epiphora stage: {stage}."

# -------------------------
# Main logic
# -------------------------

def main():
    parser = HfArgumentParser(SLIDPrepArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    root = args.dataset_root
    for split in ["train", "val", "test"]:
        if not (root / split).exists():
            raise RuntimeError(f"Missing required directory: {root / split}")

    if args.csv_path is None:
        csv_path = root / "SLID_E_information.csv"
        if not csv_path.exists():
            download_csv(csv_path)
    else:
        csv_path = args.csv_path

    logger.info(f"Using CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    colmap = require_columns(df)

    records: List[dict] = []
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing SLID-E"):
        filename = safe_str(row[colmap["filename"]])
        stage = safe_str(row[colmap["epiphora stage"]])
        split_raw = safe_str(row[colmap["partition group"]])

        if not filename:
            continue

        split = normalize_split(split_raw)
        if split:
            img = find_image(root / split, filename)
            if img is None:
                img = find_image_any(root, filename)
        else:
            img = find_image_any(root, filename)

        if img is None:
            missing += 1
            continue

        rel_path = img.relative_to(root)
        records.append(
            {
                "text": build_text(stage),
                "modalities": [{"type": "image", "value": str(rel_path)}],
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "dataset": "SLID-E",
        "num_samples": len(records),
        "missing_images": missing,
        "csv": str(csv_path),
    }
    meta_path = args.output_jsonl.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"[DONE] Wrote {len(records)} samples → {args.output_jsonl}")
    logger.info(f"[META] {meta_path}")

if __name__ == "__main__":
    main()
