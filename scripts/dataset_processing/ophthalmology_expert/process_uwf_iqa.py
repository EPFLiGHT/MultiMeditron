"""
Prepare the Open UWF Fundus IQA dataset into a clean JSONL manifest.

This script:
- Assumes dataset downloaded from Figshare
- Converts all images to safe RGB PNG (fixes TIFF/JPEG issues)
- Normalizes diagnosis text
- Writes a single JSONL manifest
- Never requires post-hoc path or text fixing
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from PIL import Image
from tqdm import tqdm
from transformers import HfArgumentParser

logger = logging.getLogger(__name__)

# -------------------------
# Constants
# -------------------------

FIGSHARE_URL = (
    "https://springernature.figshare.com/articles/dataset/"
    "Open_ultrawidefield_fundus_image_dataset_with_disease_diagnosis_and_clinical_image_quality_assessment/26936446"
)

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

DX_MAP: Dict[str, str] = {
    "AMD": "Age-related macular degeneration",
    "DR": "Diabetic retinopathy",
    "Healthy": "Healthy control",
    "PM": "Pathologic myopia",
    "RD": "Retinal detachment",
    "RVO": "Retinal vein occlusion",
    "Uveitis": "Uveitis",
}

# -------------------------
# Arguments
# -------------------------

@dataclass
class UWFIQAPrepArguments:
    dataset_root: Path = field(
        metadata={
            "help": (
                "Root directory containing 'Original UWF Image/' with class subfolders"
            )
        }
    )
    output_dir: Path = field(
        metadata={"help": "Directory where fixed images will be written"}
    )
    output_jsonl: Path = field(
        metadata={"help": "Path to output JSONL manifest"}
    )

# -------------------------
# Utilities
# -------------------------

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def normalize_image(src: Path, dst: Path) -> None:
    """
    Convert image to safe RGB PNG.
    Fixes TIFF / CMYK / broken metadata issues.
    """
    try:
        with Image.open(src) as im:
            im.info.clear()
            im = im.convert("RGB")
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="PNG", optimize=True)
    except Exception as e:
        raise RuntimeError(f"Failed to convert {src}: {e}")

def build_text(cls: str) -> str:
    long_dx = DX_MAP.get(cls, cls)
    return f"Ultra-wide-field fundus image with diagnosis: {long_dx}."

# -------------------------
# Main logic
# -------------------------

def main():
    parser = HfArgumentParser(UWFIQAPrepArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    root = args.dataset_root
    if not root.exists():
        raise RuntimeError(f"Dataset root not found: {root}")

    logger.info("Preparing Open UWF Fundus IQA dataset")
    logger.info(f"Dataset root: {root}")
    logger.info(f"Figshare source: {FIGSHARE_URL}")

    items = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        for img in cls_dir.rglob("*"):
            if is_image(img):
                items.append((img, cls))

    if not items:
        raise RuntimeError("No images found in dataset")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for src, cls in tqdm(items, total=len(items), desc="Processing images"):
            rel = src.relative_to(root)
            dst = args.output_dir / rel.with_suffix(".png")

            normalize_image(src, dst)

            record = {
                "text": build_text(cls),
                "modalities": [{"type": "image", "value": str(dst.relative_to(args.output_dir))}],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    meta = {
        "dataset": "Open UWF Fundus IQA",
        "source": FIGSHARE_URL,
        "num_samples": written,
        "classes": sorted({cls for _, cls in items}),
    }
    meta_path = args.output_jsonl.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"[DONE] Wrote {written} samples → {args.output_jsonl}")
    logger.info(f"[IMAGES] Fixed images → {args.output_dir}")
    logger.info(f"[META] {meta_path}")

if __name__ == "__main__":
    main()
