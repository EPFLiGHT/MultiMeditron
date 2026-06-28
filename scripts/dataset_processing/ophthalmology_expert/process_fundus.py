"""
Prepare the UWF Fundus Images of Intraocular Tumors dataset into a JSONL manifest.

- Assumes dataset is downloaded from Figshare
- Iterates class-based folders
- Writes a single JSONL with correct relative paths
- No datasets are committed; paths are provided via CLI/config
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

from tqdm import tqdm
from transformers import HfArgumentParser

logger = logging.getLogger(__name__)

# -------------------------
# Constants
# -------------------------

FIGSHARE_URL = (
    "https://springernature.figshare.com/articles/dataset/"
    "An_ultra-wide-field_fundus_image_dataset_for_intelligent_diagnosis_of_intraocular_tumors/27986258"
)

CLASS_FOLDERS: List[str] = [
    "Normal",
    "Choroidal Hemangioma (CH)",
    "Retinal Capillary Hemangioma (RCH)",
    "Choroidal Osteoma (CO)",
    "Retinoblastoma (RB)",
    "Uveal Melanoma (UM)",
]

IMAGE_EXTS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF",
}

# -------------------------
# Arguments
# -------------------------

@dataclass
class UWFTumorPrepArguments:
    dataset_root: Path = field(
        metadata={
            "help": (
                "Root directory containing 'UWF Fundus Images of Intraocular Tumors/' "
                "with class subfolders"
            )
        }
    )
    output_jsonl: Path = field(
        metadata={"help": "Path to output JSONL manifest"}
    )

# -------------------------
# Utilities
# -------------------------

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix in IMAGE_EXTS

def build_text(class_name: str) -> str:
    return f"Ultra-wide-field fundus image with intraocular tumor diagnosis: {class_name}."

# -------------------------
# Main logic
# -------------------------

def main():
    parser = HfArgumentParser(UWFTumorPrepArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    root = args.dataset_root
    if not root.exists():
        raise RuntimeError(f"Dataset root does not exist: {root}")

    logger.info("Preparing UWF Fundus Tumor dataset")
    logger.info(f"Dataset root: {root}")
    logger.info(f"Figshare source: {FIGSHARE_URL}")

    items = []
    for cls in CLASS_FOLDERS:
        cls_dir = root / cls
        if not cls_dir.exists():
            raise RuntimeError(f"Expected class folder missing: {cls_dir}")

        for img in sorted(cls_dir.rglob("*")):
            if is_image(img):
                items.append((img, cls))

    if not items:
        raise RuntimeError(f"No images found under {root}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for img_path, cls in tqdm(items, total=len(items), desc="Images"):
            rel_path = img_path.relative_to(root)
            record = {
                "text": build_text(cls),
                "modalities": [{"type": "image", "value": str(rel_path)}],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    meta = {
        "dataset": "UWF Fundus Images of Intraocular Tumors",
        "source": FIGSHARE_URL,
        "num_samples": written,
        "classes": CLASS_FOLDERS,
        "root": str(root),
    }
    meta_path = args.output_jsonl.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"[DONE] Wrote {written} samples → {args.output_jsonl}")
    logger.info(f"[META] {meta_path}")

if __name__ == "__main__":
    main()
