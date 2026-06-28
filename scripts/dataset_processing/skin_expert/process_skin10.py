"""
Prepare the Skin Diseases Image Dataset (Skin-10) into a canonical JSONL manifest
compatible with MultiMeditron (Skin Expert).

Design principles:
- single deterministic pipeline
- official numeric-folder → diagnosis mapping
- no filename-based label inference
- no post-hoc cleaning
- reproducible from scratch
"""

import json
import logging
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from transformers import HfArgumentParser

# -------------------------------------------------
# Official Skin-10 class mapping (dataset-provided)
# -------------------------------------------------

NUM_TO_DIAG: Dict[str, str] = {
    "1": "Eczema",
    "2": "Melanoma",
    "3": "Atopic dermatitis",
    "4": "Basal cell carcinoma",
    "5": "Melanocytic nevi",
    "6": "Benign keratosis-like lesions",
    "7": "Psoriasis / lichen planus",
    "8": "Seborrheic keratoses and benign tumors",
    "9": "Fungal infections (tinea, candidiasis)",
    "10": "Viral infections (warts, molluscum)",
}

PARAPHRASE_TEMPLATES: List[str] = [
    "The diagnosis for this skin image is {v}.",
    "This photo shows a case of {v}.",
    "Clinical appearance suggests {v}.",
    "This image depicts {v}.",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Arguments
# -------------------------------------------------

@dataclass
class Skin10PrepArguments:
    # REQUIRED (no defaults) — must come first
    output_images_root: Path = field(
        metadata={"help": "Root directory where images are organized by class"}
    )
    output_jsonl: Path = field(
        metadata={"help": "Path to output JSONL manifest"}
    )

    # OPTIONAL (defaults) — must come after
    kaggle_dataset: str = field(
        default="ismailpromus/skin-diseases-image-dataset",
        metadata={"help": "Kaggle dataset slug for Skin-10"},
    )
    paraphrase: bool = field(
        default=True,
        metadata={"help": "Whether to paraphrase diagnosis text"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for deterministic paraphrasing"},
    )

# -------------------------------------------------
# Utilities
# -------------------------------------------------

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def slugify(s: str) -> str:
    s = re.sub(r"[()]", "", s)
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "unknown"

def extract_archives(root: Path) -> None:
    for arc in root.rglob("*"):
        if not arc.is_file():
            continue
        if arc.name.lower().endswith((".zip", ".tar", ".tar.gz", ".tar.bz2", ".tar.xz")):
            target = arc.parent / arc.stem
            if target.exists():
                continue
            try:
                target.mkdir(parents=True, exist_ok=True)
                shutil.unpack_archive(str(arc), str(target))
                logger.info(f"Extracted {arc}")
            except Exception as e:
                logger.warning(f"Failed to extract {arc}: {e}")

# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = HfArgumentParser(Skin10PrepArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    rng = random.Random(args.seed)

    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError("Please install kagglehub: pip install kagglehub") from e

    logger.info("Downloading Skin-10 from KaggleHub")
    cache_dir = Path(kagglehub.dataset_download(args.kaggle_dataset)).resolve()

    extract_archives(cache_dir)

    args.output_images_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for class_dir in sorted(cache_dir.rglob("*")):
            if not class_dir.is_dir():
                continue

            # Expected directory names like: "1. Eczema 1677"
            m = re.match(r"^\s*(\d+)[\.\-]?\s*(.+)$", class_dir.name)
            if not m:
                continue

            class_idx = m.group(1)
            diagnosis = NUM_TO_DIAG.get(class_idx)
            if diagnosis is None:
                continue

            dst_dir = args.output_images_root / slugify(diagnosis)
            dst_dir.mkdir(parents=True, exist_ok=True)

            images = [p for p in class_dir.rglob("*") if is_image(p)]
            for img in images:
                dst = dst_dir / img.name
                shutil.copy2(img, dst)

                text = diagnosis
                if args.paraphrase:
                    text = rng.choice(PARAPHRASE_TEMPLATES).format(v=diagnosis)

                record = {
                    "text": text,
                    "modalities": [{"type": "image", "value": str(dst)}],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    meta = {
        "dataset": "Skin Diseases Image Dataset (Skin-10)",
        "num_samples": written,
        "paraphrase": args.paraphrase,
        "seed": args.seed,
    }
    args.output_jsonl.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    logger.info(f"[DONE] Wrote {written} samples → {args.output_jsonl}")

if __name__ == "__main__":
    main()
