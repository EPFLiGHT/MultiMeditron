#!/usr/bin/env python3
import os
import sys
import json
import shutil
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -----------------------------
# CLI arguments
# -----------------------------
@dataclass
class ISIC4PrepArguments:
    # required
    output_images_root: Path = field(
        metadata={"help": "Root directory where images will be stored by class"}
    )
    output_jsonl: Path = field(
        metadata={"help": "Path to output JSONL manifest"}
    )

    # optional
    kaggle_dataset: str = field(
        default="abhii1929/isic-skin-disease-image-dataset-4-classes",
        metadata={"help": "Kaggle dataset slug"},
    )
    paraphrase: bool = field(
        default=False,
        metadata={"help": "Whether to paraphrase diagnosis text"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed (only used if paraphrase=True)"},
    )


# -----------------------------
# Helpers
# -----------------------------
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[()]", "", name)
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-") or "unknown"


def collect_class_dirs(root: Path) -> List[Path]:
    """
    Collect leaf directories that contain images.
    """
    class_dirs = []
    for d in root.rglob("*"):
        if d.is_dir() and any(is_image(p) for p in d.iterdir()):
            class_dirs.append(d)
    return class_dirs


def load_kaggle_dataset(slug: str) -> Path:
    try:
        import kagglehub
    except ImportError:
        print("Please install kagglehub: pip install kagglehub", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Downloading ISIC-4 dataset: {slug}")
    path = Path(kagglehub.dataset_download(slug)).resolve()
    print(f"[INFO] Dataset cached at: {path}")
    return path


# -----------------------------
# Main
# -----------------------------
def main():
    from transformers import HfArgumentParser

    parser = HfArgumentParser(ISIC4PrepArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.paraphrase:
        import random
        random.seed(args.seed)
        TEMPLATES = [
            "The diagnosis for this skin lesion is {v}.",
            "This dermoscopic image shows {v}.",
            "Clinical assessment suggests {v}.",
            "This image corresponds to {v}.",
        ]

    # 1. Download dataset
    cache_root = load_kaggle_dataset(args.kaggle_dataset)

    # 2. Find class folders
    class_dirs = collect_class_dirs(cache_root)
    if not class_dirs:
        raise RuntimeError("No class directories with images found.")

    print(f"[INFO] Found {len(class_dirs)} class folders")

    # 3. Prepare output dirs
    args.output_images_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # 4. Process images
    written = 0
    records = []

    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        label_raw = class_dir.name
        label = label_raw.replace("_", " ").title()
        label_slug = slugify(label_raw)

        dst_class_dir = args.output_images_root / label_slug
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in class_dir.iterdir() if is_image(p)]

        for img in images:
            dst = dst_class_dir / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

            text = label
            if args.paraphrase:
                text = random.choice(TEMPLATES).format(v=label)

            rel_path = os.path.relpath(dst, start=args.output_jsonl.parent)
            records.append(
                {
                    "text": text,
                    "modalities": [{"type": "image", "value": rel_path}],
                }
            )
            written += 1

    # 5. Write JSONL
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] Wrote {written} samples")
    print(f"[IMAGES] {args.output_images_root}")
    print(f"[JSONL]  {args.output_jsonl}")


if __name__ == "__main__":
    main()
