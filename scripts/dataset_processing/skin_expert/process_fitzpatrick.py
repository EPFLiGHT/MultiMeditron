"""
Prepare the Fitzpatrick Black Skin Disease dataset into a JSONL manifest
compatible with MultiMeditron (Skin Expert).

- Downloads dataset via KaggleHub
- Extracts nested archives if present
- Infers disease labels from directory structure
- Optionally de-duplicates against training datasets via MD5 hashes
- Copies images into a flat output directory
- Writes JSONL manifest + metadata
"""

import json
import logging
import re
import shutil
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

from tqdm import tqdm
from transformers import HfArgumentParser

logger = logging.getLogger(__name__)

# -------------------------
# Constants
# -------------------------

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

GENERIC_DIRS = {
    "release_v0", "images", "img", "image", "data", "dataset",
    "train", "val", "valid", "validation", "test", "__macosx"
}

# -------------------------
# Arguments
# -------------------------

@dataclass
class FitzpatrickPrepArguments:
    kaggle_dataset: str = field(
        default="oyebamijimicheal/fitzpatrick-black-skin-disease-dataset",
        metadata={"help": "Kaggle dataset slug for Fitzpatrick Black Skin Disease"},
    )
    output_dir: Path = field(
        metadata={"help": "Directory where images will be copied"}
    )
    output_jsonl: Path = field(
        metadata={"help": "Path to output JSONL manifest"}
    )
    dedup_train_dirs: List[Path] = field(
        default_factory=list,
        metadata={
            "help": (
                "Optional list of directories containing training images; "
                "images overlapping by MD5 hash will be skipped"
            )
        },
    )

# -------------------------
# Utilities
# -------------------------

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[()]", "", s)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unknown"

def class_from_ancestors(p: Path) -> str:
    """
    Walk up from the image file to find the first parent directory
    that isn't a generic container; use that as the disease label.
    """
    cur = p.parent
    while cur and cur != cur.parent:
        name = cur.name.strip()
        if name and name.lower() not in GENERIC_DIRS:
            return name
        cur = cur.parent
    return "unknown"

def extract_archives(root: Path) -> None:
    for arc in root.rglob("*"):
        if not arc.is_file():
            continue
        name = arc.name.lower()
        target = None
        if name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
            target = arc.parent / arc.name.rsplit(".", 2)[0]
        elif name.endswith((".zip", ".tar", ".gz", ".bz2", ".xz")):
            target = arc.with_suffix("")
        if target is None or target.exists():
            continue
        try:
            target.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(arc), str(target))
            logger.info(f"Extracted {arc} → {target}")
        except Exception as e:
            logger.warning(f"Failed to extract {arc}: {e}")

def md5sum(p: Path, chunk: int = 65536) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def build_train_hashes(dirs: List[Path]) -> Set[str]:
    hashes = set()
    for d in dirs:
        if not d.exists():
            continue
        for img in d.rglob("*"):
            if is_image(img):
                try:
                    hashes.add(md5sum(img))
                except Exception:
                    pass
    if hashes:
        logger.info(f"Loaded {len(hashes)} training image hashes for de-duplication")
    return hashes

# -------------------------
# Main logic
# -------------------------

def main():
    parser = HfArgumentParser(FitzpatrickPrepArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Preparing Fitzpatrick Black Skin Disease dataset (Skin Expert)")
    logger.info(f"Kaggle dataset: {args.kaggle_dataset}")

    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError("Please install kagglehub: pip install kagglehub") from e

    cache_dir = Path(kagglehub.dataset_download(args.kaggle_dataset)).resolve()
    logger.info(f"Kaggle cache directory: {cache_dir}")

    extract_archives(cache_dir)

    src_images = [p for p in cache_dir.rglob("*") if is_image(p)]
    if not src_images:
        raise RuntimeError("No images found after extraction")

    logger.info(f"Found {len(src_images)} candidate images")

    train_hashes = build_train_hashes(args.dedup_train_dirs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def uniquify(name: str) -> str:
        base = slugify(Path(name).stem)
        suf = Path(name).suffix.lower()
        candidate = f"{base}{suf}"
        i = 1
        while (args.output_dir / candidate).exists():
            candidate = f"{base}__{i}{suf}"
            i += 1
        return candidate

    written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for src in tqdm(src_images, desc="Preparing Fitzpatrick Black Skin"):
            label = class_from_ancestors(src)
            if label == "unknown":
                continue

            if train_hashes:
                try:
                    if md5sum(src) in train_hashes:
                        continue
                except Exception:
                    pass

            dst_name = uniquify(src.name)
            dst = args.output_dir / dst_name

            try:
                shutil.copy2(src, dst)
            except Exception as e:
                logger.warning(f"Copy failed {src} → {dst}: {e}")
                continue

            record = {
                "text": label,
                "modalities": [{"type": "image", "value": str(dst)}],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    meta = {
        "dataset": "Fitzpatrick Black Skin Disease",
        "kaggle_dataset": args.kaggle_dataset,
        "num_samples": written,
        "deduplicated_against": [str(p) for p in args.dedup_train_dirs],
    }
    meta_path = args.output_jsonl.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"[DONE] Wrote {written} samples → {args.output_jsonl}")
    logger.info(f"[IMAGES] {args.output_dir}")
    logger.info(f"[META] {meta_path}")

if __name__ == "__main__":
    main()
