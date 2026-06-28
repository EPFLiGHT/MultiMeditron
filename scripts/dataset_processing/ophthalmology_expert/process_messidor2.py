"""
Prepare Messidor-2 dataset into a JSONL manifest compatible with MultiMeditron.

- Downloads Messidor-2 from Kaggle via kagglehub
- Filters non-gradable images if metadata is available
- Copies images to an output directory
- Writes JSONL with image paths + diagnostic captions
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set

import pandas as pd
from tqdm import tqdm
from transformers import HfArgumentParser

logger = logging.getLogger(__name__)

# -------------------------
# Constants
# -------------------------

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".tif"}

DIAGNOSIS_MAP: Dict[int, str] = {
    0: "No diabetic retinopathy",
    1: "Mild diabetic retinopathy",
    2: "Moderate diabetic retinopathy",
    3: "Severe diabetic retinopathy",
    4: "Proliferative diabetic retinopathy",
}

DME_MAP: Dict[int, str] = {
    0: "without macular edema",
    1: "with macular edema",
    2: "macular edema uncertain",
}

# -------------------------
# Arguments
# -------------------------

@dataclass
class Messidor2PrepArguments:
    kaggle_dataset: str = field(
        default="mariaherrerot/messidor2preprocess",
        metadata={"help": "Kaggle dataset slug for Messidor-2"},
    )
    output_dir: Path = field(
        metadata={"help": "Directory where images will be copied"}
    )
    output_jsonl: Path = field(
        metadata={"help": "Path to output JSONL manifest"}
    )
    filter_non_gradable: bool = field(
        default=True,
        metadata={"help": "Whether to drop non-gradable images if metadata is available"},
    )

# -------------------------
# Utilities
# -------------------------

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS

# -------------------------
# Main logic
# -------------------------

def main():
    parser = HfArgumentParser(Messidor2PrepArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Preparing Messidor-2 dataset")
    logger.info(f"Arguments: {args}")

    # Lazy import to avoid hard dependency if not used
    import kagglehub

    cache_dir = Path(kagglehub.dataset_download(args.kaggle_dataset)).resolve()
    logger.info(f"Kaggle cache directory: {cache_dir}")

    # Load metadata CSV
    csv_files = list(cache_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError("No CSV metadata file found in downloaded dataset")

    meta_path = csv_files[0]
    logger.info(f"Using metadata file: {meta_path}")

    df = pd.read_csv(meta_path)

    if args.filter_non_gradable and "adjudicated_gradable" in df.columns:
        before = len(df)
        df = df[df["adjudicated_gradable"] != 0]
        logger.info(f"Filtered non-gradable images: {before} → {len(df)}")

    # Diagnosis text
    df["diagnosis_text"] = (
        df["diagnosis"]
        .map(DIAGNOSIS_MAP)
        .fillna(df["diagnosis"].astype(str))
    )

    # DME text (optional)
    if "adjudicated_dme" in df.columns:
        df["dme_text"] = (
            df["adjudicated_dme"]
            .map(DME_MAP)
            .fillna("")
        )
    else:
        df["dme_text"] = ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Index all images by filename
    image_map = {p.name: p for p in cache_dir.rglob("*") if is_image(p)}
    logger.info(f"Indexed {len(image_map)} candidate images")

    written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing Messidor-2"):
            img_id = str(row["id_code"]).strip()
            src = image_map.get(img_id)

            if src is None or not src.exists():
                continue

            dst = args.output_dir / src.name
            shutil.copy2(src, dst)

            text = row["diagnosis_text"]
            if row["dme_text"]:
                text = f"{text} {row['dme_text']}"

            record = {
                "text": text.strip(),
                "modalities": [
                    {"type": "image", "value": str(dst)}
                ],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    meta = {
        "kaggle_dataset": args.kaggle_dataset,
        "num_samples": written,
        "filter_non_gradable": args.filter_non_gradable,
    }
    meta_path = args.output_jsonl.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"[DONE] Wrote {written} samples")
    logger.info(f"[META] {meta_path}")
    logger.info(f"[IMAGES] {args.output_dir}")

if __name__ == "__main__":
    main()
