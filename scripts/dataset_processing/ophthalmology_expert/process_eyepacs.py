"""
Prepare EYEPACS dataset into a JSONL manifest compatible with MultiMeditron.

- Downloads EYEPACS from HuggingFace Datasets
- Saves images grouped by diagnosis
- Writes JSONL with image paths + paraphrased captions
- No datasets are committed; paths are provided via CLI/config
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed

logger = logging.getLogger(__name__)

# -------------------------
# Label mappings
# -------------------------

NUM_TO_HUMAN: Dict[int, str] = {
    0: "No diabetic retinopathy",
    1: "Mild diabetic retinopathy",
    2: "Moderate diabetic retinopathy",
    3: "Severe diabetic retinopathy",
    4: "Proliferative diabetic retinopathy",
}

CODE_TO_HUMAN: Dict[str, str] = {
    "no_dr": "No diabetic retinopathy",
    "no_diabetic_retinopathy": "No diabetic retinopathy",
    "mild_retinopathy": "Mild diabetic retinopathy",
    "moderate_retinopathy": "Moderate diabetic retinopathy",
    "severe": "Severe diabetic retinopathy",
    "proliferative_dr": "Proliferative diabetic retinopathy",
    "proliferative_diabetic_retinopathy": "Proliferative diabetic retinopathy",
}

TEMPLATES: List[str] = [
    "Fundus image diagnosis: {v}.",
    "This retinal image shows {v}.",
    "Clinical assessment suggests {v}.",
    "The diagnosis for this fundus photograph is {v}.",
    "Findings are consistent with {v}.",
    "Ophthalmic features indicate {v}.",
    "This picture illustrates {v}.",
]

# -------------------------
# Arguments
# -------------------------

@dataclass
class EyepacsPrepArguments:
    hf_dataset: str = field(
        default="bumbledeep/eyepacs",
        metadata={"help": "HuggingFace dataset name for EYEPACS"},
    )
    split: str = field(
        default="train",
        metadata={"help": "Dataset split to process (train/val/test if available)"},
    )
    output_dir: Path = field(
        metadata={"help": "Directory where images and manifest will be written"}
    )
    output_jsonl: Path = field(
        metadata={"help": "Path to output JSONL manifest"}
    )
    image_format: str = field(
        default="JPEG",
        metadata={"help": "Image save format: JPEG or PNG"},
    )
    jpeg_quality: int = field(
        default=95,
        metadata={"help": "JPEG quality if JPEG is used"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for caption paraphrasing"},
    )

# -------------------------
# Utilities
# -------------------------

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "unknown"

def derive_human_label(example: dict) -> str:
    lbl_code = example.get("label_code")
    lbl = example.get("label")

    if isinstance(lbl_code, str):
        key = lbl_code.lower().strip()
        if key in CODE_TO_HUMAN:
            return CODE_TO_HUMAN[key]
        return key.replace("_", " ").capitalize()

    if isinstance(lbl, (int, float)):
        return NUM_TO_HUMAN.get(int(lbl), f"Unknown grade ({lbl})")

    return "Unknown diagnosis"

def save_image(img, out_path: Path, fmt: str, jpeg_quality: int) -> None:
    if fmt.upper() == "JPEG":
        img = img.convert("RGB")
        img.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
        img.save(out_path, format="PNG")

# -------------------------
# Main logic
# -------------------------

def main():
    parser = HfArgumentParser(EyepacsPrepArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Preparing EYEPACS dataset")
    logger.info(f"Arguments: {args}")

    set_seed(args.seed)
    rng = random.Random(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset {args.hf_dataset} [{args.split}]")
    dataset = load_dataset(args.hf_dataset, split=args.split)

    written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for idx, ex in tqdm(enumerate(dataset), total=len(dataset)):
            diagnosis = derive_human_label(ex)
            class_dir = args.output_dir / slugify(diagnosis)
            class_dir.mkdir(parents=True, exist_ok=True)

            image = ex["image"]
            filename = f"sample_{idx}.jpg" if args.image_format == "JPEG" else f"sample_{idx}.png"
            image_path = class_dir / filename

            save_image(image, image_path, args.image_format, args.jpeg_quality)

            caption = rng.choice(TEMPLATES).format(v=diagnosis)

            record = {
                "text": caption,
                "modalities": [
                    {"type": "image", "value": str(image_path)}
                ],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    meta = {
        "hf_dataset": args.hf_dataset,
        "split": args.split,
        "seed": args.seed,
        "num_samples": written,
    }
    meta_path = args.output_jsonl.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"[DONE] Wrote {written} samples")
    logger.info(f"[META] {meta_path}")

if __name__ == "__main__":
    main()
