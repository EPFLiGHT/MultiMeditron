"""Prepare the Brain Tumor MRI Dataset (Masoud Nickparvar) into canonical JSONL manifests.

Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
4 classes: glioma, meningioma, no_tumor, pituitary
Split: pre-defined by the dataset (Training/ 5712 images, Testing/ 1311 images).

Usage:
    python src/multimeditron/experts/dataset_processing/mri_expert/process_brain_tumor.py \
        --output_dir /lightscratch/users/cljordan/datasets/brain_tumor_mri
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm
from transformers import HfArgumentParser

KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"

FOLDER_TO_LABEL = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "notumor": "no_tumor",
    "pituitary": "pituitary",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

logger = logging.getLogger(__name__)


@dataclass
class BrainTumorPrepArguments:
    output_dir = field(
        metadata={"help": "Root directory for output images and JSONL manifests"}
    )
    kaggle_dataset = field(
        default=KAGGLE_DATASET,
        metadata={"help": "Kaggle dataset slug"},
    )


def _is_image(p):
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _process_split(
    source_dir,
    output_images_dir,
    output_jsonl,
    split_name,
):
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for folder_name, label in FOLDER_TO_LABEL.items():
            class_src = source_dir / folder_name
            if not class_src.is_dir():
                logger.warning("[%s] class folder not found: %s", split_name, class_src)
                continue

            class_dst = output_images_dir / label
            class_dst.mkdir(parents=True, exist_ok=True)

            images = [p for p in class_src.iterdir() if _is_image(p)]
            for img in tqdm(images, desc=f"{split_name}/{label}", leave=False):
                dst = class_dst / img.name
                shutil.copy2(img, dst)
                record = {
                    "text": label,
                    "label": label,
                    "modalities": [{"type": "image", "value": str(dst)}],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    logger.info("[%s] wrote %d records → %s", split_name, written, output_jsonl)
    return written


def main():
    parser = HfArgumentParser(BrainTumorPrepArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError("Please install kagglehub: pip install kagglehub") from e

    logger.info("Downloading %s from KaggleHub...", args.kaggle_dataset)
    cache_dir = Path(kagglehub.dataset_download(args.kaggle_dataset)).resolve()
    logger.info("Downloaded to %s", cache_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_count = _process_split(
        source_dir=cache_dir / "Training",
        output_images_dir=output_dir / "images" / "train",
        output_jsonl=output_dir / "brain_tumor_train.jsonl",
        split_name="train",
    )
    test_count = _process_split(
        source_dir=cache_dir / "Testing",
        output_images_dir=output_dir / "images" / "test",
        output_jsonl=output_dir / "brain_tumor_test.jsonl",
        split_name="test",
    )

    meta = {
        "dataset": "Brain Tumor MRI Dataset (Masoud Nickparvar)",
        "source": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
        "classes": list(FOLDER_TO_LABEL.values()),
        "train_count": train_count,
        "test_count": test_count,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Done. train=%d, test=%d", train_count, test_count)


if __name__ == "__main__":
    main()
