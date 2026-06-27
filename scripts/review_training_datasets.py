"""
Generate a visual PDF review of the datasets used for 7-expert MultiMeditron training.

For each training dataset, samples --num_samples rows and renders one page per sample
showing the image alongside the conversation text.

Usage (inside multimeditron container):
    export STORAGE_ROOT=/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow
    pip install reportlab -q

    python scripts/review_training_datasets.py \
        --output /path/to/training_data_review.pdf \
        --num_samples 20
"""

import argparse
import io
import os
import random
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from datasets import load_from_disk
from PIL import Image

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Datasets used for stage-2 7-expert end-to-end training (from stage2_end2end.yaml)
TRAINING_DATASETS = [
    "BUSI",
    "COVID_US",
    "ct2",
    "iu_xray",
    "PMC_VQA_FULL",
    "llava_instruct",
    "medtrinity_conversations_1_formatted",
    "medtrinity_conversations_2_formatted",
    "image_mammoth",
    "eye_dataset_converted",
    "skin_dataset_converted",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="training_data_review.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Samples per dataset (default 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=TRAINING_DATASETS,
        help="Subset of datasets to review (default: all 11).",
    )
    return parser.parse_args()


def get_pil_image(sample: dict) -> Image.Image | None:
    """Extract the first image from a modalities list."""
    for mod in sample.get("modalities", []):
        if mod.get("type") != "image":
            continue
        value = mod.get("value")
        if isinstance(value, dict) and value.get("bytes"):
            try:
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            except Exception:
                continue
        if isinstance(value, (bytes, bytearray)) and value:
            try:
                return Image.open(io.BytesIO(value)).convert("RGB")
            except Exception:
                continue
    return None


def get_text(sample: dict) -> str:
    """Extract assistant response from conversations, or text field."""
    for turn in sample.get("conversations", []):
        if turn.get("role") == "assistant":
            return turn.get("content", "")
    return sample.get("text", "")


def get_user_prompt(sample: dict) -> str:
    for turn in sample.get("conversations", []):
        if turn.get("role") == "user":
            content = turn.get("content", "")
            # Strip the image attachment token for display
            content = content.replace("<|reserved_special_token_0|>", "[IMAGE]")
            return content[:300] + ("…" if len(content) > 300 else "")
    return ""


def build_pdf(datasets_data: dict, output_path: str):
    styles = getSampleStyleSheet()
    story = []

    for ds_name, records in datasets_data.items():
        story.append(Paragraph(f"<b>Dataset: {ds_name}</b>", styles["Title"]))
        story.append(Paragraph(f"{len(records)} samples shown", styles["Normal"]))
        story.append(Spacer(1, 12))

        for i, (sample, pil_img) in enumerate(records):
            # Image thumbnail
            img_flowable = None
            if pil_img:
                thumb = pil_img.copy()
                thumb.thumbnail((6 * cm, 6 * cm))
                buf = io.BytesIO()
                thumb.save(buf, format="PNG")
                buf.seek(0)
                img_flowable = RLImage(buf, width=thumb.width, height=thumb.height)
            else:
                img_flowable = Paragraph("<i>[no image]</i>", styles["Normal"])

            # User prompt (right of image)
            user_txt = xml_escape(get_user_prompt(sample)).replace("\n", "<br/>")
            user_para = Paragraph(f"<b>User:</b><br/>{user_txt}", styles["BodyText"])

            table = Table(
                [[img_flowable, "", user_para]],
                colWidths=[6 * cm, 0.4 * cm, 11.6 * cm],
            )
            table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(table)
            story.append(Spacer(1, 8))

            # Assistant response
            asst_txt = xml_escape(get_text(sample)).replace("\n", "<br/>")
            story.append(
                Paragraph(f"<b>Assistant:</b><br/>{asst_txt}", styles["BodyText"])
            )
            story.append(Spacer(1, 20))

        story.append(PageBreak())

    doc = SimpleDocTemplate(
        output_path,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    doc.build(story)
    print(f"Written: {output_path}")


def main():
    args = parse_args()
    random.seed(args.seed)

    storage_root = os.environ.get(
        "STORAGE_ROOT",
        "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow",
    )

    datasets_data = {}
    for name in args.datasets:
        path = os.path.join(storage_root, name)
        if not os.path.exists(path):
            print(f"  SKIP {name}: path not found ({path})")
            continue
        print(f"Loading {name} …", flush=True)
        try:
            ds = load_from_disk(path)
            if hasattr(ds, "keys"):
                ds = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        except Exception as e:
            print(f"  SKIP {name}: {e}")
            continue

        n = min(args.num_samples, len(ds))
        indices = random.sample(range(len(ds)), n)

        records = []
        for idx in indices:
            sample = ds[idx]
            pil_img = get_pil_image(sample)
            records.append((sample, pil_img))

        print(f"  {name}: {n} samples, {sum(1 for _, img in records if img)} with images")
        datasets_data[name] = records

    print(f"\nBuilding PDF …")
    build_pdf(datasets_data, args.output)


if __name__ == "__main__":
    main()
