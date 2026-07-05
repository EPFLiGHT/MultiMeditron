import os
import io
import json
import argparse
from pathlib import Path

from datasets import load_from_disk
from PIL import Image

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_dir", default="context_examples")
    parser.add_argument("--output_prefix", default="dataset_visualization")
    parser.add_argument("--num_people", type=int, default=1)
    return parser.parse_args()


DATASET_CACHE = {}


def get_dataset(subset, storage_root):
    if subset not in DATASET_CACHE:
        path = os.path.join(storage_root, subset)
        DATASET_CACHE[subset] = load_from_disk(path)["train"]
    return DATASET_CACHE[subset]


def extract_image(record, storage_root):
    subset = record["source_subset"]
    index = record["source_index"]

    ds = get_dataset(subset, storage_root)
    sample = ds[index]

    for mod in sample.get("modalities", []):
        if mod.get("type") != "image":
            continue

        value = mod.get("value")

        if isinstance(value, dict) and "bytes" in value:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")

        if isinstance(value, (bytes, bytearray)):
            return Image.open(io.BytesIO(value)).convert("RGB")

        if isinstance(value, Image.Image):
            return value.convert("RGB")

    return None



def get_description(record):
    for msg in record.get("conversations", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def get_context(record):
    return record.get("generated_context", "")


def split_records(records, num_people):
    chunks = [[] for _ in range(num_people)]

    for i, rec in enumerate(records):
        chunks[i % num_people].append(rec)

    return chunks


# PDF CREATION
def create_pdf(records_by_subset, storage_root, output_path):

    styles = getSampleStyleSheet()
    story = []

    for subset_name, records in records_by_subset.items():

        story.append(Paragraph(f"<b>Subset: {subset_name}</b>", styles["Title"]))
        story.append(Spacer(1, 20))

        for record in records:

            # IMAGE
            pil_img = extract_image(record, storage_root)

            img_flowable = None
            if pil_img:
                pil_img.thumbnail((6 * cm, 6 * cm))

                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)

                img_flowable = RLImage(buf, width=pil_img.width, height=pil_img.height)

            # CONTEXT
            context = get_context(record)
            context_text = f"<b>Generated Context</b><br/>{context.replace('\n','<br/>')}"
            context_para = Paragraph(context_text, styles["BodyText"])

            # TABLE WITH SPACING COLUMN
            data = [[img_flowable, "", context_para]]

            table = Table(
                data,
                colWidths=[6 * cm, 0.5 * cm, 11.5 * cm],
            )

            table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )

            story.append(table)
            story.append(Spacer(1, 10))

            # DESCRIPTION BELOW
            desc = get_description(record)
            desc_text = f"<b>Description</b><br/>{desc.replace('\n','<br/>')}"
            story.append(Paragraph(desc_text, styles["BodyText"]))
            story.append(Spacer(1, 25))

        story.append(PageBreak())

    doc = SimpleDocTemplate(
        output_path,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    doc.build(story)


def main():

    args = parse_args()

    storage_root = os.environ.get("STORAGE_ROOT")
    if not storage_root:
        raise RuntimeError("STORAGE_ROOT not set")

    context_dir = Path(args.context_dir)
    jsonl_files = sorted(context_dir.glob("*.jsonl"))

    # Load all records grouped by subset
    all_data = {}

    for file in jsonl_files:
        subset = file.stem
        with open(file) as f:
            records = [json.loads(line) for line in f]
        all_data[subset] = records

    # Split per subset into N groups (no overlap)
    split_data_per_person = [{} for _ in range(args.num_people)]

    for subset, records in all_data.items():

        chunks = split_records(records, args.num_people)

        for i in range(args.num_people):
            split_data_per_person[i][subset] = chunks[i]

    # Generate PDFs
    for i in range(args.num_people):

        output_file = f"{args.output_prefix}_person{i+1}.pdf"

        print(f"Creating {output_file}")

        create_pdf(
            split_data_per_person[i],
            storage_root,
            output_file,
        )


if __name__ == "__main__":
    main()