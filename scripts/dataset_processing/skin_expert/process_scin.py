"""
Prepare the SCIN (Skin Condition Image Network) dataset into a JSONL manifest
compatible with MultiMeditron (Skin Expert).

- Downloads images anonymously from public GCS
- Converts all images to safe RGB JPEG
- Merges cases + labels metadata
- Builds structured clinical text
- Writes a single JSONL manifest + metadata
"""

import ast
import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import fsspec
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import HfArgumentParser

logger = logging.getLogger(__name__)

# -------------------------
# Constants
# -------------------------

DEFAULT_BUCKET = "dx-scin-public-data"
DEFAULT_CASES_CSV = "dataset/scin_cases.csv"
DEFAULT_LABELS_CSV = "dataset/scin_labels.csv"

IMAGE_COL_PATTERN = re.compile(r"^image_(\d+)_path$")

# -------------------------
# Arguments
# -------------------------

@dataclass
class SCINPrepArguments:
    gcs_bucket: str = field(
        default=DEFAULT_BUCKET,
        metadata={"help": "Public GCS bucket containing SCIN data"},
    )
    cases_csv: str = field(
        default=DEFAULT_CASES_CSV,
        metadata={"help": "Path to scin_cases.csv inside the bucket"},
    )
    labels_csv: str = field(
        default=DEFAULT_LABELS_CSV,
        metadata={"help": "Path to scin_labels.csv inside the bucket"},
    )
    output_dir: Path = field(
        metadata={"help": "Directory where images will be saved"}
    )
    output_jsonl: Path = field(
        metadata={"help": "Path to output JSONL manifest"}
    )
    jpeg_quality: int = field(
        default=95,
        metadata={"help": "JPEG quality for saved images"},
    )

# -------------------------
# Utilities
# -------------------------

def read_csv_gcs(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        storage_options={"token": "anon"},
        dtype={"case_id": str},
    )

def save_as_jpeg(raw: bytes, out_path: Path, quality: int) -> None:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=quality, optimize=True)

def yes_list_from_prefix(row: pd.Series, prefix: str) -> List[str]:
    return [
        c[len(prefix):].replace("_", " ").lower()
        for c in row.index
        if c.startswith(prefix)
        and isinstance(row[c], str)
        and row[c].strip().upper() == "YES"
    ]

def parse_weighted_labels(val: Optional[str]) -> Dict[str, float]:
    if not isinstance(val, str) or not val.strip():
        return {}
    try:
        d = ast.literal_eval(val)
        if isinstance(d, dict):
            return {str(k): float(v) for k, v in d.items()}
    except Exception:
        pass
    return {}

def top_k_labels_text(weighted: Dict[str, float], k: int = 5) -> str:
    items = sorted(weighted.items(), key=lambda x: x[1], reverse=True)[:k]
    return ", ".join(f"{name} ({w:.2f})" for name, w in items)

def build_text(row: pd.Series) -> str:
    parts = []
    if isinstance(row.get("age_group"), str):
        parts.append(f"Age group: {row['age_group'].replace('_',' ').title()}.")
    if isinstance(row.get("sex_at_birth"), str):
        parts.append(f"Sex at birth: {row['sex_at_birth'].replace('_',' ').title()}.")
    if isinstance(row.get("fitzpatrick_skin_type"), str):
        parts.append(f"Fitzpatrick skin type: {row['fitzpatrick_skin_type']}.")
    races = yes_list_from_prefix(row, "race_ethnicity_")
    if races:
        parts.append("Race/ethnicity: " + ", ".join(races) + ".")
    weighted = parse_weighted_labels(row.get("weighted_skin_condition_label"))
    if weighted:
        parts.append(
            "Dermatologist differential (weighted): "
            + top_k_labels_text(weighted)
            + "."
        )
    parts.append("<attachment>")
    return " ".join(parts)

# -------------------------
# Main logic
# -------------------------

def main():
    parser = HfArgumentParser(SCINPrepArguments)
    (args,) = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    fs = fsspec.filesystem("gs", token="anon")

    cases_path = f"gs://{args.gcs_bucket}/{args.cases_csv}"
    labels_path = f"gs://{args.gcs_bucket}/{args.labels_csv}"

    logger.info("Reading SCIN CSVs from public GCS")
    cases = read_csv_gcs(cases_path)
    labels = read_csv_gcs(labels_path)

    df = pd.merge(cases, labels, on="case_id", how="left")
    df["case_id"] = df["case_id"].astype(str)

    image_cols = [
        c for c in df.columns
        if IMAGE_COL_PATTERN.match(c)
    ]
    if not image_cols:
        raise RuntimeError("No image_*_path columns found in scin_cases.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = saved = failed = 0

    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="SCIN cases"):
            case_id = row["case_id"]

            # choose first available image deterministically
            img_rel = None
            for col in image_cols:
                val = row.get(col)
                if isinstance(val, str) and val.strip():
                    img_rel = val
                    break
            if not img_rel:
                continue

            out_img = args.output_dir / f"{case_id}.jpg"
            if not out_img.exists():
                try:
                    with fs.open(f"gs://{args.gcs_bucket}/{img_rel}", "rb") as f:
                        raw = f.read()
                    save_as_jpeg(raw, out_img, args.jpeg_quality)
                    saved += 1
                except Exception as e:
                    failed += 1
                    logger.warning(f"Image download failed for case {case_id}: {e}")
                    continue

            record = {
                "text": build_text(row),
                "modalities": [
                    {
                        "type": "image",
                        "value": str(out_img.relative_to(args.output_jsonl.parent)),
                    }
                ],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    meta = {
        "dataset": "SCIN",
        "source": f"gs://{args.gcs_bucket}",
        "num_cases": written,
        "images_saved": saved,
        "image_failures": failed,
        "image_columns": image_cols,
    }
    meta_path = args.output_jsonl.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"[DONE] JSONL written: {written}")
    logger.info(f"[IMAGES] saved={saved}, failed={failed}")
    logger.info(f"[META] {meta_path}")

if __name__ == "__main__":
    main()
