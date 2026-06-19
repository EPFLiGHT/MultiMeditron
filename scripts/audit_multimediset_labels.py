#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from datasets import DatasetDict, load_from_disk


DEFAULT_BASE_ROOT = Path("/lightscratch/datasets/MultiMediset/general_purpose")
DEFAULT_MRI_ROOT = Path("/lightscratch/users/nemo/datasets/MRI_data/MRI-glob")
DEFAULT_DOC_PATH = Path("docs/source/multimediset_split_audit.md")
DEFAULT_RULES_PATH = Path("config/multimediset_label_rules.json")

TARGET_BENCHMARKS = {
    "ultrasound": ("COVID-US-2026", "BUSI", "DDTI"),
    "ct": ("CT2D-glob-mini",),
    "xray": ("XR-glob-mini",),
    "skin": ("SkinDataset",),
    "eye": ("EyeDataset",),
    "mri": ("MRI-glob",),
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    benchmark: str
    label_source: str
    extractor: object
    expected_labels: tuple = ()
    group_key: object = None
    notes: str = ""
    needs_review: bool = False


def first_modality_value(row):
    modalities = row.get("modalities") or []
    if not modalities:
        return ""
    first = modalities[0]
    if isinstance(first, dict):
        return str(first.get("value") or "")
    return str(first)


def lower_text(row):
    return str(row.get("text") or "").lower()


def path_parent_label(row, allowed):
    value = first_modality_value(row).replace("\\", "/")
    parts = [part.strip().lower() for part in value.split("/") if part.strip()]
    for part in parts:
        normalized = part.replace("_", "-")
        if normalized in allowed:
            return normalized
    return None


def busi_label(row):
    return path_parent_label(row, {"normal", "benign", "malignant"})


def covid_us_label(row):
    haystack = f"{first_modality_value(row)} {lower_text(row)}".lower()
    if "pneumonia" in haystack:
        return "pneumonia"
    if "covid" in haystack:
        return "covid"
    if "other" in haystack:
        return "other"
    if any(token in haystack for token in ("regular", "healthy", "normal")):
        return "normal"
    return None


def covid_us_2026_label(row):
    for key in ("class", "disease", "class_on_website"):
        value = str(row.get(key) or "").strip()
        if value and value.lower() != "none":
            return value
    return None


def ct2d_label(row):
    text = lower_text(row)
    if "tumor" in text:
        return "tumor"
    if "atherosoma" in text:
        return "atherosoma"
    if "glioblastoma" in text:
        return "glioblastoma"
    if "covid" in text:
        return "Covid"
    if text:
        return "healthy"
    return None


def ddti_label(row):
    text = lower_text(row)
    match = re.search(r"\btirads\s*([0-9]\s*[a-z]?)\b", text)
    if match:
        return "tirads_" + match.group(1).replace(" ", "")
    return None


def skin_label(row):
    haystack = f"{first_modality_value(row)} {lower_text(row)}".lower()
    rules = (
        ("basal-cell-carcinoma", ("basal cell carcinoma", "basal-cell-carcinoma")),
        ("benign-keratosis-like-lesions", ("benign keratosis", "benign-keratosis")),
        (
            "melanocytic-nevi",
            ("melanocytic nevus", "melanocytic nevi", "nevus", "nevi"),
        ),
        ("melanoma", ("melanoma",)),
        ("atopic-dermatitis", ("atopic dermatitis", "atopic-dermatitis")),
        ("eczema", ("eczema",)),
        ("psoriasis", ("psoriasis",)),
        ("tinea-ringworm-candidiasis", ("tinea", "ringworm", "candidiasis")),
        ("warts-molluscum-viral", ("wart", "molluscum", "viral infection")),
        ("seborrheic-keratoses", ("seborrheic keratos",)),
    )
    for label, needles in rules:
        if any(needle in haystack for needle in needles):
            return label
    return None


def eye_label(row):
    haystack = f"{first_modality_value(row)} {lower_text(row)}".lower()
    normal_markers = (
        "no-diabetic-retinopathy",
        "no diabetic retinopathy",
        "absence of diabetic retinopathy",
        "normal fundus",
        "healthy fundus",
        "healthy retina",
    )
    if any(marker in haystack for marker in normal_markers):
        return "normal"
    if "diabetic retinopathy" in haystack:
        return "diabetic_retinopathy"
    return None


def eye_patient_id(row):
    """Patient ID extracted from image path for sub-datasets that encode it.

    Messidor-2: 20051201_37462_0400_PP.jpg → 'messidor2_37462'
    EyePACS (sample_N) and others: None (no recoverable ID).
    """
    path = first_modality_value(row)
    if not path:
        return None
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) >= 2 and len(parts[0]) == 8 and parts[0].isdigit():
        return f"messidor2_{parts[1]}"
    return None


def mri_label(row):
    text = lower_text(row)
    if "brain tumor" in text:
        return "brain tumor"
    if "crohn" in text:
        return "crohn"
    if "bone infection" in text:
        return "Bone infection"
    if "healthy" in text:
        return "healthy"
    return None


XR_KEYWORDS = (
    "cardiomegaly",
    "atelectasis",
    "edema",
    "effusion",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "fibrosis",
    "emphysema",
    "nodule",
    "mass",
    "hernia",
)


def xr_keyword_label(row):
    text = lower_text(row)
    found = [keyword for keyword in XR_KEYWORDS if keyword in text]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        return "multi_keyword"
    if any(
        marker in text
        for marker in ("no acute", "unremarkable", "normal chest", "clear lungs")
    ):
        return "no_finding_candidate"
    return None


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dataset_like(root):
    if (root / "dataset_dict.json").exists() or (root / "state.json").exists():
        loaded = load_from_disk(str(root))
        if isinstance(loaded, DatasetDict):
            return {split: loaded[split] for split in loaded.keys()}
        return {"train": loaded}

    train_jsonl = root / "MRI-glob-train.jsonl"
    test_jsonl = root / "MRI-glob-test.jsonl"
    if train_jsonl.exists() and test_jsonl.exists():
        return {
            "train": load_jsonl(train_jsonl),
            "test": load_jsonl(test_jsonl),
        }

    jsonl = root / "MRI-glob.jsonl"
    if jsonl.exists():
        return {"all": load_jsonl(jsonl)}

    raise FileNotFoundError(f"Could not load dataset from {root}")


def row_at(split_data, idx):
    return dict(split_data[idx])


def split_size(split_data):
    return len(split_data)


def safe_columns(split_data, sample):
    if hasattr(split_data, "column_names"):
        return list(split_data.column_names)
    return list(sample.keys())


def summarize_dataset(spec, max_examples_per_split):
    dataset = load_dataset_like(spec.root)
    label_counts = {}
    missing_counts = {}
    group_counts = {}
    samples = []
    columns_by_split = {}
    sizes = {}

    for split_name, split_data in dataset.items():
        size = split_size(split_data)
        sizes[split_name] = size
        counts = Counter()
        groups = set()
        missing = 0
        limit = (
            size
            if max_examples_per_split is None
            else min(size, max_examples_per_split)
        )

        if size:
            first_row = row_at(split_data, 0)
            columns_by_split[split_name] = safe_columns(split_data, first_row)
        else:
            columns_by_split[split_name] = []

        for idx in range(limit):
            row = row_at(split_data, idx)
            label = spec.extractor(row)
            if label is None:
                missing += 1
            else:
                counts[label] += 1
            if (
                spec.group_key
                and spec.group_key in row
                and row.get(spec.group_key) is not None
            ):
                groups.add(str(row.get(spec.group_key)))
            if len(samples) < 3:
                samples.append(
                    {
                        "split": split_name,
                        "index": idx,
                        "label": label,
                        "modality_value": first_modality_value(row),
                        "text_preview": str(row.get("text") or "")[:180].replace(
                            "\n", " "
                        ),
                    }
                )

        label_counts[split_name] = counts
        missing_counts[split_name] = missing
        if spec.group_key:
            group_counts[split_name] = len(groups)

    return {
        "name": spec.name,
        "root": str(spec.root),
        "benchmark": spec.benchmark,
        "splits": sizes,
        "columns": columns_by_split,
        "label_source": spec.label_source,
        "expected_labels": list(spec.expected_labels),
        "group_key": spec.group_key,
        "label_counts": {split: dict(counts) for split, counts in label_counts.items()},
        "missing_label_counts": missing_counts,
        "group_counts": group_counts,
        "samples": samples,
        "notes": spec.notes,
        "needs_review": spec.needs_review,
    }


def make_specs(base_root, mri_root):
    return [
        DatasetSpec(
            "BUSI",
            base_root / "BUSI",
            "ultrasound",
            "modalities[0].value path parent",
            busi_label,
            expected_labels=("benign", "malignant", "normal"),
        ),
        DatasetSpec(
            "COVID-US-2026",
            base_root / "COVID-US-2026",
            "ultrasound",
            "class, then disease",
            covid_us_2026_label,
            expected_labels=("COVID", "Normal", "Other", "Pneumonia"),
            group_key="patient",
        ),
        DatasetSpec(
            "CT2D-glob-mini",
            base_root / "CT2D-glob-mini",
            "ct",
            "text keyword heuristic matching current CTBenchmark",
            ct2d_label,
            expected_labels=("Covid", "atherosoma", "glioblastoma", "healthy", "tumor"),
        ),
        DatasetSpec(
            "DDTI",
            base_root / "DDTI",
            "ultrasound",
            "tirads pattern in text",
            ddti_label,
            expected_labels=(
                "tirads_2",
                "tirads_3",
                "tirads_4a",
                "tirads_4b",
                "tirads_4c",
                "tirads_5",
            ),
            needs_review=True,
        ),
        DatasetSpec(
            "XR-glob-mini",
            base_root / "XR-glob-mini",
            "xray",
            "single chest finding keyword in free text",
            xr_keyword_label,
            expected_labels=(
                "atelectasis",
                "cardiomegaly",
                "consolidation",
                "edema",
                "effusion",
                "emphysema",
                "fibrosis",
                "hernia",
                "mass",
                "multi_keyword",
                "no_finding_candidate",
                "nodule",
                "pneumonia",
                "pneumothorax",
            ),
            needs_review=True,
            notes="Free-text X-ray reports are likely multi-label. This heuristic is only for audit, not final splitting.",
        ),
        DatasetSpec(
            "SkinDataset",
            base_root / "SkinDataset",
            "skin",
            "path/text dermatology keyword heuristic",
            skin_label,
            expected_labels=(
                "atopic-dermatitis",
                "basal-cell-carcinoma",
                "benign-keratosis-like-lesions",
                "eczema",
                "melanocytic-nevi",
                "melanoma",
                "psoriasis",
                "seborrheic-keratoses",
                "tinea-ringworm-candidiasis",
                "warts-molluscum-viral",
            ),
            needs_review=True,
        ),
        DatasetSpec(
            "EyeDataset",
            base_root / "EyeDataset",
            "eye",
            "path/text diabetic retinopathy heuristic",
            eye_label,
            expected_labels=("diabetic_retinopathy", "normal"),
            group_key=eye_patient_id,
        ),
        DatasetSpec(
            "MRI-glob",
            mri_root,
            "mri",
            "text keyword heuristic matching current MRIBenchmark",
            mri_label,
            expected_labels=("Bone infection", "brain tumor", "crohn", "healthy"),
        ),
    ]


def write_markdown(results, output_path, max_examples_per_split):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scope = (
        "all examples"
        if max_examples_per_split is None
        else f"first {max_examples_per_split} examples per split"
    )
    lines = [
        "# MultiMediset Split Audit",
        "",
        f"Audit scope: {scope}.",
        "",
        "This file records the first-pass label and grouping rules needed before generating disjoint splits for model training, MLP training, benchmark evaluation, and held-out testing.",
        "",
        "Target benchmark mapping:",
    ]
    for benchmark, dataset_names in TARGET_BENCHMARKS.items():
        joined = ", ".join(f"`{name}`" for name in dataset_names)
        lines.append(f"- `{benchmark}`: {joined}")
    lines.extend(
        [
            "",
            "Full `COVID-US` and full `CT2D-glob` are intentionally excluded from the target scope.",
            "",
        ]
    )

    for result in results:
        lines.extend(
            [
                f"## {result['name']}",
                "",
                f"- Root: `{result['root']}`",
                f"- Target benchmark: `{result['benchmark']}`",
                f"- Splits: `{result['splits']}`",
                f"- Label source: `{result['label_source']}`",
                f"- Expected labels: `{result['expected_labels']}`",
                f"- Group key: `{result['group_key']}`",
                f"- Needs review: `{result['needs_review']}`",
            ]
        )
        if result["notes"]:
            lines.append(f"- Notes: {result['notes']}")
        lines.append("")

        lines.append("Columns:")
        for split, columns in result["columns"].items():
            lines.append(f"- `{split}`: {', '.join(columns)}")
        lines.append("")

        lines.append("Label counts:")
        for split, counts in result["label_counts"].items():
            missing = result["missing_label_counts"].get(split, 0)
            ordered = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
            lines.append(f"- `{split}`: labels={ordered}, missing={missing}")
        if result["group_counts"]:
            lines.append("")
            lines.append("Group counts:")
            for split, count in result["group_counts"].items():
                lines.append(f"- `{split}`: {count}")
        lines.append("")

        lines.append("Samples:")
        for sample in result["samples"]:
            lines.append(
                f"- `{sample['split']}[{sample['index']}]`: label=`{sample['label']}`, "
                f"image=`{sample['modality_value']}`, text=`{sample['text_preview']}`"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_rules(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rules = {
        "version": 1,
        "description": "First-pass label/group rules for generating MultiMediset benchmark splits.",
        "target_benchmarks": TARGET_BENCHMARKS,
        "datasets": {},
    }
    for result in results:
        observed_labels = {
            label for counts in result["label_counts"].values() for label in counts
        }
        labels = sorted(set(result["expected_labels"]) | observed_labels)
        rules["datasets"][result["name"]] = {
            "root": result["root"],
            "benchmark": result["benchmark"],
            "splits": list(result["splits"].keys()),
            "label_source": result["label_source"],
            "labels": labels,
            "group_key": result["group_key"],
            "needs_review": result["needs_review"],
            "notes": result["notes"],
        }
    output_path.write_text(
        json.dumps(rules, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--mri-root", type=Path, default=DEFAULT_MRI_ROOT)
    parser.add_argument("--output-doc", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--output-rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=None,
        help="Limit audit work per split. By default, scan every example.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = [
        summarize_dataset(spec, max_examples_per_split=args.max_examples_per_split)
        for spec in make_specs(args.base_root, args.mri_root)
    ]
    write_markdown(results, args.output_doc, args.max_examples_per_split)
    write_rules(results, args.output_rules)
    print(f"Wrote {args.output_doc}")
    print(f"Wrote {args.output_rules}")


if __name__ == "__main__":
    main()
