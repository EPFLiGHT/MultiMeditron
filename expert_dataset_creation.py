import argparse
import json
import os
from pathlib import Path

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description="Create expert-format dataset JSONL from stored subsets.")
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Subset names to process. If omitted and --all_subsets is not set, uses built-in defaults.",
    )
    parser.add_argument(
        "--all_subsets",
        action="store_true",
        help="Process every subset directory found under STORAGE_ROOT.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to export (default: train).",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path.home() / "expert_datasets"),
        help="Output dir for expert JSONL files.",
    )
    return parser.parse_args()


def get_text_from_sample(sample: dict) -> str:
    if isinstance(sample, dict):
        if sample.get("text"):
            return str(sample["text"]).strip()

        if sample.get("content"):
            return str(sample["content"]).strip()

        conversations = sample.get("conversations")
        if isinstance(conversations, list):
            for turn in conversations:
                if isinstance(turn, dict) and turn.get("role") == "assistant" and turn.get("content"):
                    return str(turn["content"]).strip()
            for turn in conversations:
                if isinstance(turn, dict) and turn.get("content"):
                    return str(turn["content"]).strip()

    return "No text provided."


def normalize_modalities(sample: dict) -> list[dict]:
    modalities = []
    if not isinstance(sample, dict):
        return modalities

    mlist = sample.get("modalities")
    if isinstance(mlist, list):
        for m in mlist:
            if not isinstance(m, dict):
                continue
            typ = m.get("type", "image")
            value = m.get("value")
            if value is None:
                # fallback to path field
                if m.get("path"):
                    value = m["path"]
                elif m.get("image"):
                    value = m["image"]
            if isinstance(value, dict):
                if "bytes" in value:
                    modalities.append({"type": typ, "value": {"bytes": value["bytes"]}})
                elif "path" in value:
                    modalities.append({"type": typ, "value": {"bytes": value["path"]}})
                else:
                    modalities.append({"type": typ, "value": value})
            elif isinstance(value, str):
                modalities.append({"type": typ, "value": {"bytes": value}})
            else:
                continue

    # fallback: if sample directly has 'image' or 'image_path'
    if not modalities:
        if sample.get("image"):
            modalities.append({"type": "image", "value": {"bytes": sample["image"]}})
        elif sample.get("image_path"):
            modalities.append({"type": "image", "value": {"bytes": sample["image_path"]}})

    return modalities


def export_subset(subset_name: str, split: str, storage_root: str, output_dir: Path) -> None:
    subset_path = os.path.join(storage_root, subset_name)
    if not os.path.isdir(subset_path):
        print(f"[WARN] subset path not found: {subset_path}")
        return

    try:
        dataset_dict = load_from_disk(subset_path)
    except Exception as e:
        print(f"[ERROR] failed to load subset {subset_name}: {e}")
        return

    if split not in dataset_dict:
        print(f"[ERROR] split '{split}' not found in subset '{subset_name}'. Available splits: {list(dataset_dict.keys())}")
        return

    ds = dataset_dict[split]
    print(f"Exporting subset '{subset_name}' split '{split}' ({len(ds)} samples)")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subset_name}_expert_{split}.jsonl"
    count = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(ds):
            text = get_text_from_sample(sample)
            modalities = normalize_modalities(sample)
            if not modalities:
                skipped += 1
                continue

            record = {
                "text": text,
                "modalities": modalities,
                "source_subset": subset_name,
                "source_index": idx,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"  -> wrote {count} records to {out_path} ({skipped} skipped without modalities)")


def list_subsets(storage_root: str) -> list[str]:
    all_names = [d for d in sorted(os.listdir(storage_root)) if os.path.isdir(os.path.join(storage_root, d))]
    return all_names


def main():
    args = parse_args()
    storage_root = os.environ.get("STORAGE_ROOT")
    if not storage_root:
        raise RuntimeError("STORAGE_ROOT environment variable is not set.")

    if args.all_subsets:
        subset_names = list_subsets(storage_root)
    elif args.subsets:
        subset_names = args.subsets
    else:
        subset_names = [
            "image_BUSI",
            "image_COVID_US",
            "image_DDTI",
            "image_PMC_VQA",
            "image_ct2",
            "image_iu_xray",
            "image_mammoth",
            "image_medtrinity_conversations_1",
            "image_medtrinity_conversations_1_formatted",
            "image_medtrinity_conversations_2",
            "image_medtrinity_conversations_2_formatted",
        ]

    if not subset_names:
        print("No subsets to process.")
        return

    print(f"STORAGE_ROOT={storage_root}")
    print(f"Processing subsets: {subset_names}")

    output_dir = Path(args.output_dir)
    for subset_name in subset_names:
        export_subset(subset_name, args.split, storage_root, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
