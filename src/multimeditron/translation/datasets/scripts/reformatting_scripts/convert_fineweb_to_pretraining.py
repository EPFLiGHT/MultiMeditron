"""
FineWeb to Meditron Pretraining Format Converter

Memory-efficient converter for FineWeb2 datasets to Meditron pretraining format.
Handles multi-GB, multi-million-line JSONL files efficiently using streaming processing.

Input: FineWeb2 JSONL files with 'text' field
Output: Meditron-formatted JSONL with {"text": "...", "modalities": []}

Usage:
    python convert_fineweb_to_pretraining.py
"""

import os
import ujson as json
from tqdm import tqdm


def reformat_to_pretraining(input_path, output_path, log_interval=10000):
    """
    Convert FineWeb JSONL to Meditron pretraining format.
    Streams line-by-line to minimize memory usage.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    pbar = tqdm(desc=f"Formatting {os.path.basename(input_path)}",
                unit=" lines", dynamic_ncols=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                if not text:
                    continue
                fout.write(json.dumps({"text": text, "modalities": []}, ensure_ascii=False) + "\n")
                count += 1
                if count % log_interval == 0:
                    pbar.update(log_interval)
            except Exception:
                continue

    pbar.update(count % log_interval)
    pbar.close()
    print(f"✅ Wrote {count:,} samples → {output_path}\n")


if __name__ == "__main__":
    src_train = "../../../nemo/datasets/polyglot/fineweb2_am/train.jsonl"
    src_test = "../../../nemo/datasets/polyglot/fineweb2_am/test.jsonl"

    dest_dir = "src/multimeditron/translation/datasets/formatted_datasets/general_datasets/fineweb/fineweb_am"
    os.makedirs(dest_dir, exist_ok=True)

    reformat_to_pretraining(src_train, os.path.join(dest_dir, "formatted_pretraining_train.jsonl"))
    reformat_to_pretraining(src_test, os.path.join(dest_dir, "formatted_pretraining_test.jsonl"))