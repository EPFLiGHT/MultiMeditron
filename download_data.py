"""Download the OpenMeditron/MultiMediset dataset and save each split to disk.

Reads two environment variables:
  STORAGE_ROOT  — directory to save the Arrow datasets into (one subdir per split)
  DS_NUM_PROC   — number of processes for the parallel download/save (default: 8)

Usage:
    STORAGE_ROOT=/path/to/arrow DS_NUM_PROC=32 python download_data.py
"""

import os
import sys

from datasets import load_dataset

DATASET_NAME = "OpenMeditron/MultiMediset"


def main() -> None:
    try:
        storage_root = os.environ["STORAGE_ROOT"]
    except KeyError:
        sys.exit("error: STORAGE_ROOT is not set (target directory for the datasets).")

    # num_proc must be an int; os.environ values are strings.
    num_proc = int(os.environ.get("DS_NUM_PROC", "8"))

    ds_dict = load_dataset(DATASET_NAME, num_proc=num_proc)

    for split_name, split_dataset in ds_dict.items():
        split_dir = os.path.join(storage_root, split_name)
        print(f"Saving split '{split_name}' -> {split_dir}")
        split_dataset.save_to_disk(split_dir)


if __name__ == "__main__":
    main()
