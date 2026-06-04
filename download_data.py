from datasets import load_dataset
import os

STORAGE_ROOT = os.environ["STORAGE_ROOT"]
DS_NUM_PROC = os.environ["DS_NUM_PROC"]

dataset_name = "OpenMeditron/MultiMediset"

ds_dict = load_dataset(dataset_name, num_proc=DS_NUM_PROC)

for split_name, split_dataset in ds_dict.items():
    split_dir = os.path.join(STORAGE_ROOT, split_name)
    split_dataset.save_to_disk(split_dir)
