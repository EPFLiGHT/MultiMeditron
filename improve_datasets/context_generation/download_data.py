from datasets import load_dataset, get_dataset_config_names
import os

STORAGE_ROOT = os.environ["STORAGE_ROOT"]
DS_NUM_PROC = int(os.environ["DS_NUM_PROC"])

dataset_name = "OpenMeditron/MultiMediset"
configs = get_dataset_config_names(dataset_name)

for split_name in configs:
    split_dir = os.path.join(STORAGE_ROOT, split_name)
    split_dataset = load_dataset(dataset_name, split_name, num_proc=DS_NUM_PROC)
    split_dataset.save_to_disk(split_dir)
