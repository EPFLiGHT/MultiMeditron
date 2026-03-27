import csv
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

import kagglehub
from Benchmark import Benchmark
from load_from_clip import load_model, encode_img
from mlp_eval import MLP_eval


XRAY_LABELS = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "No Finding",
]


def randomize_csv(input_path, seed=None):
    # function used to shuffle the dataset before using it for the benchmark
    df = pd.read_csv(input_path)
    df_randomized = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_randomized{ext}"

    df_randomized.to_csv(output_path, index=False)

    print(f"New file created : {output_path}")


def download_data():
    # Download the latest version of the NIH dataset used in this benchmark,
    # the dataset still needs to be shuffled with the function randomize_csv.
    path = kagglehub.dataset_download("nih-chest-xrays/data")

    print("Path to dataset files:", path)


def multi_label_f1_score(probabilities, labels, threshold=0.5):
    preds = (probabilities >= threshold).float()
    labels = labels.float()

    true_positive = (preds * labels).sum().item()
    false_positive = (preds * (1 - labels)).sum().item()
    false_negative = ((1 - preds) * labels).sum().item()

    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 1.0
    return (2 * true_positive) / denominator


def load_clip(model_id, is_lion_model):
    """Compatibility wrapper kept for callers that still import it."""
    return load_model(model_id)


def _resolve_xray_paths():
    env_data_root = os.environ.get("XRAY_DATA_ROOT")
    data_root = Path(env_data_root) if env_data_root else Path(__file__).resolve().parent / "xray_data"

    csv_path = data_root / "Data_Entry_2017_randomized.csv"
    images_root = Path(os.environ.get("XRAY_IMAGES_ROOT", data_root / "images"))
    kaggle_root = os.environ.get("XRAY_KAGGLE_DATA_ROOT")

    return data_root, csv_path, images_root, Path(kaggle_root) if kaggle_root else None


def _resolve_image_path(image_name, images_root, kaggle_root=None):
    direct_path = Path(images_root) / image_name
    candidates = [direct_path]

    # The repository may contain symlinks into a kagglehub cache. Keep using the
    # symlink target when it is still valid.
    if direct_path.is_symlink():
        link_target = direct_path.readlink()
        if not link_target.is_absolute():
            link_target = direct_path.parent / link_target
        candidates.append(link_target)

    # When the symlinks point to an ephemeral HOME, allow callers to pass the
    # extracted Kaggle dataset root directly and look up the image there.
    if kaggle_root is not None:
        for shard_dir in sorted(kaggle_root.glob("images_*/images")):
            candidates.append(shard_dir / image_name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not find NIH XRay image "
        f"{image_name!r}. Checked {len(candidates)} candidate paths. "
        f"images_root={images_root!s}. "
        "If your local images directory contains broken symlinks into a temporary kagglehub cache, "
        "set XRAY_IMAGES_ROOT to a persistent images directory or XRAY_KAGGLE_DATA_ROOT to the "
        "persistent root returned by kagglehub.dataset_download(...)."
    )


class Xray_Dataset(Dataset):
    def get_label(self, finding_labels):
        label_list = finding_labels.split('|')
        output = [0] * len(XRAY_LABELS)

        for i, label_name in enumerate(XRAY_LABELS):
            if label_name in label_list:
                output[i] = 1

        if output == [0] * len(XRAY_LABELS):
            print(f"Unknown labels encountered: {finding_labels}")

        return torch.tensor(output, dtype=torch.float32)

    def __init__(self, evaluated_clip, saving_name, rows, csv_path, images_root, kaggle_root=None):
        self.data = []
        self.label = []
        self.evaluated_clip = evaluated_clip
        data_root = os.path.dirname(csv_path)
        self.images_root = images_root
        self.kaggle_root = kaggle_root

        embeddings_dir = os.path.join(data_root, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        file_name_data = os.path.join(embeddings_dir, f"data_{saving_name}.pt")
        file_name_lab = os.path.join(embeddings_dir, f"lab_{saving_name}.pt")

        if os.path.exists(file_name_data) and os.path.exists(file_name_lab):
            self.data = torch.load(file_name_data, map_location="cpu")
            self.label = torch.load(file_name_lab, map_location="cpu")
            return

        encoded_images = []
        encoded_labels = []
        for row in tqdm(rows):
            image_path = _resolve_image_path(row["Image Index"], self.images_root, self.kaggle_root)
            encoded_images.append(encode_img(self.evaluated_clip, image_path).cpu())
            encoded_labels.append(self.get_label(row["Finding_Labels"]))

        if encoded_images:
            self.data = torch.stack(encoded_images)
            self.label = torch.stack(encoded_labels)
        else:
            self.data = torch.empty((0, 512), dtype=torch.float32)
            self.label = torch.empty((0, len(XRAY_LABELS)), dtype=torch.float32)

        torch.save(self.data, file_name_data)
        torch.save(self.label, file_name_lab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def eval_pipeline(clip_path, clip_name, csv_path, device):
    evaluated_clip = load_model(clip_path).to(device)
    evaluated_clip.eval()

    _, resolved_csv_path, images_root, kaggle_root = _resolve_xray_paths()
    csv_path = str(resolved_csv_path if csv_path is None else csv_path)

    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    train_end = int(len(rows) * 0.8)
    train_rows = rows[:train_end]
    test_rows = rows[train_end:]

    train_dataset = Xray_Dataset(evaluated_clip, clip_name + "_train", train_rows, csv_path, images_root, kaggle_root)
    test_dataset = Xray_Dataset(evaluated_clip, clip_name + "_test", test_rows, csv_path, images_root, kaggle_root)
    mlp_eval = MLP_eval(
        15,
        train_dataset,
        test_dataset,
        loss=nn.BCEWithLogitsLoss(),
        accuracy_function=multi_label_f1_score,
    )
    accuracy = mlp_eval.evaluate()
    return accuracy


class XRay_benchmark(Benchmark):
    def __init__(self, is_lion_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_lion_model = is_lion_model
        _, csv_path, _, _ = _resolve_xray_paths()
        self.csv_path = str(csv_path)

    def evaluate(self, clip_path):
        clip_name = os.path.basename(os.path.normpath(clip_path))
        self.clip_name = clip_name
        return eval_pipeline(clip_path, clip_name, self.csv_path, self.device)


if __name__ == "__main__":
    is_lion_model = sys.argv[2]
    download_data()
    xray_bench = XRay_benchmark(is_lion_model)
    xray_bench.evaluate(sys.argv[1])
