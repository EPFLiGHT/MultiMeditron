import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import kagglehub
except ModuleNotFoundError:
    kagglehub = None

from .base import ClassificationBenchmark
from ..load_from_clip import encode_img


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


def _resolve_xray_paths():
    env_data_root = os.environ.get("XRAY_DATA_ROOT")
    if env_data_root:
        data_root = Path(env_data_root)
    else:
        module_data_root = Path(__file__).resolve().parent / "xray_data"
        legacy_data_root = Path(__file__).resolve().parents[1] / "xray_data"
        data_root = module_data_root if module_data_root.exists() else legacy_data_root

    csv_path = data_root / "Data_Entry_2017.csv"
    images_root = Path(os.environ.get("XRAY_IMAGES_ROOT", data_root / "images"))
    kaggle_root = os.environ.get("XRAY_KAGGLE_DATA_ROOT")

    return data_root, csv_path, images_root, Path(kaggle_root) if kaggle_root else None


def _resolve_image_path(image_name, images_root, kaggle_root=None):
    direct_path = Path(images_root) / image_name
    candidates = [direct_path]

    if direct_path.is_symlink():
        link_target = direct_path.readlink()
        if not link_target.is_absolute():
            link_target = direct_path.parent / link_target
        candidates.append(link_target)

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
    """PyTorch Dataset of NIH ChestX-ray14 image embeddings and multi-hot labels.

    Embeddings are computed once and cached to disk as .pt files under
    {data_root}/embeddings/. Labels are multi-hot vectors of length 15.
    """

    def get_label(self, finding_labels):
        label_list = finding_labels.split("|")
        output = [0] * len(XRAY_LABELS)
        for i, label_name in enumerate(XRAY_LABELS):
            if label_name in label_list:
                output[i] = 1
        if output == [0] * len(XRAY_LABELS):
            print(f"Unknown labels encountered: {finding_labels}")
        return torch.tensor(output, dtype=torch.float32)

    def __init__(
        self, evaluated_clip, saving_name, rows, csv_path, images_root, kaggle_root=None
    ):
        self.evaluated_clip = evaluated_clip
        self.images_root = images_root
        self.kaggle_root = kaggle_root

        data_root = os.path.dirname(csv_path)
        embeddings_dir = os.path.join(data_root, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        file_name_data = os.path.join(embeddings_dir, f"data_{saving_name}.pt")
        file_name_lab = os.path.join(embeddings_dir, f"lab_{saving_name}.pt")

        if os.path.exists(file_name_data) and os.path.exists(file_name_lab):
            self.data = torch.load(file_name_data, map_location="cpu", weights_only=True)
            self.labels = torch.load(file_name_lab, map_location="cpu", weights_only=True)
            return

        encoded_images = []
        encoded_labels = []
        for row in tqdm(rows):
            image_path = _resolve_image_path(
                row["Image Index"], self.images_root, self.kaggle_root
            )
            encoded_images.append(encode_img(self.evaluated_clip, image_path).cpu())
            encoded_labels.append(self.get_label(row["Finding Labels"]))

        if encoded_images:
            self.data = torch.stack(encoded_images)
            self.labels = torch.stack(encoded_labels)
        else:
            self.data = torch.empty((0, 512), dtype=torch.float32)
            self.labels = torch.empty((0, len(XRAY_LABELS)), dtype=torch.float32)

        torch.save(self.data, file_name_data)
        torch.save(self.labels, file_name_lab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class XRay_benchmark(ClassificationBenchmark):
    """NIH ChestX-ray14 multi-label classification benchmark.

    Evaluates a vision encoder on 15 pathology classes using a multi-label MLP.
    An 80/20 patient-level train/test split is applied deterministically (seed 42).
    Loss is BCEWithLogitsLoss and the multi-label score is micro F1.
    """

    num_classes = len(XRAY_LABELS)
    max_train_examples = 5_000
    max_test_examples = 3_000

    def __init__(
        self,
        max_train_examples=None,
        max_test_examples=None,
        cache_root=None,
    ):
        super().__init__(
            cache_root=cache_root,
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
        )
        _, csv_path, self.images_root, self.kaggle_root = _resolve_xray_paths()
        self.csv_path = str(csv_path)
        self._split_cache = None

    def build_loss(self, train_dataset):
        return nn.BCEWithLogitsLoss()

    def build_mlp_kwargs(self):
        return {"accuracy_function": multi_label_f1_score}

    def _get_split_rows(self):
        if self._split_cache is None:
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            # Group by patient to prevent the same patient appearing in both splits.
            # NIH ChestX-ray14 has multiple images per patient (longitudinal follow-ups).
            patient_to_rows = defaultdict(list)
            for row in rows:
                patient_to_rows[row["Patient ID"]].append(row)

            patient_ids = sorted(patient_to_rows.keys())
            rng = random.Random(42)
            rng.shuffle(patient_ids)

            train_end = int(len(patient_ids) * 0.8)
            train_rows = [
                row
                for pid in patient_ids[:train_end]
                for row in patient_to_rows[pid]
            ]
            test_rows = [
                row
                for pid in patient_ids[train_end:]
                for row in patient_to_rows[pid]
            ]

            if self.max_train_examples is not None:
                train_rows = train_rows[: self.max_train_examples]
            if self.max_test_examples is not None:
                test_rows = test_rows[: self.max_test_examples]
            self._split_cache = (train_rows, test_rows)
        return self._split_cache

    def build_train_dataset(self, model, model_name, use_cache=True):
        train_rows, _ = self._get_split_rows()
        return Xray_Dataset(
            model, model_name + "_train", train_rows, self.csv_path,
            self.images_root, self.kaggle_root,
        )

    def build_test_dataset(self, model, model_name, use_cache=True):
        _, test_rows = self._get_split_rows()
        return Xray_Dataset(
            model, model_name + "_test", test_rows, self.csv_path,
            self.images_root, self.kaggle_root,
        )
