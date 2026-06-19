"""
Skin Disease Classification Benchmark on Frozen CLIP Embeddings.

This module defines a downstream evaluation protocol for CLIP-style models in which
image embeddings are extracted from a frozen vision encoder and used to train a small
neural classifier for skin disease classification. The benchmark measures overall
accuracy, per-class accuracy, and optionally produces a confusion matrix.

The benchmark is designed to probe representation quality rather than end-to-end
task performance, enabling fair comparison between different CLIP training runs.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from Benchmark import Benchmark
from load_from_clip import load_model, encode_img
from sklearn.metrics import confusion_matrix


class SkinClassifier(torch.nn.Module):
    def __init__(
        self, input_dim=512, hidden_dim=50, num_classes=20, init_mode="xavier"
    ):
        super().__init__()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, 30)
        self.lin3 = torch.nn.Linear(30, num_classes)

        if init_mode == "xavier":
            torch.nn.init.xavier_uniform_(self.lin1.weight)
            torch.nn.init.xavier_uniform_(self.lin2.weight)
            torch.nn.init.xavier_uniform_(self.lin3.weight)
        elif init_mode == "kaiming":
            torch.nn.init.kaiming_normal_(
                self.lin1.weight, mode="fan_in", nonlinearity="sigmoid"
            )
            torch.nn.init.kaiming_normal_(
                self.lin2.weight, mode="fan_in", nonlinearity="sigmoid"
            )
            torch.nn.init.kaiming_normal_(
                self.lin3.weight, mode="fan_in", nonlinearity="sigmoid"
            )
        elif init_mode == "orthogonal":
            torch.nn.init.orthogonal_(self.lin1.weight)
            torch.nn.init.orthogonal_(self.lin2.weight)
            torch.nn.init.orthogonal_(self.lin3.weight)

    def forward(self, x):
        x = torch.sigmoid(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        x = self.lin3(x)  # logits
        return x


def load_skin_dataset(jsonl_path, clip_model, image_root, label2id):
    """
    Reads JSONL with:
        - "text": disease category (string)
        - "modalities"[0]["value"]: relative image path
    Returns (embeddings, labels) as tensors.
    """
    X, Y = [], []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            label_text = ex["text"]
            if label_text not in label2id:
                # skip unknown label if any
                continue
            label_id = label2id[label_text]

            rel_path = ex["modalities"][0][
                "value"
            ]  # e.g. "images/eczema-subacute-68.jpg"
            img_path = os.path.join(image_root, rel_path)

            emb = encode_img(clip_model, img_path)  # 1D tensor
            X.append(emb)
            Y.append(label_id)

    X = torch.stack(X)  # [N, D]
    Y = torch.tensor(Y, dtype=torch.long)  # [N]
    return X, Y


class SkinDiseaseTrainDataset(Dataset):
    def __init__(
        self,
        clip_model,
        model_name,
        jsonl_path,
        image_root,
        label2id,
        cache_dir="",
        load_cached=False,
    ):
        emb_file = os.path.join(cache_dir, f"skin_train_emb_{model_name}.pt")
        lab_file = os.path.join(cache_dir, f"skin_train_lab_{model_name}.pt")

        if not load_cached:
            X, Y = load_skin_dataset(jsonl_path, clip_model, image_root, label2id)
            self.data = X
            self.labels = Y
            if cache_dir:
                torch.save(self.data, emb_file)
                torch.save(self.labels, lab_file)
        else:
            self.data = torch.load(emb_file)
            self.labels = torch.load(lab_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SkinDiseaseTestDataset(Dataset):
    def __init__(
        self,
        clip_model,
        model_name,
        jsonl_path,
        image_root,
        label2id,
        cache_dir="",
        load_cached=False,
    ):
        emb_file = os.path.join(cache_dir, f"skin_test_emb_{model_name}.pt")
        lab_file = os.path.join(cache_dir, f"skin_test_lab_{model_name}.pt")

        if not load_cached:
            X, Y = load_skin_dataset(jsonl_path, clip_model, image_root, label2id)
            self.data = X
            self.labels = Y
            if cache_dir:
                torch.save(self.data, emb_file)
                torch.save(self.labels, lab_file)
        else:
            self.data = torch.load(emb_file)
            self.labels = torch.load(lab_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_skin_classifier(seed, init_method, train_loader, train_dataset, num_classes):
    torch.manual_seed(seed)

    model = SkinClassifier(
        input_dim=train_dataset.data.shape[1],
        num_classes=num_classes,
        init_mode=init_method,
    )

    labels_np = np.array(train_dataset.labels)
    classes = np.unique(labels_np.astype(int))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_np,
    )
    weights = torch.tensor(class_weights, dtype=torch.float)

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_EPOCH = 100  # you can reduce this to speed up Optuna
    model.train()
    for epoch in range(N_EPOCH):
        epoch_loss = 0.0
        for inputs, lab in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, lab)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    return avg_loss, model


def multiple_train_skin(train_loader, train_dataset, num_classes):
    configs = [
        (41, "xavier"),
        (14, "kaiming"),
        (5, "orthogonal"),
    ]
    runs = [
        train_skin_classifier(seed, init, train_loader, train_dataset, num_classes)
        for seed, init in configs
    ]
    losses = [r[0] for r in runs]
    best_idx = losses.index(min(losses))
    best_model = runs[best_idx][1]
    return best_model


def evaluate_skin_classifier(model, test_loader, num_classes, return_preds=False):
    model.eval()
    total = 0
    correct = 0

    per_class_correct = np.zeros(num_classes, dtype=int)
    per_class_total = np.zeros(num_classes, dtype=int)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)  # logits
            preds = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

            for p, y in zip(preds, labels):
                y = int(y.item())
                p = int(p.item())
                per_class_total[y] += 1
                if p == y:
                    per_class_correct[y] += 1

            if batch_idx == 0:
                print("[Eval] first batch predictions:", preds[:10].tolist())
                print("[Eval] first batch labels     :", labels[:10].tolist())

    acc = correct / total if total > 0 else 0.0
    print(
        f"Skin classifier accuracy: {acc * 100:.2f}% (correct={correct}, total={total})"
    )

    per_class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        if per_class_total[c] > 0:
            per_class_acc[c] = per_class_correct[c] / per_class_total[c]

    if return_preds:
        return (
            acc,
            per_class_acc,
            per_class_total,
            np.array(all_labels),
            np.array(all_preds),
        )
    else:
        return acc, per_class_acc, per_class_total


class SkinDiseaseBenchmark(Benchmark):
    """
    Benchmark: skin disease classification from CLIP image embeddings.

    For each CLIP model at `model_path`:
      - load CLIP encoder
      - compute embeddings for train & test skin datasets
      - train a small classifier on train embeddings
      - evaluate accuracy on test embeddings
      - return accuracy to be maximized by Optuna
    """

    def __init__(
        self, train_jsonl, test_jsonl, image_root, cache_dir="", batch_size=64
    ):
        self.train_jsonl = train_jsonl
        self.test_jsonl = test_jsonl
        self.image_root = image_root
        self.cache_dir = cache_dir
        self.batch_size = batch_size

        # Build label mapping from training file: text -> id
        self.label2id = self._build_label_map(train_jsonl)
        self.num_classes = len(self.label2id)
        print(f"SkinDiseaseBenchmark: {len(self.label2id)} unique labels")

    def _build_label_map(self, jsonl_path):
        label2id = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                label_text = ex["text"]  # e.g. "Eczema Photos"
                if label_text not in label2id:
                    label2id[label_text] = len(label2id)
        return label2id

    def evaluate(self, model_path):
        # model_path is training_args.output_dir for this Optuna trial
        model_name = os.path.basename(model_path.rstrip("/"))
        print(f"[SkinBenchmark] model_path = {model_path}")

        # 1) Load the CLIP image encoder for this trial
        clip_model = load_model(model_path)

        # 2) Build train/test datasets
        train_dataset = SkinDiseaseTrainDataset(
            clip_model=clip_model,
            model_name=model_name,
            jsonl_path=self.train_jsonl,
            image_root=self.image_root,
            label2id=self.label2id,
            cache_dir="",  # no cross-model caching
            load_cached=False,
        )
        test_dataset = SkinDiseaseTestDataset(
            clip_model=clip_model,
            model_name=model_name,
            jsonl_path=self.test_jsonl,
            image_root=self.image_root,
            label2id=self.label2id,
            cache_dir="",
            load_cached=False,
        )
        print(
            f"[SkinBenchmark] train size = {len(train_dataset)}, test size = {len(test_dataset)}"
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # 3) Train classifier on embeddings
        classifier = multiple_train_skin(train_loader, train_dataset, self.num_classes)

        # 4) Evaluate and get per-class stats
        acc, per_class_acc, per_class_total = evaluate_skin_classifier(
            classifier, test_loader, self.num_classes
        )

        # 5) Save metrics for plotting
        id2label = {v: k for k, v in self.label2id.items()}
        out_path = os.path.join(model_path, "skin_per_class_metrics.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "overall_acc": acc,
                    "id2label": {int(k): v for k, v in id2label.items()},
                    "per_class_acc": per_class_acc.tolist(),
                    "per_class_total": per_class_total.tolist(),
                },
                f,
                indent=2,
            )
        print(f"[SkinBenchmark] saved per-class metrics to {out_path}")

        return float(acc)

    def evaluate_with_confusion(self, model_path):
        model_name = os.path.basename(model_path.rstrip("/"))
        print(f"[SkinBenchmark] model_path = {model_path}")

        clip_model = load_model(model_path)

        train_dataset = SkinDiseaseTrainDataset(
            clip_model=clip_model,
            model_name=model_name,
            jsonl_path=self.train_jsonl,
            image_root=self.image_root,
            label2id=self.label2id,
            cache_dir="",
            load_cached=False,
        )
        test_dataset = SkinDiseaseTestDataset(
            clip_model=clip_model,
            model_name=model_name,
            jsonl_path=self.test_jsonl,
            image_root=self.image_root,
            label2id=self.label2id,
            cache_dir="",
            load_cached=False,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        classifier = multiple_train_skin(train_loader, train_dataset, self.num_classes)

        acc, per_class_acc, per_class_total, y_true, y_pred = evaluate_skin_classifier(
            classifier, test_loader, self.num_classes, return_preds=True
        )

        # build confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))

        id2label = {v: k for k, v in self.label2id.items()}
        return acc, per_class_acc, per_class_total, cm, id2label
