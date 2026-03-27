from random import choices, seed, shuffle
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
from sklearn.metrics import classification_report
import copy
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from load_from_clip import load_model, preprocess_dataset, preprocess_dataset_biomed, make

# Logs
logger = logging.getLogger(__name__)

# Config
dataset_path = "/mloscratch/users/nemo/datasets/CT_data/CT2D-glob"
dataset_path_jsonl = "/mloscratch/users/tagemoua/processing-scripts/experts/CT2D-glob.jsonl"
train_rate = 0.8
save_nn = True
sep_by_patient = False
num_epochs = 1000
progress_file = "progress.json"

labels = ["atherosoma", "Covid", "healthy", "glioblastoma", "tumor"]

def find_label(example: dict):
    return "tumor" if "tumor" in example["text"] else "atherosoma" if "atherosoma" in example["text"] else "glioblastoma" if "glioblastoma" in example["text"] else "Covid" if "Covid" in example["text"] else "healthy"

def get_patient_id(example: dict) -> str:
    return example["modalities"][0]["value"].split("_")[0].split("/")[-1]

# Get available models
clips = [(model, "/mloscratch/users/mberruye/processing-scripts/experts/models" + model) for model in os.listdir("/mloscratch/users/mberruye/processing-scripts/experts/models") if model.startswith("combined")]

# Load dataset
logger.info("Loading dataset")
with open(dataset_path_jsonl, "r") as f:
    lines = [json.loads(line) for line in f if "healthy" in line or "atherosoma" in line or "Covid" in line or "glioblastoma" in line or "tumor" in line]
logger.info(f"{len(lines)} examples")
data_lines = lines

# Load progress file
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        processed_models = set(json.load(f))
else:
    processed_models = set()

def save_progress(name_clip):
    processed_models.add(name_clip)
    with open(progress_file, "w") as f:
        json.dump(sorted(list(processed_models)), f)

def setup_model(seed=42) -> nn.Module:
    input_size = 512
    num_classes = len(labels)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    class SimpleNN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x

    model = SimpleNN(input_size, num_classes)
    initialize_weights(model)
    return model

def train_mlp(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, num_epochs: int = 1000) -> Tuple[nn.Module, int, List[dict]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_epoch = -1
    best_val_accuracy = 0.0
    best_model_params = None
    history = {"train_loss": [], "val_accuracy": [], "classification_report": []}

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        history["train_loss"].append(loss.item() / len(X_train))

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            y_pred_classes = torch.argmax(y_pred, axis=1)
            val_accuracy = (y_pred_classes == y_test).float().mean()

        history["val_accuracy"].append(val_accuracy.item())
        history["classification_report"].append(classification_report(y_test.numpy(), y_pred_classes.numpy(), target_names=labels, labels=list(range(len(labels))), output_dict=True))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    return model, best_epoch, history

# Preprocess images
logger.info("Processing dataset…")
data_lines_img = preprocess_dataset(data_lines, dataset_path)

for name_clip, path_model in clips:
    print("Name of the CLIP model:", name_clip)
    if name_clip in processed_models:
        print(f"{name_clip} already processed, skipping.")
        continue

    embeds_path = os.path.join("embeds", f"images-{name_clip}-eval.pt")
    os.makedirs(os.path.dirname(embeds_path), exist_ok=True)

    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    label_names = [find_label(line) for line in data_lines]
    labels_numeric = np.array([label_to_idx[label] for label in label_names])

    if not os.path.exists(embeds_path):
        logger.info(f"Loading CLIP model {name_clip}")
        model = load_model(path_model)
        model.eval()
        logger.info("Making embeds…")
        image_embeds = make(model, data_lines_img)

        X = torch.tensor(image_embeds, dtype=torch.float32)
        y = torch.tensor(labels_numeric, dtype=torch.long)
        torch.save(X, embeds_path)
        torch.save(y, embeds_path + "-y.pt")
    else:
        logger.info(f"Loading embeddings for {name_clip}")
        X = torch.load(embeds_path)
        y = torch.load(embeds_path + "-y.pt")

    # Train-test split
    train_size = int(train_rate * len(X))
    test_size = len(X) - train_size
    seed(42)
    np.random.seed(42)

    if sep_by_patient:
        patients = [get_patient_id(example) for example in data_lines]
        patients_set = sorted(set(patients))
        train_numbers = choices(patients_set, weights=[sum(x == i for x in patients) for i in patients_set], k=round(0.8 * len(patients_set)))
        train_ids, test_ids = [], []
        for i, path in enumerate(patients):
            if path in train_numbers:
                train_ids.append(i)
            else:
                test_ids.append(i)
    else:
        ids = list(range(len(data_lines)))
        shuffle(ids)
        train_ids = ids[:train_size]
        test_ids = ids[train_size:]

    X_train, X_test = X[train_ids], X[test_ids]
    y_train, y_test = y[train_ids], y[test_ids]

    logger.info(f"Training {name_clip}")
    best_model, best_accuracy, best_history, best_seed = None, 0, None, 0
    for model_seed in (42, 3, 314, 1602, 512):
        model = setup_model(model_seed)
        new_model, best_epoch, history = train_mlp(model, X_train, y_train, X_test, y_test, num_epochs)
        final_accuracy = history["val_accuracy"][best_epoch]

        if best_model is None or final_accuracy > best_accuracy:
            best_model = new_model
            best_accuracy = final_accuracy
            best_history = history
            best_seed = model_seed

    logger.info(f"CLIP: {name_clip} Final accuracy: {best_accuracy}")
    os.makedirs("analysis", exist_ok=True)
    with open(f"analysis/{name_clip}_report.jsonl", "w") as f:
        for epoch in range(num_epochs):
            data_epoch = {k: v[epoch] for k, v in best_history.items()}
            f.write(json.dumps(data_epoch) + "\n")

    if save_nn:
        os.makedirs("models", exist_ok=True)
        torch.save(best_model.state_dict(), f'models/{name_clip}-seed-{best_seed}.pth')

    # Mark this model as done
    save_progress(name_clip)
