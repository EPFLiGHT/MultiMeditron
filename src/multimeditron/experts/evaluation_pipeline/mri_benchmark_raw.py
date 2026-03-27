from random import choices, seed, shuffle
from tqdm import tqdm
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import copy
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

import logging

from load_from_clip import load_model, preprocess_dataset, preprocess_dataset_biomed, make

# Specify here the CLIP-based models you want to test
# The first element in the tuple is used for naming the model in the logs and when saving the weights of the classification neural networl
# So write whatever you want that makes you tell the models apart

# The second element in the tuple is either a reference to a Hugging Face model or the local path to the model/checkpoint
clips = [("standard_clip", "openai/clip-vit-base-patch32")]
clips = [(model, "/mloscratch/users/noize/processing-scripts/experts/models/" + model) for model in os.listdir("/mloscratch/users/noize/processing-scripts/experts/models") if model.startswith("combined")]

dataset_path = "/mloscratch/users/nemo/datasets/MRI_data/MRI-glob"
dataset_path_jsonl = "/mloscratch/users/tagemoua/processing-scripts/experts/MRI-5.jsonl"

# Dataset specific part
train_rate = 0.8

labels = ["brain tumor", "Crohn", "healthy", "Bone infection"]
def find_label(example: dict):
    """
    Determine the label from an example in the dataset.

    Args:
      example, an example from the dataset (format: {"text": text, "modalities": [{"type": type, "value": path_to_image}]})
    
    Returns the str of the identified label. It must be all the elements from labels defined right above this function.
    """
    return "brain tumor" if "brain tumor" in example["text"] else "crohn" if "crohn" in example["text"] else "Bone infection" if "Bone infection" in example["text"] else "healthy"

sep_by_patient = False
#if there are several examples with the same patient, set sep_by_patient to True and define get_find_patient_id based on the dataset
#else, all the examples are assumed to be independent and can be randomly shuffled
def get_patient_id(example: dict) -> str:
    """
    Determine the patient ID of an example.

    Args:
      example, an example from the dataset (format: {"text": text, "modalities": [{"type": type, "value": path_to_image}]})
    
    Returns the patient ID as an str.
    """
    return example["modalities"][0]["value"].split("_")[0].split("/")[-1]

# Specify here whether you want the script to save the weights of the classification neural network
# The script will save one file per CLIP-based model
save_nn = True

#######################################################################################################################

# Logs
logger = logging.getLogger(__name__)

# Get the data from the dataset
logger.info(f"Loading dataset")
with open(dataset_path_jsonl, "r") as f:
    lines = [json.loads(line) for line in f if "healthy" in line or "brain tumor" in line or "crohn" in line or "Bone infection" in line]
logger.info(f"{len(lines)} examples")

data_lines = lines # at this point the list of the examples (as json dictionaries) has to be named "data_lines"

def setup_model(seed=42) -> nn.Module:
    """
    Setup the Multi Layer Perceptron for a given seed.

    Args:
      seed, the random seed for initializing the parameters of the MLP
    
    Returns a Multi Layer Perceptron Model.
    """

    # Constants
    input_size = 512 #number of dimensions of the image embedding
    num_classes = len(labels) #number of classes

    # Define a simple neural network
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define a custom weight initialization function
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He initialization
                nn.init.constant_(m.bias, 0)  # Bias initialized to 0

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

    # Initialize the neural network
    model = SimpleNN(input_size, num_classes)
    # Apply consistent weight initialization
    initialize_weights(model)

    return model

def train_mlp(model, X_train, y_train, X_test, y_test, num_epochs=1000):


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
        report = classification_report(
            y_test.cpu().numpy(),
            y_pred_classes.cpu().numpy(),
            target_names=labels,
            labels=list(range(len(labels))),
            output_dict=True
        )
        history["classification_report"].append(report)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    return model, best_epoch, history

##############################################################################################################

# Preprocessing the dataset for the model
logger.info("Processing dataset…")
data_lines_img = preprocess_dataset(data_lines, dataset_path)
#data_lines_img_biomed = preprocess_dataset_biomed(data_lines, dataset_path)

for name_clip, path_model in clips:
    if os.path.exists(f"analysis/{name_clip}_report.jsonl"):
        print(name_clip, "ignored because it was already processed")
        continue

    # Compute the image embeddings

    #- Path to store the embeddings
    try:
        embeds_path = os.path.join("embeds", f"images-{name_clip}-eval.pt")
        os.makedirs(os.path.dirname(embeds_path), exist_ok=True)

        if not os.path.exists(embeds_path): #the embeddings for the current CLIP model are not already stored in a file, let's compute them now and save them for later
            # Convert textual labels to numeric
            label_to_idx = {label: i for i, label in enumerate(labels)}
            idx_to_label = {v: k for k, v in label_to_idx.items()}
            labels = [find_label(line) for line in data_lines]
            labels_numeric = np.array([label_to_idx[label] for label in labels])

            # Get image embeddings
            logger.info(f"Loading CLIP model {name_clip}…")
            model = load_model(path_model)
            model.eval()

            logger.info("Making embeds…")
            if path_model == "biomedclip":
                image_embeds = make(model, data_lines_img_biomed)
            else:
                image_embeds = make(model, data_lines_img)

            X = torch.tensor(image_embeds, dtype=torch.float32)
            y = torch.tensor(labels_numeric, dtype=torch.long)

            # Save the embeddings in a file to easily load them later instead of recomputing them
            torch.save(X, embeds_path)
            torch.save(y, embeds_path+"-y.pt")
        else: #load the embeddings from the save files
            logger.info(f"Loading embeddings {name_clip}…")
            X = torch.load(embeds_path)
            y = torch.load(embeds_path+"-y.pt")

        # Only use 50% of the full dataset for training and evaluation
        subset_fraction = 0.5
        subset_size = int(subset_fraction * len(X))

        # Fix seed and shuffle indices
        seed(42)
        np.random.seed(42)
        indices = list(range(len(X)))
        shuffle(indices)
        subset_indices = indices[:subset_size]

        X = X[subset_indices]
        y = y[subset_indices]
        data_lines = [data_lines[i] for i in subset_indices if i < len(data_lines)]


        # Now split that 50% subset into training and testing sets (e.g., 80/20 split)
        train_size = int(train_rate * len(X))
        train_ids = list(range(train_size))
        test_ids = list(range(train_size, len(X)))

        X_train, X_test = X[train_ids], X[test_ids]
        y_train, y_test = y[train_ids], y[test_ids]
    
    except Exception as e:
        logger.error(f"Error processing {name_clip}: {e}")
        continue


    # Find the best MLP
    num_epochs = 1000

    logger.info(f"Training {name_clip}")
    best_model, best_accuracy, best_history, best_seed = None, 0, None, 0
    for model_seed in (42, 3, 314, 1602, 512): #5 initial settings for the MLP, take the best one according to validation accuracy
        model = setup_model(model_seed)
        new_model, best_epoch, history = train_mlp(model, X_train, y_train, X_test, y_test, num_epochs)
        final_accuracy = history["val_accuracy"][best_epoch]

        if best_model is None or final_accuracy > best_accuracy:
            best_model = new_model
            best_accuracy = final_accuracy
            best_history = history
            best_seed = model_seed

    logger.info(f"CLIP: {name_clip} Final accuracy: {best_accuracy}") #save the history for further analysis
    with open(f"analysis/{name_clip}_report.jsonl", "w") as f:
        for epoch in range(num_epochs):
            data_epoch = {k: v[epoch] for k, v in best_history.items()}
            f.write(json.dumps(data_epoch) + "\n")

    # Save the trained model (optional)
    if save_nn:
        os.makedirs("models", exist_ok=True)
        torch.save(best_model.state_dict(), f'models/{name_clip}-seed-{best_seed}.pth')

    # Load the model later with:
    # model.load_state_dict(torch.load(f'models/{name_clip}-seed-{best_seed}.pth'))
