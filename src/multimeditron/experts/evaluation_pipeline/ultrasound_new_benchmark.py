import torch
import torch.nn as nn
from load_from_clip import load_model, encode_img
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from Benchmark import Benchmark
import numpy as np
import os
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    VisionTextDualEncoderConfig
)
from transformers import CLIPModel, CLIPProcessor
from mlp_eval import MLP_eval
from tqdm import tqdm
import sys

# Integer labels used for anatomical region classification
BREAST  = 0
OTHER = 1
ABDOMEN = 2
TYROID = 3

# Loads a JSONL file and encodes all images using the provided model with the provided label
def load_dataset(path : str, label: int, image_path: str, model):
    X_train = []
    Y_train = []
    with open(path, "r", encoding="utf-8") as file:
        for f in tqdm(file):
            Y_train.append(label)

            correct_line = json.loads(f) 
            new_path = image_path + correct_line["modalities"][0]['value']

            X_train.append(encode_img(model, new_path))
    return (torch.stack(X_train), torch.tensor(Y_train))

# PyTorch Dataset for anatomical classification (training split)
# Aggregates multiple ultrasound datasets into a single multi-class dataset
class BodyPartsDataset(Dataset):
    
    def __init__(self, model, model_name, load, path):
        path = "/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/"
        if not load:
            BUSI = (load_dataset(path+"classifier-breast-radiopedia-final_train.jsonl", BREAST, "", model))
            CAMUS = (load_dataset(path+"classifier-heart-radiopedia-final_train.jsonl", OTHER, "", model))
            COVIDUS = load_dataset(path+"classifier-lungs-radiopedia-final_train.jsonl", OTHER, "", model)
            CT2 = (load_dataset(path+"classifier-abdomen-radiopedia-final_train.jsonl", ABDOMEN, "", model))
            DDTI = (load_dataset(path+"classifier-thyroid-radiopedia-final_train.jsonl", TYROID, "", model))

            self.data = torch.cat([BUSI[0], CAMUS[0], COVIDUS[0], CT2[0], DDTI[0]], dim=0)
            self.labels = torch.cat([BUSI[1], CAMUS[1], COVIDUS[1], CT2[1], DDTI[1]], dim=0)

            torch.save(self.data, "data_embl_" + model_name + ".pt")
            torch.save(self.labels, "data_lab_" + model_name + ".pt")
        else:
            self.data = torch.load(path + "/data_embl_"+ model_name +".pt")
            self.labels = torch.load(path + "/data_embl_"+ model_name + ".pt")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# PyTorch Dataset for anatomical classification (test split)
# Same logic as training dataset but uses test JSON files
class BodyPartsDatasetTEST(Dataset):
    
    def __init__(self, model, model_name, load, save_path):
        path = "/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/"
        if not load:
            BUSI = (load_dataset(path+"classifier-breast-radiopedia-final_test.jsonl", BREAST, "", model))
            CAMUS = (load_dataset(path+"classifier-heart-radiopedia-final_test.jsonl", OTHER, "", model))
            COVIDUS = load_dataset(path+"classifier-lungs-radiopedia-final.jsonl", OTHER, "", model)
            CT2 = (load_dataset(path+"classifier-abdomen-radiopedia-final_test.jsonl", ABDOMEN, "", model))
            DDTI = (load_dataset(path+"classifier-thyroid-radiopedia-final_test.jsonl", TYROID, "", model))

            self.data = torch.cat([BUSI[0], CAMUS[0], COVIDUS[0], CT2[0], DDTI[0]], dim=0)
            self.labels = torch.cat([BUSI[1], CAMUS[1], COVIDUS[1], CT2[1], DDTI[1]], dim=0)

            torch.save(self.data, save_path + "/data_embl_test_"+ model_name + ".pt")
            torch.save(self.labels, save_path + "/data_lab_test_" + model_name + ".pt")
        else:
            self.data = torch.load(save_path + "/data_embl_test_"+ model_name +".pt")
            self.labels = torch.load(save_path + "/data_lab_test_"+ model_name +".pt")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Full evaluation pipeline:
# - Builds datasets
# - Computes class weights for imbalance
# - Runs an MLP benchmark on top of CLIP embeddings
def evaluate_pipeline(model, model_name):
    device = "cuda"
    model = model.to(device)
    print("beginnig of the evaluation")
    train_dataset = BodyPartsDataset(model, model_name, False, "/mloscratch/users/deschryv/clipFineTune/embeddings")
    print("traning dataset loaded")
    data_loader = DataLoader(dataset=train_dataset, batch_size=512)
    test_dataset = BodyPartsDatasetTEST(model, model_name, False, "/mloscratch/users/deschryv/clipFineTune/embeddings")
    print("test dataset loaded")
    print("start of the mlp evaluation")
    labelsWEIGHT = np.array(train_dataset.labels)
    labelsWEIGHT = np.unique(labelsWEIGHT.astype(int))

    class_weights = compute_class_weight(class_weight='balanced', classes=labelsWEIGHT, y=np.array(train_dataset.labels))
    weights = torch.tensor(class_weights, dtype=torch.float)

    mlp_bench = MLP_eval(output_dim=4, training_set=train_dataset, test_set=test_dataset, loss=nn.CrossEntropyLoss(weight=weights))
    return mlp_bench.evaluate()

# Benchmark for an ultrasound image encoder. 
#Evaluates the model's ability to classify different images with a multi layer perceptron head into four categories: BREAST, OTHER, ABDOMEN, TYROID
class Anatomical_benchmark(Benchmark):

    def evaluate(self, model_path):

        return evaluate_pipeline(VisionTextDualEncoderModel.from_pretrained(model_path), "test")

if __name__ == "__main__":
    bench = Anatomical_benchmark()
    model_path = sys.argv[1]
    bench.evaluate(model_path)