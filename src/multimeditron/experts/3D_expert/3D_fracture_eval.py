import torch
import json 
from load_from_clip import load_model, encode_img
import torch.nn as nn
from torch.utils.data import Dataset
from Benchmark import Benchmark
import os
from tqdm import tqdm

from mlp_eval import MLP_eval
import sys

class Fracture(Dataset):

    def __init__(self, evaluated_clip, saving_name, path_list):

        self.data = []
        self.label = []      
        self.evaluated_clip = evaluated_clip
        current_dir = os.getcwd()
        file_name_data = current_dir + "/embeddings/data_" + saving_name + ".pt"
        file_name_lab = current_dir + "/embeddings/lab_" + saving_name + ".pt"

        if os.path.exists(file_name_data):
                self.data = torch.load(file_name_data)
                self.label = torch.load(file_name_lab)
        else:
            for path, frac in path_list:
                with open(path) as f:
                    lines = f.readlines()
                    
                    for line in tqdm(lines):
                            row = json.loads(line)
                            # The dataset contains relative paths like "data/ct_quizze_XX/..."
                            # Convert to absolute path based on current directory
                            image_path = os.path.join(current_dir, row["modalities"][0]["value"])
                            try:
                                image_encoded = encode_img(self.evaluated_clip, image_path).detach().cpu()
                            except FileNotFoundError as e:
                                # Skip entries with missing data files
                                print(f"Skipping: {e}")
                                continue

                            self.data.append(image_encoded)
                            if frac:
                                lab = torch.tensor([1])
                            else:
                                lab= torch.tensor([0])
                            self.label.append(lab)
            torch.save(self.data, file_name_data)
            torch.save(self.label, file_name_lab)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].squeeze(), self.label[idx].squeeze()

class Fracture_benchmark(Benchmark):
    def __init__(self, clip_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_path = clip_path
        self.clip_name = ""
    def evaluate(self):
        EMBEDDING_DIM: int = 768  # Model hidden_size from config
        l = self.clip_path.split('/')
        l.reverse()
        clip_name = l[0]
        self.clip_name=clip_name
        evaluated_clip = load_model(self.clip_path, device=self.device)
        evaluated_clip = evaluated_clip.to(self.device)

        evaluated_clip.eval()
        tr_datasets = [("/mloscratch/users/deschryv/clipFineTune/3D_dataset/fracture_train_file.jsonl", True), 
                       ("/mloscratch/users/deschryv/clipFineTune/3D_dataset/no_fracture_train_file.jsonl", False)]
    
        test_datasets = [("/mloscratch/users/deschryv/clipFineTune/3D_dataset/fracture_test_file.jsonl", True), 
                       ("/mloscratch/users/deschryv/clipFineTune/3D_dataset/no_fracture_test_file.jsonl", False)]
        train_dataset = Fracture(evaluated_clip,self.clip_name,tr_datasets)
        test_dataset =Fracture(evaluated_clip,self.clip_name,test_datasets)
        mlp_eval = MLP_eval(2, train_dataset, test_dataset, embedding_dim=EMBEDDING_DIM)
        mlp_eval.evaluate()

if __name__ == "__main__":
    xray_bench = Fracture_benchmark(sys.argv[1])
    xray_bench.evaluate()