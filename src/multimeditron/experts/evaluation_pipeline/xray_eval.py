import torch
import csv 
from load_from_clip import load_model, encode_img
import torch.nn as nn
from torch.utils.data import Dataset
from Benchmark import Benchmark
import os
from tqdm import tqdm
from transformers import VisionTextDualEncoderModel
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderConfig
)
from mlp_eval import MLP_eval
import sys
import kagglehub
from load_from_clip import load_model

def randomize_csv(input_path, seed=None):
    #function used to shuffle the dataset before using it for the benchmark

    df = pd.read_csv(input_path)
    
    df_randomized = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_randomized{ext}"
    
    df_randomized.to_csv(output_path, index=False)
    
    print(f"Nouveau fichier créé : {output_path}")


def download_data():
    # Download the latest version of the NIH dataset used in this benchmark, the dataset still needs to be shuffled with the function randomize_csv 
    path = kagglehub.dataset_download("nih-chest-xrays/data")

    print("Path to dataset files:", path)


def multi_label_single_match_acc(preds, labels):
    labels_indices = torch.argmax(labels, dim=1)
    return (preds == labels_indices).float().mean().item()

def load_clip(model_id, is_lion_model):
    if is_lion_model:
        config = VisionTextDualEncoderConfig.from_pretrained(model_id)
        config.vision_config.hidden_act = "gelu"
        my_clip = OpenCLIPVisionTextDualEncoderModel.from_pretrained(model_id, config=config)
    else:
        my_clip = load_model(model_id)
        
    return my_clip

class Xray_Dataset(Dataset):

    def get_label(self, finding_labels):
        label_list = finding_labels.split('|')
        output = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        labels = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia","Pleural_Thickening","Cardiomegaly","Nodule" ,"Mass","Hernia","No Finding"]
        for i in range(len(labels)):
            if labels[i] in label_list:
                output[i] = 1
        if output == [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]:
            print("probleme la oh ")

        return torch.FloatTensor(output)

    def __init__(self, evaluated_clip, saving_name, beginning, end, csv_path):

        self.data = []#contient le path des images
        self.label = []      
        self.evaluated_clip = evaluated_clip
        i = 0
        current_dir = os.getcwd()
        
        with open(csv_path) as f:
            csv_file = csv.DictReader(f)
            csv_file = list(csv_file)
            
            file_name_data = current_dir + "/embeddings/data_" + saving_name + ".pt"
            file_name_lab = current_dir + "/embeddings/lab_" + saving_name + ".pt"

            if os.path.exists(file_name_data):
                self.data = torch.load(file_name_data)
                self.label = torch.load(file_name_lab)
            else:
                for row in tqdm(csv_file):
                    if i >= beginning and i <= end and i > 0: 
                        self.data.append(encode_img(self.evaluated_clip,os.path.join(current_dir, "images",row['Image Index'])))
                        self.label.append(self.get_label(row["Finding_Labels"]))
                    i += 1     

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return  self.data[idx], self.label[idx]


def eval_pipeline(clip_path, clip_name, sample_number, csv, device):

    evaluated_clip = VisionTextDualEncoderModel.from_pretrained(clip_path).to(device)
    evaluated_clip.eval()
    upper_bound = (int)(sample_number*0.8)
    train_dataset = Xray_Dataset(evaluated_clip, clip_name+"_train", 0, upper_bound, csv)
    test_dataset = Xray_Dataset(evaluated_clip, clip_name+"_test", upper_bound, sample_number, csv)
    mlp_eval = MLP_eval(15, train_dataset, test_dataset, loss=nn.BCEWithLogitsLoss(), accuracy_function=multi_label_single_match_acc)
    mlp_eval.evaluate()

class XRay_benchmark(Benchmark):
    
    def __init__(self, is_lion_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_path = "Data_Entry_2017_randomized.csv"
        self.is_lion_model=is_lion_model
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            self.sample_number = sum(1 for line in f)
    
    def evaluate(self, clip_path):
        l = clip_path.split('/')
        l.reverse()
        clip_name = l[0]
        self.clip_name=clip_name
        eval_pipeline(clip_path, clip_name, self.sample_number, self.csv_path, self.device)

if __name__ == "__main__":
    is_lion_model = sys.argv[2]
    xray_bench = XRay_benchmark(lion_model)
    xray_bench.evaluate(sys.argv[1])