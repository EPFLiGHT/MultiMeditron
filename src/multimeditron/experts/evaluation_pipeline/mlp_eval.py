from Benchmark import Benchmark
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np

# Simple MLP classifier used on top of precomputed embeddings
class MLP_Classifier(torch.nn.Module):

    FIRST_DIM = 512
    SECOND_DIM = 256

    def __init__(self,  output_dim:int, input_dim=512):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.FIRST_DIM),
            nn.ReLU(),
            torch.nn.Linear(self.FIRST_DIM, self.SECOND_DIM),
            nn.ReLU(),
            torch.nn.Linear(self.SECOND_DIM, self.output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Benchmark that evaluates an image encoder by training and evaluating a multi-layer perceptron on its embeddings
#takes the training and testing sets of the embeddings as arguments
class MLP_eval(Benchmark):

    def __init__(self, 
                output_dim: int, 
                training_set: Dataset, 
                test_set: Dataset,
                k=10, 
                embedding_dim=512,
                iteration_number=30,
                n_epoch=100,
                loss=nn.CrossEntropyLoss(),
                accuracy_function=lambda preds, labels: (preds == labels).float().mean().item(),
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_set = training_set
        self.test_set = test_set
        self.test_loader= DataLoader(dataset=test_set, batch_size=512)
        self.train_loader= DataLoader(dataset=training_set, batch_size=512)

        self.k = k
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.kfold = KFold(n_splits=self.k, shuffle=True, random_state=42)
        self.n_epoch=n_epoch
        self.iteration_number=iteration_number
        self.loss = loss
        self.accuracy_function = accuracy_function
    
    #evaluates the model on the test dataset
    def evaluate_fold(self, model, test_loader):
        model.eval()
        total = 0
        correct = 0

        for x, label in test_loader:
            x = x.to("cuda")
            label = label.to("cuda")
            output = torch.sigmoid(model(x))
            preds = torch.argmax(output, dim=1)

            batch_acc = self.accuracy_function(preds, label)
            correct += batch_acc * len(label)
            total += len(label)

        return correct / total

    #returns a MLP classifier trained on a train dataset
    def training(self,data_loader, lr, wd):
            model = MLP_Classifier(self.output_dim, input_dim=self.embedding_dim).to(self.device)
            criterion = self.loss.to("cuda")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            losses = []

            model.train()
            for epoch in tqdm(range(self.n_epoch)):
                epoch_loss = 0.0
                for i, data in enumerate(data_loader):
                    inputs, lab = data
                    inputs = inputs.to("cuda")
                    lab = lab.to("cuda")
                    optimizer.zero_grad()
                    y = model(inputs).to("cuda")

                    l = criterion(y, lab.to("cuda"))
                    l.backward()
                
                    optimizer.step()
                    epoch_loss += l.item()

                average_loss = epoch_loss / len(data_loader)
                losses.append(average_loss)

            return average_loss, model

    #returns the accuracy of the k fold cross validation for a given lr and wd
    def k_fold_training(self,lr, wd):
        indices = np.arange(len(self.training_set))
        accuracy = 0

        for fold, (train_ids, val_ids) in enumerate(self.kfold.split(indices)):
            train_subset = Subset(self.training_set, train_ids)
            val_subset = Subset(self.training_set, val_ids)

            train_loader = DataLoader(train_subset, batch_size=512)
            val_loader = DataLoader(val_subset, batch_size=512)

            _, classifier = self.training(train_loader, lr, wd)
            accuracy += self.evaluate_fold(classifier, val_loader)
        
        return accuracy / self.k

    def evaluate(self) -> float:
        learning_rates = [0.1, 0.8, 0.001, 0.005, 0.0005]
        weight_decays = [0.1, 0.001, 0.01, 0.4] 
        best_result = -1
        best_lr = 0
        best_wd = 0
        for lr in learning_rates:
            for wd in weight_decays:
                kfold_result = self.k_fold_training(lr, wd)
                if kfold_result > best_result:
                    best_lr = lr
                    best_wd = wd
                    best_result=kfold_result
        print("best results with lr: " + str(best_lr)+ " and wd: " + str(best_wd))
        _, best_classifier = self.training(self.train_loader, best_lr, best_wd)
        final_result = self.evaluate_fold(best_classifier, self.test_loader)
        print("test value : " + str(final_result))
        return final_result