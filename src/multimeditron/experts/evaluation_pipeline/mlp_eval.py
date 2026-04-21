try:
    from .Benchmark import Benchmark
except ImportError:
    from Benchmark import Benchmark
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np

class MLP_Classifier(torch.nn.Module):
    """3-layer MLP classifier trained on top of precomputed image embeddings.

    Architecture: input_dim → 512 → 256 → output_dim, with ReLU activations.
    """

    FIRST_DIM = 512
    SECOND_DIM = 256

    def __init__(self, output_dim: int, input_dim=512):
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

class MLP_eval(Benchmark):
    """Evaluates an image encoder by training an MLP classifier on its frozen embeddings.

    Protocol:
    1. Grid-search over learning rates and weight decays using k-fold cross-validation
       on the training set to select the best hyperparameters.
    2. Retrain the MLP on the full training set with the best hyperparameters.
    3. Report accuracy on the held-out test set.

    Args:
        output_dim: Number of classes.
        training_set: Dataset of precomputed embeddings + labels for training.
        test_set: Dataset of precomputed embeddings + labels for evaluation.
        k: Number of folds for cross-validation (default: 10).
        embedding_dim: Dimensionality of input embeddings (default: 512).
        iteration_number: Unused, kept for API compatibility.
        n_epoch: Number of training epochs per fold (default: 100).
        loss: Loss function (default: CrossEntropyLoss). Use a weighted variant
            for imbalanced datasets.
        accuracy_function: Callable(preds, labels) → float. Defaults to top-1
            accuracy. Override for multi-label tasks (e.g. F1).
    """

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
            x = x.to(self.device).float()
            label = label.to(self.device)
            logits = model(x)

            # Multi-label datasets provide float multi-hot targets. In that case,
            # the accuracy function should consume probabilities directly.
            if label.ndim > 1:
                batch_acc = self.accuracy_function(torch.sigmoid(logits), label)
            else:
                preds = torch.argmax(logits, dim=1)
                batch_acc = self.accuracy_function(preds, label)
            correct += batch_acc * len(label)
            total += len(label)

        return correct / total

    #returns a MLP classifier trained on a train dataset
    def training(self,data_loader, lr, wd):
            model = MLP_Classifier(self.output_dim, input_dim=self.embedding_dim).to(self.device)
            criterion = self.loss.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            losses = []

            model.train()
            for epoch in tqdm(range(self.n_epoch)):
                epoch_loss = 0.0
                for i, data in enumerate(data_loader):
                    inputs, lab = data
                    inputs = inputs.to(self.device).float()
                    lab = lab.to(self.device)
                    optimizer.zero_grad()
                    y = model(inputs).to(self.device)

                    l = criterion(y, lab.to(self.device))
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