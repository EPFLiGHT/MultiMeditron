from Benchmark import Benchmark
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np

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

class MLP_eval(Benchmark):

    def __init__(self, 
                output_dim: int, 
                training_set: Dataset, 
                test_set: Dataset,
                k=10, 
                embedding_dim=512,
                iteration_number=30,
                n_epoch=100,
                loss=None,  # Will be computed with class weights
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
        
        # Compute class weights for handling class imbalance
        self.class_weights = self._compute_class_weights()
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.accuracy_function = accuracy_function

    def _compute_class_weights(self):
        """Compute class weights inversely proportional to class frequencies."""
        labels = []
        for _, label in self.training_set:
            labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        labels = np.array(labels)
        
        class_counts = np.bincount(labels, minlength=self.output_dim)
        total = len(labels)
        
        # Inverse frequency weighting
        class_weights = total / (self.output_dim * class_counts + 1e-6)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"\nClass distribution in training set:")
        for i, count in enumerate(class_counts):
            print(f"  Class {i}: {count} samples ({100*count/total:.1f}%)")
        print(f"Class weights: {class_weights.cpu().numpy()}")
        
        return class_weights

    def evaluate_fold(self, model, test_loader, detailed=False):
        #evaluates the model on the test dataset
        model.eval()
        all_preds = []
        all_labels = []

        for x, label in test_loader:
            x = x.to("cuda")
            label = label.to("cuda")
            output = torch.sigmoid(model(x))
            preds = torch.argmax(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute balanced accuracy (handles class imbalance)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        
        if detailed:
            # Compute additional metrics for final evaluation
            acc = (all_preds == all_labels).mean()
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            cm = confusion_matrix(all_labels, all_preds)
            
            print(f"\n{'='*50}")
            print("DETAILED EVALUATION METRICS")
            print(f"{'='*50}")
            print(f"Accuracy:          {acc*100:.2f}%")
            print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
            print(f"Precision:         {precision*100:.2f}%")
            print(f"Recall:            {recall*100:.2f}%")
            print(f"F1-Score:          {f1*100:.2f}%")
            print(f"\nConfusion Matrix:")
            print(f"                 Predicted")
            print(f"              Neg      Pos")
            print(f"Actual Neg   {cm[0,0]:5d}    {cm[0,1]:5d}")
            print(f"Actual Pos   {cm[1,0]:5d}    {cm[1,1]:5d}")
            print(f"{'='*50}\n")
        
        return bal_acc
    
    def training(self,data_loader, lr, wd):
        #returns a MLP classifier trained on a train dataset
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

                    l = criterion(y, lab.long().to("cuda"))
                    l.backward()
                
                    optimizer.step()
                    epoch_loss += l.item()

                average_loss = epoch_loss / len(data_loader)
                losses.append(average_loss)

            return average_loss, model

    def k_fold_training(self,lr, wd):
    #returns the accuracy of the k fold cross validation for a given lr and wd
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
        
        print("\n" + "="*50)
        print("HYPERPARAMETER SEARCH (K-FOLD CROSS VALIDATION)")
        print("="*50)
        
        for lr in learning_rates:
            for wd in weight_decays:
                kfold_result = self.k_fold_training(lr, wd)
                print(f"  LR={lr}, WD={wd}: Balanced Accuracy = {kfold_result*100:.2f}%")
                if kfold_result > best_result:
                    best_lr = lr
                    best_wd = wd
                    best_result=kfold_result
        
        print("\n" + "="*50)
        print("BEST HYPERPARAMETERS")
        print("="*50)
        print(f"  Learning Rate:  {best_lr}")
        print(f"  Weight Decay:   {best_wd}")
        print(f"  CV Balanced Accuracy: {best_result*100:.2f}%")
        print("="*50)
        
        print("\nTraining final model with best hyperparameters...")
        _, best_classifier = self.training(self.train_loader, best_lr, best_wd)
        
        print("\nEvaluating on test set...")
        final_accuracy = self.evaluate_fold(best_classifier, self.test_loader, detailed=True)
        
        print(f"Final Test Balanced Accuracy: {final_accuracy*100:.2f}%")
        return final_accuracy