import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader

try:
    from .Benchmark import Benchmark
except ImportError:
    from Benchmark import Benchmark


_PARAM_GRID = {
    "learning_rate_init": [0.1, 0.001, 0.005, 0.0005],
    "alpha": [0.1, 0.001, 0.01, 0.4],
}


def _dataset_to_numpy(dataset):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    all_X, all_y = [], []
    for X, y in loader:
        all_X.append(X.float().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_X), np.concatenate(all_y)


def _build_cv(y_train, requested_folds):
    n_samples = len(y_train)
    if n_samples < 2:
        raise ValueError(
            "MLP_eval needs at least 2 training examples for cross-validation"
        )

    requested_folds = min(requested_folds, n_samples)
    multilabel = y_train.ndim > 1

    if multilabel:
        actual_folds = max(2, requested_folds)
        if actual_folds != requested_folds:
            print(f"Adjusted MLP CV folds from {requested_folds} to {actual_folds}")
        return KFold(n_splits=actual_folds, shuffle=True, random_state=42)

    _, counts = np.unique(y_train, return_counts=True)
    min_class_count = int(counts.min()) if len(counts) else 0
    if min_class_count >= 2:
        actual_folds = min(requested_folds, min_class_count)
        if actual_folds != requested_folds:
            print(
                f"Adjusted MLP CV folds from {requested_folds} to {actual_folds} "
                f"because the smallest class has {min_class_count} examples"
            )
        return StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

    actual_folds = min(requested_folds, n_samples)
    print(
        "Using non-stratified MLP CV because at least one class has fewer "
        f"than 2 examples; folds={actual_folds}"
    )
    return KFold(n_splits=actual_folds, shuffle=True, random_state=42)


class MLP_eval(Benchmark):
    """Evaluates an image encoder by training an MLP classifier on frozen embeddings.

    Uses sklearn MLPClassifier with GridSearchCV for hyperparameter selection via
    k-fold cross-validation. Single-label tasks (1-D integer labels) are scored by
    macro F1; multi-label tasks (2-D binary label tensors, e.g. X-ray) are scored
    by micro F1.

    Args:
        output_dim: Number of output classes.
        training_set: Dataset of precomputed embeddings + labels for training.
        test_set: Dataset of precomputed embeddings + labels for evaluation.
        k: Number of CV folds for hyperparameter search (default: 10).
        n_epoch: Max training iterations for the MLP (default: 300).
        embedding_dim, loss, accuracy_function, iteration_number: Kept for
            backward compatibility, ignored.
    """

    def __init__(
        self,
        output_dim,
        training_set,
        test_set,
        k=10,
        embedding_dim=512,
        iteration_number=30,
        n_epoch=300,
        loss=None,
        accuracy_function=None,
    ):
        self.training_set = training_set
        self.test_set = test_set
        self.k = k
        self.n_epoch = n_epoch
        self.output_dim = output_dim

    def evaluate(self):
        """Return a dict of metrics. The ``score`` key holds the primary optimization metric."""
        X_train, y_train = _dataset_to_numpy(self.training_set)
        X_test, y_test = _dataset_to_numpy(self.test_set)

        multilabel = y_train.ndim > 1

        clf = MLPClassifier(
            hidden_layer_sizes=(512, 256),
            max_iter=self.n_epoch,
            random_state=42,
        )
        if multilabel:
            scorer = make_scorer(f1_score, average="micro", zero_division=0)
        else:
            scorer = make_scorer(
                f1_score,
                average="macro",
                labels=list(range(self.output_dim)),
                zero_division=0,
            )

        cv = _build_cv(y_train, self.k)
        grid = GridSearchCV(clf, _PARAM_GRID, cv=cv, scoring=scorer, n_jobs=-1)
        grid.fit(X_train, y_train)

        print(f"Best params: {grid.best_params_}")

        y_pred = grid.predict(X_test)
        if multilabel:
            micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            hamming_acc = 1.0 - float(hamming_loss(y_test, y_pred))
            metrics = {
                "score": micro_f1,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "hamming_accuracy": hamming_acc,
            }
        else:
            macro_f1 = f1_score(
                y_test,
                y_pred,
                average="macro",
                labels=list(range(self.output_dim)),
                zero_division=0,
            )
            bal_acc = float(balanced_accuracy_score(y_test, y_pred))
            acc = float(accuracy_score(y_test, y_pred))
            metrics = {
                "score": macro_f1,
                "macro_f1": macro_f1,
                "balanced_accuracy": bal_acc,
                "accuracy": acc,
            }

        for k, v in metrics.items():
            print(f"Test {k}: {v:.4f}")
        return metrics
