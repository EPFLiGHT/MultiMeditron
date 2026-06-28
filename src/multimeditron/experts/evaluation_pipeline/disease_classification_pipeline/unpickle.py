"""
Optuna Study Inspection Utility.

This script loads a previously saved Optuna study from disk and prints the best trial
value, corresponding hyperparameters, and a summary of all completed trials. It is
intended for quick inspection and reporting of hyperparameter search results.
"""

import pickle

with open("study_skin_1.pkl", "rb") as f:  # adjust number
    study = pickle.load(f)

print("Best value (geom. mean of benchmark):", study.best_value)
print("Best params:", study.best_params)

print("\nAll trials:")
for t in study.trials:
    print(f"Trial {t.number}: value={t.value}, params={t.params}")
