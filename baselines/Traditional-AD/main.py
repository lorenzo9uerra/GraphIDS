import os
import sys
import gc
from typing import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import wandb

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    classification_report,
)

from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from pyod.models.iforest import IForest


def find_optimal_threshold(y_true, y_scores, n_thresholds=500):
    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), num=n_thresholds)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_scores.append(f1)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1


# Check if command-line arguments are provided
if len(sys.argv) < 3:
    print("Error: Insufficient arguments provided.")
    print("Usage: python Anomal-E.py <dataset_name> <dataset_directory>")
    sys.exit(1)
dataset_name = sys.argv[1]
seed = 42

np.random.seed(seed)

file_name = os.path.join(sys.argv[2], dataset_name, f"{dataset_name}.csv")
wandb.init(
    project="Traditional-AD",
    config={
        "dataset_name": dataset_name,
        "seed": seed,
    },
)

data = pd.read_csv(file_name)
data.rename(columns=lambda x: x.strip(), inplace=True)

if "v2" in dataset_name:
    data.drop(columns=["L4_SRC_PORT", "L4_DST_PORT"], inplace=True)
elif "v3" in dataset_name:
    data.drop(
        columns=[
            "L4_SRC_PORT",
            "L4_DST_PORT",
            "FLOW_END_MILLISECONDS",
            "FLOW_START_MILLISECONDS",
        ],
        inplace=True,
    )

if "CSE-CIC-IDS2018" in dataset_name:
    data = data.groupby(by="Attack").sample(frac=0.2, random_state=seed)
X = data.drop(columns=["Attack", "Label"])
y = data[["Attack", "Label"]]
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

X_train, X_val_test, y_train, y_val_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

# NOTE: This estimator is stateless and does not need to be fitted (normalizes samples individually to unit norm).
# It is recommended to call fit_transform instead of transform, as parameter validation is only performed in fit.
scaler = Normalizer()
cols_to_norm = list(
    X_train.iloc[:, 2:].columns
)  # Ignore first two as the represents IP addresses

X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])
X_val_test[cols_to_norm] = scaler.fit_transform(X_val_test[cols_to_norm])

X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.5, random_state=seed, stratify=y_val_test
)

train = pd.concat([X_train, y_train], axis=1)
# Use only benign samples for training. No need to do it before normalization,
# because the normalization is done on each sample independently.
train = train[train["Label"] == 0]
y_train = train["Label"]
X_train = train[cols_to_norm].values
X_val = X_val[cols_to_norm].values
X_test = X_test[cols_to_norm].values
y_val = y_val["Label"]
y_test = y_test["Label"]

cblof_params = [2, 3, 5, 7, 9, 10]
hbos_params = [5, 10, 15, 20, 25, 30]
pca_params = [5, 10, 15, 20, 25, 30]
if_params = [20, 50, 100, 150]

print("\nNow testing on benign samples")
best_model_params = {}
score = -1
threshold = 0
best_param = None
best_threshold = None
# CBLOF
for param in (pbar := tqdm(cblof_params)):
    clf = CBLOF(n_clusters=param, random_state=seed)
    try:
        clf.fit(X_train)
    except (ValueError, RuntimeError) as e:
        print(f"Skipping param {param} due to: {e}")
        continue
    y_scores = clf.decision_function(X_val)
    pr_auc = average_precision_score(y_val, y_scores)
    threshold, _ = find_optimal_threshold(y_val, y_scores)

    if pr_auc > score:
        score = pr_auc
        best_param = param
        best_threshold = threshold
    pbar.set_postfix({"cur-pr_auc": pr_auc, "best-pr_auc": score})
    del clf
    gc.collect()

# Test CBLOF with best param on the test set
clf = CBLOF(n_clusters=best_param, random_state=seed)
clf.fit(X_train)
y_scores = clf.decision_function(X_test)
pr_auc = average_precision_score(y_test, y_scores)
y_pred = (y_scores >= best_threshold).astype(int)
f1 = f1_score(y_test, y_pred, average="macro")
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
os.makedirs("curves", exist_ok=True)
np.savez_compressed(
    f"curves/{dataset_name}_CBLOF_precision_recall.npz",
    precision=precision,
    recall=recall,
    thresholds=thresholds,
    y_scores=y_scores,
    y_true=y_test.values,
)
test_classification_report = classification_report(y_test, y_pred, zero_division=0)
print(f"CBLOF test set results: PR-AUC={pr_auc}, F1={f1}")
wandb.log({"CBLOF_test_PR_AUC": pr_auc, "CBLOF_test_F1": f1})
print(f"Test classification report:\n{test_classification_report}")
del clf
gc.collect()
score = -1
threshold = 0
best_param = None
best_threshold = None
# HBOS
for param in (pbar := tqdm(hbos_params)):
    clf = HBOS(n_bins=param)
    try:
        clf.fit(X_train)
    except (ValueError, RuntimeError) as e:
        print(f"Skipping param {param} due to: {e}")
        continue
    y_scores = clf.decision_function(X_val)
    pr_auc = average_precision_score(y_val, y_scores)
    threshold, _ = find_optimal_threshold(y_val, y_scores)

    if pr_auc > score:
        score = pr_auc
        best_param = param
        best_threshold = threshold
    pbar.set_postfix({"cur-pr_auc": pr_auc, "best-pr_auc": score})
    del clf
    gc.collect()

# Test HBOS with best param on the test set
clf = HBOS(n_bins=best_param)
clf.fit(X_train)
y_scores = clf.decision_function(X_test)
pr_auc = average_precision_score(y_test, y_scores)
y_pred = (y_scores >= best_threshold).astype(int)
f1 = f1_score(y_test, y_pred, average="macro")
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
os.makedirs("curves", exist_ok=True)
np.savez_compressed(
    f"curves/{dataset_name}_HBOS_precision_recall.npz",
    precision=precision,
    recall=recall,
    thresholds=thresholds,
    y_scores=y_scores,
    y_true=y_test.values,
)
test_classification_report = classification_report(y_test, y_pred, zero_division=0)
print(f"HBOS test set results: PR-AUC={pr_auc}, F1={f1}")
wandb.log({"HBOS_test_PR_AUC": pr_auc, "HBOS_test_F1": f1})
print(f"Test classification report:\n{test_classification_report}")
del clf
gc.collect()
score = -1
threshold = 0
best_param = None
best_threshold = None
# PCA
for param in (pbar := tqdm(pca_params)):
    clf = PCA(n_components=param)
    try:
        clf.fit(X_train)
    except (ValueError, RuntimeError) as e:
        print(f"Skipping param {param} due to: {e}")
        continue
    y_scores = clf.decision_function(X_val)
    pr_auc = average_precision_score(y_val, y_scores)
    threshold, _ = find_optimal_threshold(y_val, y_scores)

    if pr_auc > score:
        score = pr_auc
        best_param = param
        best_threshold = threshold
    pbar.set_postfix({"cur-pr_auc": pr_auc, "best-pr_auc": score})
    del clf
    gc.collect()
# Test PCA with best param on the test set
clf = PCA(n_components=best_param)
clf.fit(X_train)
y_scores = clf.decision_function(X_test)
pr_auc = average_precision_score(y_test, y_scores)
y_pred = (y_scores >= best_threshold).astype(int)
f1 = f1_score(y_test, y_pred, average="macro")
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
os.makedirs("curves", exist_ok=True)
np.savez_compressed(
    f"curves/{dataset_name}_PCA_precision_recall.npz",
    precision=precision,
    recall=recall,
    thresholds=thresholds,
    y_scores=y_scores,
    y_true=y_test.values,
)
test_classification_report = classification_report(y_test, y_pred, zero_division=0)
print(f"PCA test set results: PR-AUC={pr_auc}, F1={f1}")
wandb.log({"PCA_test_PR_AUC": pr_auc, "PCA_test_F1": f1})
print(f"Test classification report:\n{test_classification_report}")
del clf
gc.collect()
score = -1
threshold = 0
best_param = None
best_threshold = None
# Isolation Forest
for param in (pbar := tqdm(if_params)):
    clf = IForest(n_estimators=param, random_state=seed)
    try:
        clf.fit(X_train)
    except (ValueError, RuntimeError) as e:
        print(f"Skipping param {param} due to: {e}")
        continue
    y_scores = clf.decision_function(X_val)
    pr_auc = average_precision_score(y_val, y_scores)
    threshold, _ = find_optimal_threshold(y_val, y_scores)

    if pr_auc > score:
        score = pr_auc
        best_param = param
        best_threshold = threshold
    pbar.set_postfix({"cur-pr_auc": pr_auc, "best-pr_auc": score})
    del clf
    gc.collect()
# Test Isolation Forest with best param on the test set
clf = IForest(n_estimators=best_param, random_state=seed)
clf.fit(X_train)
y_scores = clf.decision_function(X_test)
pr_auc = average_precision_score(y_test, y_scores)
y_pred = (y_scores >= best_threshold).astype(int)
f1 = f1_score(y_test, y_pred, average="macro")
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
os.makedirs("curves", exist_ok=True)
np.savez_compressed(
    f"curves/{dataset_name}_IF_precision_recall.npz",
    precision=precision,
    recall=recall,
    thresholds=thresholds,
    y_scores=y_scores,
    y_true=y_test.values,
)
test_classification_report = classification_report(y_test, y_pred, zero_division=0)
print(f"Isolation Forest test set results: PR-AUC={pr_auc}, F1={f1}")
wandb.log({"IF_test_PR_AUC": pr_auc, "IF_test_F1": f1})
print(f"Test classification report:\n{test_classification_report}")
del clf
gc.collect()
wandb.finish()
