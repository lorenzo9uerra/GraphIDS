import os
import sys
import gc
import math
import time
from typing import *
from tqdm import tqdm

import pandas as pd
import numpy as np
import networkx as nx
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

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
save_curve = False

np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

file_name = os.path.join(sys.argv[2], dataset_name, f"{dataset_name}.csv")
run = wandb.init(
    project="Anomal-E",
    config={
        "dataset_name": dataset_name,
        "seed": seed,
    },
)

data = pd.read_csv(file_name)

data.rename(columns=lambda x: x.strip(), inplace=True)
data["IPV4_SRC_ADDR"] = data["IPV4_SRC_ADDR"].apply(str)
data["L4_SRC_PORT"] = data["L4_SRC_PORT"].apply(str)
data["IPV4_DST_ADDR"] = data["IPV4_DST_ADDR"].apply(str)
data["L4_DST_PORT"] = data["L4_DST_PORT"].apply(str)


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
# This datasets needs to be undersampled because it produces a graph too big for the original Anomal-E (>40GB VRAM)
if "CSE-CIC-IDS2018" in dataset_name:
    data = data.groupby(by="Attack").sample(frac=0.2, random_state=seed)
X = data.drop(columns=["Attack", "Label"])
y = data[["Attack", "Label"]]
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

X_train, X_val_test, y_train, y_val_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)


# Do not use TargetEncoder, as it is a supervised approach and could cause data leakage

# encoder = ce.TargetEncoder(cols=['TCP_FLAGS','L7_PROTO','PROTOCOL',
#                                   'CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS','ICMP_TYPE',
#                                   'ICMP_IPV4_TYPE','DNS_QUERY_ID','DNS_QUERY_TYPE',
#                                   'FTP_COMMAND_RET_CODE'])
# encoder.fit(X_train, y_train.Label)

# # Transform on training set
# X_train = encoder.transform(X_train)

# # Transform on testing set
# X_test = encoder.transform(X_test)

# NOTE: This estimator is stateless and does not need to be fitted (normalizes samples individually to unit norm).
# It is recommended to call fit_transform instead of transform, as parameter validation is only performed in fit.
scaler = Normalizer()
cols_to_norm = list(
    X_train.iloc[:, 2:].columns
)  # Ignore first two as the represents IP addresses

# Transform on training set
X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])
X_train["h"] = X_train.iloc[:, 2:].values.tolist()

# Transform on testing set
X_val_test[cols_to_norm] = scaler.fit_transform(X_val_test[cols_to_norm])
X_val_test["h"] = X_val_test.iloc[:, 2:].values.tolist()

X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.5, random_state=seed, stratify=y_val_test
)

train = pd.concat([X_train, y_train], axis=1)
# Use only benign samples for training. No need to do it before normalization,
# because the normalization is done on each sample independently.
train = train[train["Label"] == 0]
val = pd.concat([X_val, y_val], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Training graph
# removing the "Attack" columns, as it is not used in the model
train_g = nx.from_pandas_edgelist(
    train,
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    ["h", "Label"],
    create_using=nx.MultiGraph(),
)

train_g = train_g.to_directed()
train_g = dgl.from_networkx(train_g, edge_attrs=["h", "Label"])
nfeat_weight = torch.ones([train_g.number_of_nodes(), train_g.edata["h"].shape[1]])
train_g.ndata["h"] = nfeat_weight

# Validation graph
val_g = nx.from_pandas_edgelist(
    val, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ["h", "Label"], create_using=nx.MultiGraph()
)

val_g = val_g.to_directed()
val_g = dgl.from_networkx(val_g, edge_attrs=["h", "Label"])
nfeat_weight = torch.ones([val_g.number_of_nodes(), val_g.edata["h"].shape[1]])
val_g.ndata["h"] = nfeat_weight

# Testing graph
test_g = nx.from_pandas_edgelist(
    test, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ["h", "Label"], create_using=nx.MultiGraph()
)

test_g = test_g.to_directed()
test_g = dgl.from_networkx(test_g, edge_attrs=["h", "Label"])
nfeat_weight = torch.ones([test_g.number_of_nodes(), test_g.edata["h"].shape[1]])
test_g.ndata["h"] = nfeat_weight


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out):
        super(SAGELayer, self).__init__()
        self.W_apply = nn.Linear(ndim_in + edims, ndim_out)
        self.W_edge = nn.Linear(128 * 2, 256)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W_apply.weight, gain=gain)

    def message_func(self, edges):
        return {"m": edges.data["h"]}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata["h"] = nfeats
            g.edata["h"] = efeats
            g.update_all(self.message_func, fn.mean("m", "h_neigh"))
            g.ndata["h"] = F.relu(
                self.W_apply(torch.cat([g.ndata["h"], g.ndata["h_neigh"]], 2))
            )

            # Compute edge embeddings
            u, v = g.edges()
            edge = self.W_edge(torch.cat((g.srcdata["h"][u], g.dstdata["h"][v]), 2))
            return g.ndata["h"], edge


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128))

    def forward(self, g, nfeats, efeats, corrupt=False):
        if corrupt:
            e_perm = torch.randperm(g.number_of_edges())
            # n_perm = torch.randperm(g.number_of_nodes())
            efeats = efeats[e_perm]
            # nfeats = nfeats[n_perm]
        for i, layer in enumerate(self.layers):
            # nfeats = layer(g, nfeats, efeats)
            nfeats, e_feats = layer(g, nfeats, efeats)
        # return nfeats.sum(1)
        return nfeats.sum(1), e_feats.sum(1)


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DGI(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim):
        super(DGI, self).__init__()
        self.encoder = SAGE(ndim_in, ndim_out, edim)
        # self.discriminator = Discriminator(128)
        self.discriminator = Discriminator(256)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, n_features, e_features):
        positive = self.encoder(g, n_features, e_features, corrupt=False)
        negative = self.encoder(g, n_features, e_features, corrupt=True)

        positive = positive[1]
        negative = negative[1]

        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2


ndim_in = train_g.ndata["h"].shape[1]
hidden_features = 128
ndim_out = 128
num_layers = 1
edim = train_g.edata["h"].shape[1]
learning_rate = 1e-3
epochs = 2000  # unnecessary to increase the number of epochs further

dgi = DGI(ndim_in, ndim_out, edim).to("cuda")

dgi_optimizer = torch.optim.Adam(dgi.parameters(), lr=1e-3, weight_decay=0.0)

# Format node and edge features for E-GraphSAGE
train_g.ndata["h"] = torch.reshape(
    train_g.ndata["h"], (train_g.ndata["h"].shape[0], 1, train_g.ndata["h"].shape[1])
)
train_g.edata["h"] = torch.reshape(
    train_g.edata["h"], (train_g.edata["h"].shape[0], 1, train_g.edata["h"].shape[1])
)

# Convert to GPU
train_g = train_g.to("cuda")

cnt_wait = 0
best = 1e9
best_t = 0
dur = []
node_features = train_g.ndata["h"]
edge_features = train_g.edata["h"]

os.makedirs("checkpoints", exist_ok=True)
checkpoint = os.path.join("checkpoints", f"best_dgi_{dataset_name}.pth")
for epoch in range(epochs):
    dgi.train()
    if epoch >= 3:
        t0 = time.time()

    dgi_optimizer.zero_grad()
    loss = dgi(train_g, node_features, edge_features)
    loss.backward()
    dgi_optimizer.step()

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(dgi.state_dict(), checkpoint)
    else:
        cnt_wait += 1

    # if cnt_wait == patience:
    #     print('Early stopping!')
    #     break

    if epoch >= 3:
        dur.append(time.time() - t0)

    if epoch % 50 == 0 and epoch >= 3:
        wandb.log({"epoch": epoch, "loss": loss.item()})


dgi.load_state_dict(torch.load(checkpoint))


training_emb = (
    dgi.encoder(train_g, train_g.ndata["h"], train_g.edata["h"])[1]
    .detach()
    .cpu()
    .numpy()
)

val_g.ndata["h"] = torch.reshape(
    val_g.ndata["h"], (val_g.ndata["h"].shape[0], 1, val_g.ndata["h"].shape[1])
)
val_g.edata["h"] = torch.reshape(
    val_g.edata["h"], (val_g.edata["h"].shape[0], 1, val_g.edata["h"].shape[1])
)
test_g.ndata["h"] = torch.reshape(
    test_g.ndata["h"], (test_g.ndata["h"].shape[0], 1, test_g.ndata["h"].shape[1])
)
test_g.edata["h"] = torch.reshape(
    test_g.edata["h"], (test_g.edata["h"].shape[0], 1, test_g.edata["h"].shape[1])
)

# Convert to GPU
val_g = val_g.to("cuda")
test_g = test_g.to("cuda")

validation_samples = (
    dgi.encoder(val_g, val_g.ndata["h"], val_g.edata["h"])[1].detach().cpu().numpy()
)
test_samples = (
    dgi.encoder(test_g, test_g.ndata["h"], test_g.edata["h"])[1].detach().cpu().numpy()
)

train_labels = train_g.edata["Label"].detach().cpu().numpy()
validation_labels = val_g.edata["Label"].detach().cpu().numpy()
test_labels = test_g.edata["Label"].detach().cpu().numpy()

benign_train_mask = train_labels == 0
benign_train_samples = training_emb[benign_train_mask]

cblof_params = [2, 3, 5, 7, 9, 10]
hbos_params = [5, 10, 15, 20, 25, 30]
pca_params = [5, 10, 15, 20, 25, 30]
if_params = [20, 50, 100, 150]


print("\nNow testing on benign samples")
best_model_params = {}
threshold = 0
score = -1
# CBLOF
for param in (pbar := tqdm(cblof_params)):
    clf = CBLOF(n_clusters=param, random_state=seed)
    try:
        clf.fit(benign_train_samples)
    except (ValueError, RuntimeError) as e:
        print(f"Skipping param {param} due to: {e}")
        continue
    y_scores = clf.decision_function(validation_samples)
    pr_auc = average_precision_score(validation_labels, y_scores)
    threshold, _ = find_optimal_threshold(validation_labels, y_scores)

    if pr_auc > score:
        score = pr_auc
        best_param = param
        best_threshold = threshold
    pbar.set_postfix({"cur-pr_auc": pr_auc, "best-pr_auc": score})
    del clf
    gc.collect()

# Test CBLOF with best param on the test set
clf = CBLOF(n_clusters=best_param, random_state=seed)
clf.fit(benign_train_samples)
y_scores = clf.decision_function(test_samples)
pr_auc = average_precision_score(test_labels, y_scores)
y_pred = (y_scores >= best_threshold).astype(int)
f1 = f1_score(test_labels, y_pred, average="macro")
precision, recall, thresholds = precision_recall_curve(test_labels, y_scores)
if save_curve:
    os.makedirs("curves", exist_ok=True)
    np.savez_compressed(
        f"curves/{run.name}_{dataset_name}_CBLOF_precision_recall.npz",
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        y_scores=y_scores,
        y_true=test_labels.values,
    )
test_classification_report = classification_report(test_labels, y_pred, zero_division=0)
print(f"CBLOF test set results: PR-AUC={pr_auc}, F1={f1}")
wandb.log({"CBLOF_test_PR_AUC": pr_auc, "CBLOF_test_F1": f1})
print(f"Test classification report:\n{test_classification_report}")
del clf
gc.collect()
score = -1
threshold = 0
# HBOS
for param in (pbar := tqdm(hbos_params)):
    clf = HBOS(n_bins=param)
    try:
        clf.fit(benign_train_samples)
    except (ValueError, RuntimeError) as e:
        print(f"Skipping param {param} due to: {e}")
        continue
    y_scores = clf.decision_function(validation_samples)
    pr_auc = average_precision_score(validation_labels, y_scores)
    threshold, _ = find_optimal_threshold(validation_labels, y_scores)

    if pr_auc > score:
        score = pr_auc
        best_param = param
        best_threshold = threshold
    pbar.set_postfix({"cur-pr_auc": pr_auc, "best-pr_auc": score})
    del clf
    gc.collect()

# Test HBOS with best param on the test set
clf = HBOS(n_bins=best_param)
clf.fit(benign_train_samples)
y_scores = clf.decision_function(test_samples)
pr_auc = average_precision_score(test_labels, y_scores)
y_pred = (y_scores >= best_threshold).astype(int)
f1 = f1_score(test_labels, y_pred, average="macro")
precision, recall, thresholds = precision_recall_curve(test_labels, y_scores)
os.makedirs("curves", exist_ok=True)
np.savez_compressed(
    f"curves/{run.name}_{dataset_name}_HBOS_precision_recall.npz",
    precision=precision,
    recall=recall,
    thresholds=thresholds,
    y_scores=y_scores,
    y_true=test_labels.values,
)
test_classification_report = classification_report(test_labels, y_pred, zero_division=0)
print(f"HBOS test set results: PR-AUC={pr_auc}, F1={f1}")
wandb.log({"HBOS_test_PR_AUC": pr_auc, "HBOS_test_F1": f1})
print(f"Test classification report:\n{test_classification_report}")
del clf
gc.collect()
score = -1
threshold = 0
# PCA
for param in (pbar := tqdm(pca_params)):
    clf = PCA(n_components=param)
    try:
        clf.fit(benign_train_samples)
    except (ValueError, RuntimeError) as e:
        print(f"Skipping param {param} due to: {e}")
        continue
    y_scores = clf.decision_function(validation_samples)
    pr_auc = average_precision_score(validation_labels, y_scores)
    threshold, _ = find_optimal_threshold(validation_labels, y_scores)

    if pr_auc > score:
        score = pr_auc
        best_param = param
        best_threshold = threshold
    pbar.set_postfix({"cur-pr_auc": pr_auc, "best-pr_auc": score})
    del clf
    gc.collect()
# Test PCA with best param on the test set
clf = PCA(n_components=best_param)
clf.fit(benign_train_samples)
y_scores = clf.decision_function(test_samples)
pr_auc = average_precision_score(test_labels, y_scores)
y_pred = (y_scores >= best_threshold).astype(int)
f1 = f1_score(test_labels, y_pred, average="macro")
precision, recall, thresholds = precision_recall_curve(test_labels, y_scores)
os.makedirs("curves", exist_ok=True)
np.savez_compressed(
    f"curves/{run.name}_{dataset_name}_PCA_precision_recall.npz",
    precision=precision,
    recall=recall,
    thresholds=thresholds,
    y_scores=y_scores,
    y_true=test_labels.values,
)
test_classification_report = classification_report(test_labels, y_pred, zero_division=0)
print(f"PCA test set results: PR-AUC={pr_auc}, F1={f1}")
wandb.log({"PCA_test_PR_AUC": pr_auc, "PCA_test_F1": f1})
print(f"Test classification report:\n{test_classification_report}")
del clf
gc.collect()
score = -1
threshold = 0
# Isolation Forest
for param in (pbar := tqdm(if_params)):
    clf = IForest(n_estimators=param, random_state=seed)
    try:
        clf.fit(benign_train_samples)
    except (ValueError, RuntimeError) as e:
        print(f"Skipping param {param} due to: {e}")
        continue
    y_scores = clf.decision_function(validation_samples)
    pr_auc = average_precision_score(validation_labels, y_scores)
    threshold, _ = find_optimal_threshold(validation_labels, y_scores)

    if pr_auc > score:
        score = pr_auc
        best_param = param
        best_threshold = threshold
    pbar.set_postfix({"cur-pr_auc": pr_auc, "best-pr_auc": score})
    del clf
    gc.collect()
# Test Isolation Forest with best param on the test set
clf = IForest(n_estimators=best_param, random_state=seed)
clf.fit(benign_train_samples)
y_scores = clf.decision_function(test_samples)
pr_auc = average_precision_score(test_labels, y_scores)
y_pred = (y_scores >= best_threshold).astype(int)
f1 = f1_score(test_labels, y_pred, average="macro")
precision, recall, thresholds = precision_recall_curve(test_labels, y_scores)
os.makedirs("curves", exist_ok=True)
np.savez_compressed(
    f"curves/{run.name}_{dataset_name}_IF_precision_recall.npz",
    precision=precision,
    recall=recall,
    thresholds=thresholds,
    y_scores=y_scores,
    y_true=test_labels.values,
)
test_classification_report = classification_report(test_labels, y_pred, zero_division=0)
print(f"Isolation Forest test set results: PR-AUC={pr_auc}, F1={f1}")
wandb.log({"IF_test_PR_AUC": pr_auc, "IF_test_F1": f1})
print(f"Test classification report:\n{test_classification_report}")
del clf
gc.collect()
wandb.finish()
