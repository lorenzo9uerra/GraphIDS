import time
from sklearn.metrics import average_precision_score, f1_score
import torch
from tqdm import tqdm
import torch.nn as nn


def train_encoder(
    model,
    train_loader,
    val_loader,
    test_loader,
    start_epoch,
    num_epochs,
    optimizer,
    run,
    patience,
    checkpoint,
    device="cuda",
):
    best_pr_auc = 0.0
    cnt_wait = 0
    criterion = nn.MSELoss()
    total_train_loss = 0
    for epoch in (pbar := tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epochs")):
        total_train_loss = 0
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            block, nfeats, efeats = (
                data.blocks[0],
                data.node_features["h"],
                data.edge_features[0]["h"],
            )
            edge_embeddings, reconstructed = model(
                block, nfeats, efeats, data.compacted_seeds.T
            )
            loss = criterion(reconstructed, edge_embeddings)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_train_loss /= len(train_loader)
        val_loss, val_errors, val_labels = validate(model, val_loader, device)
        val_pr_auc = average_precision_score(val_labels.cpu(), val_errors.cpu())
        # Find the best threshold based on the validation set
        threshold = find_threshold(val_errors, val_labels, method="supervised")
        # For debugging purposes
        test_f1, test_pr_auc, _, _, _ = test(model, test_loader, device, threshold)

        # Keep saving the model if it produces the same or better validation PR-AUC
        if val_pr_auc >= best_pr_auc:
            model.save_checkpoint(
                checkpoint,
                optimizer=optimizer,
                epoch=epoch,
                threshold=threshold,
            )

        # Stop training if the validation loss does not improve for a number of epochs
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            cnt_wait = 0
        else:
            cnt_wait += 1
            if cnt_wait >= patience:
                print("Early stopping!")
                break
        pbar.set_postfix(
            {
                "train_loss": total_train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
                "test_f1": test_f1,
                "test_pr_auc": test_pr_auc,
            }
        )
        run.log(
            {
                "train_loss": total_train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
                "test_f1": test_f1,
                "test_pr_auc": test_pr_auc,
            }
        )
    chk = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(chk["model_state_dict"])
    return model, chk["threshold"]


def find_threshold(errors, labels=None, method="unsupervised", multiplier=10.0):
    if method == "unsupervised":
        median = errors.median()
        mad = (
            errors - median
        ).abs().median() * 1.4826  # Factor for normal distribution
        best_threshold = median + multiplier * mad
    elif method == "supervised":
        best_f1 = 0.0
        best_threshold = errors.mean()
        for threshold in torch.linspace(errors.min(), errors.max(), steps=500):
            val_pred = (errors > threshold).int()
            f1 = f1_score(
                labels.cpu(), val_pred.cpu(), average="macro", zero_division=0
            )
            if f1 > best_f1:
                best_threshold = threshold.item()
                best_f1 = f1
    return best_threshold


def calculate_errors(outputs, targets):
    # Calculate mean squared error for each sample
    squared_errors = (outputs - targets) ** 2
    mean_errors = torch.mean(squared_errors, dim=-1)
    return mean_errors


def validate(model, val_loader, device):
    criterion = nn.MSELoss()
    model.eval()
    errors = []
    labels = []
    total_val_loss = 0
    with torch.inference_mode():
        for data in val_loader:
            block, nfeats, efeats = (
                data.blocks[0],
                data.node_features["h"],
                data.edge_features[0]["h"],
            )
            edge_embeddings, reconstructed = model(
                block, nfeats, efeats, data.compacted_seeds.T
            )
            loss = criterion(reconstructed, edge_embeddings)
            total_val_loss += loss.item()

            batch_errors = calculate_errors(reconstructed, edge_embeddings)
            errors.append(batch_errors.cpu())
            labels.append(data.labels.cpu())

    total_val_loss /= len(val_loader)
    labels = torch.cat(labels)
    errors = torch.cat(errors)
    return total_val_loss, errors, labels


def test(model, test_loader, device, threshold):
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.perf_counter()
    model.eval()
    errors = []
    labels = []
    with torch.inference_mode():
        for data in test_loader:
            block, nfeats, efeats = (
                data.blocks[0],
                data.node_features["h"],
                data.edge_features[0]["h"],
            )
            edge_embeddings, reconstructed = model(
                block, nfeats, efeats, data.compacted_seeds.T
            )

            batch_errors = calculate_errors(reconstructed, edge_embeddings)
            errors.append(batch_errors.cpu())
            labels.append(data.labels.cpu())

    labels = torch.cat(labels)
    errors = torch.cat(errors)
    test_pred = (errors > threshold).int()
    torch.cuda.synchronize() if device == "cuda" else None
    prediction_time = time.perf_counter() - start_time
    f1 = f1_score(labels, test_pred, average="macro", zero_division=0)
    pr_auc = average_precision_score(labels, errors)
    return f1, pr_auc, errors, labels, prediction_time
