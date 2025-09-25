import time
from sklearn.metrics import average_precision_score, f1_score
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataloaders import SequentialDataset, collate_fn


def train_encoder(
    model,
    window_size,
    step_percent,
    ae_batch_size,
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
    criterion = nn.MSELoss(reduction="none")
    total_train_loss = 0
    for epoch in (pbar := tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epochs")):
        total_train_loss = 0
        model.train()
        for batch in train_loader:
            batch.batch_edge_couples = batch.edge_label_index.t()
            batch = batch.to(device)

            train_emb = model.encoder(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch_edge_couples
            )
            ae_train_loader = DataLoader(
                SequentialDataset(
                    train_emb,
                    window=window_size,
                    step=int(window_size * step_percent),
                    device=device,
                ),
                batch_size=ae_batch_size,
                collate_fn=collate_fn,
            )
            accumulated_loss = 0
            seq_count = 0
            for ae_batch, mask in ae_train_loader:
                outputs = model.transformer(ae_batch, mask)
                loss = criterion(outputs, ae_batch)
                loss = torch.sum(loss * mask) / torch.sum(mask)
                accumulated_loss += loss
                seq_count += 1

            # Calculate the mean loss for the batch and backpropagate through both components
            if seq_count > 0:
                loss = accumulated_loss / seq_count
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_train_loss /= len(train_loader)
        val_loss, val_errors, val_labels = validate(
            model, val_loader, ae_batch_size, window_size, device
        )
        val_pr_auc = average_precision_score(val_labels.cpu(), val_errors.cpu())
        # Find the best threshold based on the validation set
        threshold = find_threshold(val_errors, val_labels, method="supervised")
        # For debugging purposes
        test_f1, test_pr_auc, _, _, _ = test(
            model, test_loader, ae_batch_size, window_size, device, threshold
        )

        # Keep saving the model if it produces the same or better validation PR-AUC
        if val_pr_auc >= best_pr_auc:
            model.save_checkpoint(
                checkpoint,
                optimizer=optimizer,
                epoch=epoch,
                threshold=threshold,
            )

        # Stop training if the validation PR-AUC does not improve for a number of epochs
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
    elif method == "supervised" and labels is not None:
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
    else:
        raise ValueError(
            "Invalid method for threshold finding. Use 'unsupervised' or 'supervised' with labels."
        )
    return best_threshold


def calculate_errors(outputs, batch, mask):
    squared_errors = ((outputs - batch) ** 2) * mask
    valid_mask = mask.sum(dim=-1) > 0
    valid_counts = torch.sum(mask, dim=-1)
    mean_errors = torch.zeros_like(valid_counts, dtype=torch.float32)
    if valid_mask.any():
        mean_errors[valid_mask] = torch.sum(squared_errors, dim=-1)[
            valid_mask
        ] / torch.clamp(valid_counts[valid_mask], min=1)
    mean_errors = torch.nan_to_num(mean_errors, nan=0.0, posinf=1e6, neginf=-1e6)
    return mean_errors[valid_mask]


def validate(model, val_loader, ae_batch_size, window_size, device):
    criterion = nn.MSELoss(reduction="none")
    model.eval()
    errors = []
    labels = []
    total_val_loss = 0
    with torch.inference_mode():
        for batch in val_loader:
            batch.batch_edge_couples = batch.edge_label_index.t()
            batch = batch.to(device)

            val_emb = model.encoder(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch_edge_couples
            )
            labels.append(batch.edge_label.cpu())
            ae_val_loader = DataLoader(
                SequentialDataset(
                    val_emb, window=window_size, step=window_size, device=device
                ),
                batch_size=ae_batch_size,
                collate_fn=collate_fn,
            )
            accumulated_loss = 0
            seq_count = 0
            for ae_batch, mask in ae_val_loader:
                outputs = model.transformer(ae_batch, mask)
                loss = criterion(outputs, ae_batch)
                loss = torch.sum(loss * mask) / torch.sum(mask)
                accumulated_loss += loss
                seq_count += 1
                batch_errors = calculate_errors(outputs, ae_batch, mask)
                errors.append(batch_errors.cpu())
            if seq_count > 0:
                total_val_loss += (accumulated_loss / seq_count).item()
    total_val_loss /= len(val_loader)
    labels = torch.cat(labels)
    errors = torch.cat(errors)
    return total_val_loss, errors, labels


def test(model, test_loader, ae_batch_size, window_size, device, threshold):
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.perf_counter()
    model.eval()
    errors = []
    labels = []
    with torch.inference_mode():
        for batch in test_loader:
            batch.batch_edge_couples = batch.edge_label_index.t()
            batch = batch.to(device)

            test_emb = model.encoder(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch_edge_couples
            )
            labels.append(batch.edge_label.cpu())
            ae_test_loader = DataLoader(
                SequentialDataset(
                    test_emb, window=window_size, step=window_size, device=device
                ),
                batch_size=ae_batch_size,
                collate_fn=collate_fn,
            )
            for ae_batch, mask in ae_test_loader:
                outputs = model.transformer(ae_batch, mask)
                batch_errors = calculate_errors(outputs, ae_batch, mask)
                errors.append(batch_errors.cpu())
    labels = torch.cat(labels)
    errors = torch.cat(errors)
    if threshold is not None:
        test_pred = (errors > threshold).int()
    else:
        print("No threshold provided, using mean of errors for prediction.")
        test_pred = (errors > errors.mean()).int()
    torch.cuda.synchronize() if device == "cuda" else None
    prediction_time = time.perf_counter() - start_time
    f1 = f1_score(labels, test_pred, average="macro", zero_division=0)
    pr_auc = average_precision_score(labels, errors)
    return f1, pr_auc, errors, labels, prediction_time
