import os
import time
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import wandb
from utils.parser import Parser
import warnings

# Suppress this warning: even if in prototype stage, it works correctly in our case
warnings.filterwarnings(
    "ignore", message="The PyTorch API of nested tensors is in prototype stage"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    sequences, masks = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    return sequences_padded, masks_padded


class SequentialDataset(Dataset):
    def __init__(self, data, window, device, step=None):
        self.data = data
        self.window = window
        self.device = device
        if step is None:
            self.step = window
        else:
            self.step = step

    def __getitem__(self, index):
        start_idx = index * self.step
        end_idx = min(start_idx + self.window, len(self.data))
        x = self.data[start_idx:end_idx].to(self.device)
        mask = torch.ones_like(x, dtype=torch.bool).to(self.device)
        return x, mask

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.step + 1)


class TransformerAutoencoder(nn.Module):
    def __init__(
        self, input_dim, embed_dim, num_heads, num_layers, dropout, mask_ratio
    ):
        super(TransformerAutoencoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.mask_ratio = mask_ratio
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(embed_dim, input_dim)

    def forward(self, src, padding_mask=None):
        src = self.input_projection(src)

        if padding_mask is not None:
            padding_mask = ~torch.any(padding_mask, dim=-1)

        if self.training and self.mask_ratio > 0:
            seq_len = src.size(1)
            mask = torch.rand(seq_len, seq_len, device=src.device) < self.mask_ratio
            attention_mask = mask | mask.transpose(0, 1)  # Symmetric masking
        else:
            attention_mask = None

        memory = self.encoder(
            src, mask=attention_mask, src_key_padding_mask=padding_mask
        )

        output = self.decoder(
            src,
            memory,
            memory_key_padding_mask=padding_mask,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=padding_mask,
        )

        output = self.output_projection(output)
        return output


def train_encoder(
    model,
    train_loader,
    val_loader,
    test_loader,
    val_labels,
    test_labels,
    start_epoch,
    num_epochs,
    optimizer,
    run,
    patience=7,
    checkpoint="checkpoints/model.ckpt",
):
    best_pr_auc = 0.0
    cnt_wait = 0
    criterion = nn.MSELoss()
    total_train_loss = 0
    for epoch in (pbar := tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epochs")):
        total_train_loss = 0
        model.train()
        for batch, mask in train_loader:
            outputs = model(batch, mask)
            loss = criterion(outputs, batch)
            loss = torch.sum(loss * mask) / torch.sum(mask)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        total_train_loss /= len(train_loader)
        val_loss, val_errors = validate(model, val_loader)
        val_pr_auc = average_precision_score(val_labels.cpu(), val_errors.cpu())
        # Find the best threshold based on the validation set
        threshold = find_threshold(val_errors, val_labels, method="supervised")
        # For debugging purposes
        test_f1, test_pr_auc, _, _ = test(model, test_loader, test_labels, threshold)

        # if val_pr_auc continues to be the best_pr_auc seen, save the model
        # This produces a model that better generalizes to the test set
        if val_pr_auc >= best_pr_auc:
            if not os.path.exists(os.path.dirname(checkpoint)):
                os.makedirs(os.path.dirname(checkpoint))
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "threshold": threshold,
                },
                checkpoint,
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


def calculate_errors(outputs, batch, mask):
    squared_errors = ((outputs - batch) ** 2) * mask
    valid_mask = mask.any(dim=-1)
    valid_counts = torch.sum(mask, dim=-1)
    # Avoid division by zero
    mean_errors = torch.zeros_like(valid_counts, dtype=torch.float32)
    mean_errors[valid_mask] = torch.sum(squared_errors, dim=-1)[
        valid_mask
    ] / torch.clamp(valid_counts[valid_mask], min=1)
    return mean_errors[valid_mask]


def validate(model, val_loader):
    criterion = nn.MSELoss()
    model.eval()
    total_val_loss = []
    errors = []
    with torch.no_grad():
        total_train_loss = 0
        for batch, mask in val_loader:
            outputs = model(batch, mask)
            loss = criterion(outputs, batch)
            loss = torch.sum(loss * mask) / torch.sum(mask)
            total_val_loss.append(loss.item())
            batch_errors = calculate_errors(outputs, batch, mask)
            errors.append(batch_errors.cpu())

        total_train_loss += loss.item()
    total_val_loss = sum(total_val_loss) / len(total_val_loss)
    errors = torch.cat(errors)
    return total_val_loss, errors


def test(model, test_loader, test_labels, threshold):
    start_time = time.time()
    model.eval()
    errors = []
    with torch.no_grad():
        for batch, mask in test_loader:
            outputs = model(batch, mask)
            batch_errors = calculate_errors(outputs, batch, mask)
            errors.append(batch_errors.cpu())
    errors = torch.cat(errors)
    test_pred = (errors > threshold).int()
    prediction_time = time.time() - start_time
    f1 = f1_score(test_labels.cpu(), test_pred.cpu(), average="macro", zero_division=0)
    pr_auc = average_precision_score(test_labels.cpu(), errors.cpu())
    return f1, pr_auc, errors, prediction_time


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def main(run):
    config = run.config

    set_seed(config.seed)

    dataset_dir = os.path.join("datasets", config.dataset)
    train_file = os.path.join(dataset_dir, "train.csv")
    val_file = os.path.join(dataset_dir, "val.csv")
    test_file = os.path.join(dataset_dir, "test.csv")

    if (
        os.path.exists(train_file)
        and os.path.exists(val_file)
        and os.path.exists(test_file)
        and not config.reload_dataset
    ):
        print(f"Loading preprocessed dataset from {dataset_dir}")
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        df_test = pd.read_csv(test_file)
    else:
        print(f"Preprocessing dataset {config.dataset}")
        os.makedirs(dataset_dir, exist_ok=True)
        df = pd.read_csv(
            os.path.join(config.data_dir, config.dataset, f"{config.dataset}.csv")
        )
        if config.fraction is not None:
            df = df.groupby(by="Attack").sample(
                frac=config.fraction, random_state=config.seed
            )
        X = df.drop(columns=["Attack", "Label"])
        y = df[["Attack", "Label"]]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        if "v3" in config.dataset:
            edge_features = [
                col
                for col in X.columns
                if col
                not in [
                    "IPV4_SRC_ADDR",
                    "IPV4_DST_ADDR",
                    "FLOW_END_MILLISECONDS",
                    "FLOW_START_MILLISECONDS",
                ]
            ]
        else:
            edge_features = [
                col
                for col in X.columns
                if col not in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
            ]
        df = pd.concat([X, y], axis=1)
        df_train, df_val_test = train_test_split(
            df, test_size=0.2, random_state=config.seed, stratify=y["Attack"]
        )
        df_train = df_train[df_train["Label"] == 0]
        scaler = MinMaxScaler()
        df_train[edge_features] = scaler.fit_transform(df_train[edge_features])
        df_val_test_scaled = scaler.transform(df_val_test[edge_features])
        df_val_test[edge_features] = np.clip(df_val_test_scaled, -10, 10)
        df_val, df_test = train_test_split(
            df_val_test,
            test_size=0.5,
            random_state=config.seed,
            stratify=df_val_test["Attack"],
        )
        if "v3" in config.dataset:
            df_train = df_train.sort_values(by="FLOW_START_MILLISECONDS").drop(
                columns=[
                    "Attack",
                    "FLOW_END_MILLISECONDS",
                    "FLOW_START_MILLISECONDS",
                    "IPV4_SRC_ADDR",
                    "IPV4_DST_ADDR",
                ]
            )
            df_val = df_val.sort_values(by="FLOW_START_MILLISECONDS").drop(
                columns=[
                    "Attack",
                    "FLOW_END_MILLISECONDS",
                    "FLOW_START_MILLISECONDS",
                    "IPV4_SRC_ADDR",
                    "IPV4_DST_ADDR",
                ]
            )
            df_test = df_test.sort_values(by="FLOW_START_MILLISECONDS").drop(
                columns=[
                    "Attack",
                    "FLOW_END_MILLISECONDS",
                    "FLOW_START_MILLISECONDS",
                    "IPV4_SRC_ADDR",
                    "IPV4_DST_ADDR",
                ]
            )
        else:
            df_train = df_train.drop(
                columns=["Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
            )
            df_val = df_val.drop(columns=["Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"])
            df_test = df_test.drop(columns=["Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"])
        df_train = df_train.replace([np.inf, -np.inf], 0)
        df_train = df_train.fillna(0)
        df_val = df_val.replace([np.inf, -np.inf], 0)
        df_val = df_val.fillna(0)
        df_test = df_test.replace([np.inf, -np.inf], 0)
        df_test = df_test.fillna(0)
        df_train.to_csv(train_file, index=False)
        df_val.to_csv(val_file, index=False)
        df_test.to_csv(test_file, index=False)

    X_train = df_train.drop(columns=["Label"]).values
    y_val = df_val["Label"].values
    X_val = df_val.drop(columns=["Label"]).values
    y_test = df_test["Label"].values
    X_test = df_test.drop(columns=["Label"]).values

    model = TransformerAutoencoder(
        X_train.shape[1],
        config.ae_embedding_dim,
        4,  # Number of attention heads
        config.num_layers,
        config.ae_dropout,
        config.mask_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=config.ae_weight_decay,
        lr=config.learning_rate,
    )

    if config.checkpoint is not None and os.path.exists(config.checkpoint):
        print("Loading model from checkpoint")
        chk = torch.load(config.checkpoint, weights_only=True)
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk["epoch"]
        threshold = chk["threshold"]
        run.config.epoch = start_epoch
    else:
        checkpoint = f"checkpoints/T-MAE_{config.dataset}_{config.seed}.ckpt"
        start_epoch = 0

    window_size = config.window_size
    step_percent = config.step_percent
    ae_batch_size = config.ae_batch_size
    if start_epoch >= config.num_epochs or config.test:
        print("Model already trained")
        test_data = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_labels = torch.tensor(y_test, dtype=torch.float32).to(device)
        test_loader = DataLoader(
            SequentialDataset(
                test_data,
                window=window_size,
                step=int(window_size * step_percent),
                device=device,
            ),
            batch_size=ae_batch_size,
            collate_fn=collate_fn,
        )
    else:
        train_data = torch.tensor(X_train, dtype=torch.float32).to(device)
        val_data = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_labels = torch.tensor(y_val, dtype=torch.float32).to(device)
        test_data = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_labels = torch.tensor(y_test, dtype=torch.float32).to(device)

        train_loader = DataLoader(
            SequentialDataset(
                train_data,
                window=window_size,
                step=int(window_size * step_percent),
                device=device,
            ),
            batch_size=ae_batch_size,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            SequentialDataset(
                val_data,
                window=window_size,
                step=int(window_size * step_percent),
                device=device,
            ),
            batch_size=ae_batch_size,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            SequentialDataset(
                test_data,
                window=window_size,
                step=int(window_size * step_percent),
                device=device,
            ),
            batch_size=ae_batch_size,
            collate_fn=collate_fn,
        )
        model, threshold = train_encoder(
            model,
            train_loader,
            val_loader,
            test_loader,
            val_labels,
            test_labels,
            start_epoch,
            config.num_epochs,
            optimizer,
            run,
            patience=config.patience,
            checkpoint=checkpoint,
        )
    print("Loaded embeddings")

    test_f1, test_pr_auc, errors, prediction_time = test(
        model,
        test_loader,
        test_labels,
        threshold,
    )
    precision, recall, _ = precision_recall_curve(test_labels.cpu(), errors.cpu())
    if config.save_curve:
        os.makedirs("curves", exist_ok=True)
        np.savez(
            f"curves/precision_recall_{run.name}.npz",
            precision=precision,
            recall=recall,
        )
    test_pred = (errors > threshold).int()
    run.log(
        {
            "final_test_f1": test_f1,
            "final_test_pr_auc": test_pr_auc,
            "test_threshold": threshold,
            "test_prediction_time": prediction_time,
        }
    )
    run.log(
        {
            "Validation Confusion Matrix": wandb.plot.confusion_matrix(
                y_true=test_labels.ravel().tolist(),
                preds=test_pred.ravel().tolist(),
                class_names=["Benign", "Malicious"],
                title="Validation Confusion Matrix",
            ),
        }
    )
    run.finish()


if __name__ == "__main__":
    args = Parser().parse_args()
    if args.config is not None:
        config = args.config
    else:
        config = {
            "data_type": args.data_type,
            "dataset": args.dataset,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "ae_weight_decay": args.ae_weight_decay,
            "num_layers": args.num_layers,
            "mask_ratio": args.mask_ratio,
            "patience": args.patience,
            "ae_batch_size": args.ae_batch_size,
            "window_size": args.window_size,
            "ae_embedding_dim": args.ae_embedding_dim,
            "ae_dropout": args.ae_dropout,
            "reload_dataset": args.reload_dataset,
            "step_percent": args.step_percent,
            "test": args.test,
            "fraction": args.fraction,
        }
    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(project="T-MAE", config=config)

    # Set of parameters that must be passed via command line
    run.config["data_dir"] = args.data_dir
    run.config["checkpoint"] = args.checkpoint
    run.config["save_curve"] = args.save_curve
    run.config["seed"] = args.seed

    main(run)
