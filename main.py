import os
import random
import warnings

import numpy as np
import torch
import wandb
from sklearn.metrics import precision_recall_curve
from torch_geometric.loader import LinkNeighborLoader

from models.graphids import GraphIDS
from utils.dataloaders import NetFlowDataset
from utils.parser import Parser
from utils.trainers import test, train

# Suppress this warning: even if in prototype stage, it works correctly for our use case
warnings.filterwarnings(
    "ignore", message="The PyTorch API of nested tensors is in prototype stage"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENT_CONFIG_KEYS = (
    "data_type",
    "dataset",
    "num_epochs",
    "learning_rate",
    "weight_decay",
    "ae_weight_decay",
    "edim_out",
    "batch_size",
    "fanout",
    "agg_type",
    "num_layers",
    "mask_ratio",
    "patience",
    "ae_batch_size",
    "window_size",
    "step_percent",
    "ae_embedding_dim",
    "ae_feedforward_dim",
    "ae_dropout",
    "dropout",
    "positional_encoding",
    "fraction",
    "split_mode",
    "distribution_segment",
)
RUNTIME_CONFIG_KEYS = (
    "data_dir",
    "checkpoint",
    "reload_dataset",
    "test",
    "save_curve",
    "seed",
    "wandb",
)
BACKWARD_COMPATIBLE_DEFAULT_KEYS = (
    "split_mode",
    "distribution_segment",
    "ae_feedforward_dim",
)
REQUIRED_CONFIG_KEYS = EXPERIMENT_CONFIG_KEYS + RUNTIME_CONFIG_KEYS


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_wandb_config(args):
    if args.config is not None:
        return args.config
    return {key: getattr(args, key) for key in EXPERIMENT_CONFIG_KEYS}


def apply_cli_config(run_config, args):
    for key in RUNTIME_CONFIG_KEYS:
        run_config[key] = getattr(args, key)

    for key in BACKWARD_COMPATIBLE_DEFAULT_KEYS:
        if key not in run_config:
            run_config[key] = getattr(args, key)


def ensure_config_keys(config):
    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise KeyError(f"Missing required configuration values: {missing}")


def default_checkpoint_path(config, run_name):
    checkpoint_id = run_name if config.wandb else config.seed
    checkpoint_dataset = config.dataset
    if config.split_mode != "stratified":
        checkpoint_dataset = f"{checkpoint_dataset}_{config.split_mode}"
        if config.split_mode == "temporal_shift_aware":
            checkpoint_dataset = f"{checkpoint_dataset}_{config.distribution_segment}"
    return f"checkpoints/GraphIDS_{checkpoint_dataset}_{checkpoint_id}.ckpt"


def resolve_checkpoint_path(config, run_name):
    if config.checkpoint is not None:
        return config.checkpoint
    return default_checkpoint_path(config, run_name)


def main(run):
    config = run.config

    set_seed(config.seed)
    split_mode = config.split_mode
    distribution_segment = config.distribution_segment

    dataset = NetFlowDataset(
        name=config.dataset,
        data_dir=config.data_dir,
        force_reload=config.reload_dataset,
        fraction=config.fraction,
        data_type=config.data_type,
        seed=config.seed,
        split_mode=split_mode,
        distribution_segment=distribution_segment,
    )

    ndim_in = dataset.num_node_features
    edim_in = dataset.num_edge_features
    print("Number of features:", edim_in)

    model = GraphIDS(
        ndim_in=ndim_in,
        edim_in=edim_in,
        edim_out=config.edim_out,
        embed_dim=config.ae_embedding_dim,
        num_heads=4,
        num_layers=config.num_layers,
        window_size=config.window_size,
        dropout=config.dropout,
        ae_dropout=config.ae_dropout,
        positional_encoding=config.positional_encoding,
        agg_type=config.agg_type,
        mask_ratio=config.mask_ratio,
        ae_feedforward_dim=config.ae_feedforward_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {  # Higher weight decay for embedding layer
                "params": model.encoder.parameters(),
                "weight_decay": config.weight_decay,
            },
            {  # Lower weight decay for the transformer
                "params": model.transformer.parameters(),
                "weight_decay": config.ae_weight_decay,
            },
        ],
        lr=config.learning_rate,
    )
    checkpoint = resolve_checkpoint_path(config, run.name)
    if os.path.exists(checkpoint):
        print("Loading model from checkpoint")
        start_epoch, threshold = model.load_checkpoint(checkpoint, optimizer)
        run.config.epoch = start_epoch
    else:
        checkpoint_dir = os.path.dirname(checkpoint)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        start_epoch = 0
        threshold = None

    shuffle = config.positional_encoding == "None"
    fanout_list = [config.fanout] if config.fanout != -1 else [-1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cpu_count = os.cpu_count()
    recommended_workers = min(cpu_count, 6) if cpu_count is not None else 0
    if start_epoch >= config.num_epochs or config.test:
        print("Model already trained")
        test_loader = LinkNeighborLoader(
            data=dataset.test_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.test_graph.edge_index,
            edge_label=dataset.test_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )
    else:
        train_loader = LinkNeighborLoader(
            data=dataset.train_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.train_graph.edge_index,
            edge_label=dataset.train_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        val_loader = LinkNeighborLoader(
            data=dataset.val_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.val_graph.edge_index,
            edge_label=dataset.val_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        test_loader = LinkNeighborLoader(
            data=dataset.test_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.test_graph.edge_index,
            edge_label=dataset.test_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )
        print("Starting training...")
        model, threshold = train(
            model,
            config.window_size,
            config.step_percent,
            config.ae_batch_size,
            train_loader,
            val_loader,
            start_epoch,
            config.num_epochs,
            optimizer,
            run,
            config.patience,
            checkpoint,
            device=device,
        )

    print("Evaluating on test set...")
    test_f1, test_pr_auc, errors, test_labels, prediction_time = test(
        model,
        test_loader,
        config.ae_batch_size,
        config.window_size,
        device,
        threshold=threshold,
    )
    if config.save_curve:
        precision, recall, _ = precision_recall_curve(test_labels.cpu(), errors.cpu())
        run.log(
            {
                "Precision-Recall Curve": wandb.plot.pr_curve(
                    y_true=test_labels.cpu().numpy(),
                    y_probas=errors.cpu().numpy(),
                    title="Precision-Recall Curve",
                ),
            }
        )
        os.makedirs("curves", exist_ok=True)
        np.savez(
            f"curves/precision_recall_{run.name}.npz",
            precision=precision,
            recall=recall,
        )
    test_pred = (errors > threshold).int()
    print(f"Test macro F1-score: {test_f1:.4f}")
    print(f"Test PR-AUC/AP: {test_pr_auc:.4f}")
    print(f"Test prediction time: {prediction_time:.4f} seconds")
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU memory usage: {peak_memory_mb:.2f} MB")
    else:
        peak_memory_mb = 0
    run.log(
        {
            "final_test_f1": test_f1,
            "final_test_pr_auc": test_pr_auc,
            "final_test_ap": test_pr_auc,
            "test_threshold": threshold,
            "test_prediction_time": prediction_time,
            "peak_gpu_memory_mb": peak_memory_mb,
            "Test Confusion Matrix": wandb.plot.confusion_matrix(
                y_true=test_labels.ravel().tolist(),
                preds=test_pred.ravel().tolist(),
                class_names=["Benign", "Malicious"],
                title="Test Confusion Matrix",
            ),
        }
    )
    run.finish()


if __name__ == "__main__":
    args = Parser().parse_args()
    config = build_wandb_config(args)
    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(project="GraphIDS", config=config)
    apply_cli_config(run.config, args)
    ensure_config_keys(run.config)

    main(run)
