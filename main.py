import numpy as np
import random
import torch
import wandb
import os
from models.GraphIDS import GraphIDS
from utils.dataloaders import NetFlowDataset, GraphDataLoader
from utils.trainers import test, train_encoder
from utils.parser import Parser
import warnings
from sklearn.metrics import precision_recall_curve

# Suppress this warning: even if in prototype stage, it works correctly for our use case
warnings.filterwarnings(
    "ignore", message="The PyTorch API of nested tensors is in prototype stage"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def main(run):
    config = run.config

    set_seed(config.seed)

    dataset = NetFlowDataset(
        name=config.dataset,
        data_dir=config.data_dir,
        force_reload=config.reload_dataset,
        fraction=config.fraction,
        data_type=config.data_type,
        seed=config.seed,
    )

    ndim_in = dataset.train_data.feature.read("node", None, "h").shape[1]
    edim_in = dataset.train_data.feature.read("edge", None, "h").shape[1]
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
        nhops=config.nhops,
        agg_type="mean",
        mask_ratio=config.mask_ratio,
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
    checkpoint = config.checkpoint
    if checkpoint is not None and os.path.exists(checkpoint):
        print("Loading model from checkpoint")
        start_epoch, threshold = model.load_checkpoint(checkpoint, optimizer)
        run.config.epoch = start_epoch
    else:
        checkpoint = f"checkpoints/GraphIDS_{config.dataset}_{config.seed}.ckpt"
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        start_epoch = 0

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    if config.positional_encoding == "None":
        shuffle = True
    else:
        shuffle = False

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if start_epoch >= config.num_epochs or config.test:
        print("Model already trained")
        test_loader = GraphDataLoader(
            dataset.test_data,
            batch_size=config.batch_size,
            nhops=config.nhops,
            seed=config.seed,
            shuffle=shuffle,
            device=device,
        )
    else:
        train_loader = GraphDataLoader(
            dataset.train_data,
            batch_size=config.batch_size,
            nhops=config.nhops,
            seed=config.seed,
            fanout=config.fanout,
            shuffle=shuffle,
            device=device,
        )
        val_loader = GraphDataLoader(
            dataset.val_data,
            batch_size=config.batch_size,
            nhops=config.nhops,
            seed=config.seed,
            shuffle=shuffle,
            device=device,
        )
        test_loader = GraphDataLoader(
            dataset.test_data,
            batch_size=config.batch_size,
            nhops=config.nhops,
            seed=config.seed,
            shuffle=shuffle,
            device=device,
        )
        model, threshold = train_encoder(
            model,
            config.window_size,
            config.step_percent,
            config.ae_batch_size,
            train_loader,
            val_loader,
            test_loader,
            start_epoch,
            config.num_epochs,
            optimizer,
            run,
            config.patience,
            checkpoint,
        )

    test_f1, test_pr_auc, errors, test_labels, prediction_time = test(
        model,
        test_loader,
        config.ae_batch_size,
        config.window_size,
        device,
        threshold=threshold,
    )
    precision, recall, _ = precision_recall_curve(test_labels.cpu(), errors.cpu())
    if config.save_curve:
        run.log(
            {
                "Precision-Recall Curve": wandb.plot.pr_curve(
                    y_true=test_labels.cpu().numpy(),
                    probs=errors.cpu().numpy(),
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
    print(f"Test PR-AUC: {test_pr_auc:.4f}")
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
            "test_threshold": threshold,
            "test_prediction_time": prediction_time,
            "peak_gpu_memory_mb": peak_memory_mb,
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
            "weight_decay": args.weight_decay,
            "ae_weight_decay": args.ae_weight_decay,
            "edim_out": args.edim_out,
            "batch_size": args.batch_size,
            "nhops": args.nhops,
            "fanout": args.fanout,
            "agg_type": args.agg_type,
            "num_layers": args.num_layers,
            "mask_ratio": args.mask_ratio,
            "patience": args.patience,
            "ae_batch_size": args.ae_batch_size,
            "window_size": args.window_size,
            "step_percent": args.step_percent,
            "ae_embedding_dim": args.ae_embedding_dim,
            "ae_dropout": args.ae_dropout,
            "dropout": args.dropout,
            "positional_encoding": args.positional_encoding,
            "fraction": args.fraction,
        }
    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(project="GraphIDS-nhops", config=config)

    # Set of parameters that must be passed via command line
    run.config["data_dir"] = args.data_dir
    run.config["checkpoint"] = args.checkpoint
    run.config["reload_dataset"] = args.reload_dataset
    run.config["test"] = args.test
    run.config["save_curve"] = args.save_curve
    run.config["seed"] = args.seed

    main(run)
