import argparse


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description="GraphIDS")
        self.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to the config file",
        )
        self.add_argument(
            "--data_dir",
            type=str,
            required=True,
            help="Path to the data directory",
        )
        self.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Path to the checkpoint file",
        )
        self.add_argument(
            "--dataset",
            type=str,
            default="NF-CSE-CIC-IDS2018-v2",
            choices=[
                "NF-CSE-CIC-IDS2018-v2",
                "NF-UNSW-NB15-v2",
                "NF-CSE-CIC-IDS2018-v3",
                "NF-UNSW-NB15-v3",
            ],
            help="Dataset to use for training, validation and testing",
        )
        self.add_argument(
            "--fanout",
            type=int,
            default=-1,
            help="The number of edges to be sampled for each node",
        )
        self.add_argument(
            "--nhops",
            "-k",
            type=int,
            default=1,
            help="Number of hops for the graph neural network",
        )
        self.add_argument(
            "--edim_out",
            type=int,
            default=64,
            help="Output dimension of the edges features for the encoder",
        )
        self.add_argument(
            "--ae_embedding_dim",
            type=int,
            default=32,
            help="Dimension of the embeddings for the Transformer",
        )
        self.add_argument(
            "--dropout",
            type=float,
            default=0.0,
            help="Dropout rate for the GNN",
        )
        self.add_argument(
            "--ae_dropout",
            type=float,
            default=0.0,
            help="Dropout rate for the Transformer",
        )
        self.add_argument(
            "--retrain",
            "-r",
            action="store_true",
            help="If true, retrain the graph encoder",
        )
        self.add_argument(
            "--learning_rate",
            "-lr",
            type=float,
            default=5e-4,
            help="Learning rate for the optimizer",
        )
        self.add_argument(
            "--weight_decay",
            type=float,
            default=0.9,
            help="Weight decay for the encoder",
        )
        self.add_argument(
            "--ae_weight_decay",
            type=float,
            default=1e-2,
            help="Weight decay for the Transformer",
        )
        self.add_argument(
            "--num_epochs", type=int, default=100, help="Number of epochs to train for"
        )
        self.add_argument(
            "--fraction",
            type=float,
            default=None,
            help="Fraction of the dataset to use for training and testing",
        )
        self.add_argument(
            "--patience",
            type=int,
            default=30,
            help="Patience for early stopping of the GNN",
        )
        self.add_argument(
            "--positional_encoding",
            "-pe",
            type=str,
            default="None",
            choices=["None", "learnable", "sinusoidal"],
            help="Use positional encoding for the Transformer",
        )
        self.add_argument(
            "--agg_type",
            type=str,
            default="mean",
            choices=["mean"],
            help="Type of aggregation to use for the GNN",
        )
        self.add_argument(
            "--num_layers",
            type=int,
            default=1,
            help="Number of layers for Transformer",
        )
        self.add_argument(
            "--mask_ratio",
            type=float,
            default=0.0,
            help="Mask ratio for the Transformer",
        )
        self.add_argument(
            "--step_percent",
            type=float,
            default=1.0,
            help="Step percent for the Transformer",
        )
        self.add_argument(
            "--data_type",
            type=str,
            default="benign",
            choices=["benign", "mixed"],
            help="Type of training data to use (either 'benign' or 'mixed')",
        )
        self.add_argument(
            "--test", action="store_true", help="If true, don't train the model"
        )
        self.add_argument(
            "--reload_dataset",
            action="store_true",
            help="If true, force reload the dataset",
        )
        self.add_argument(
            "--batch_size",
            type=int,
            default=16384,
            help="Batch size for training the encoder",
        )
        self.add_argument(
            "--ae_batch_size",
            type=int,
            default=64,
            help="Batch size for the Transformer",
        )
        self.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Seed for reproducibility",
        )
        self.add_argument(
            "--wandb",
            action="store_true",
            help="If true, enable wandb online logging",
        )
        self.add_argument(
            "--save_curve",
            action="store_true",
            help="If true, save the curves to disk",
        )
        self.add_argument(
            "--window_size",
            type=int,
            default=512,
            help="Window size for the anomaly detection algorithm",
        )
