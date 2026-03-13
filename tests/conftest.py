import csv
import os
import warnings
from pathlib import Path

import pytest
import torch
from torch_geometric.loader import LinkNeighborLoader

from models.graphids import GraphIDS
from utils.dataloaders import NetFlowDataset
from utils.trainers import train

warnings.filterwarnings(
    "ignore", message="The PyTorch API of nested tensors is in prototype stage"
)


class DummyRun:
    def __init__(self) -> None:
        self.logged_metrics: list[dict[str, float]] = []

    def log(self, metrics: dict[str, float]) -> None:
        numeric_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float)
        }
        self.logged_metrics.append(numeric_metrics)


def _write_toy_netflow_csv(csv_path: Path) -> None:
    fieldnames = [
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "Attack",
        "Label",
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "IN_BYTES",
        "IN_PKTS",
    ]
    ips = [f"10.0.0.{idx}" for idx in range(1, 7)]
    rows: list[dict[str, str | int | float]] = []
    for idx in range(20):
        label = 1 if idx % 2 else 0
        rows.append(
            {
                "IPV4_SRC_ADDR": ips[idx % len(ips)],
                "IPV4_DST_ADDR": ips[(idx + 1) % len(ips)],
                "Attack": "Malicious" if label else "Benign",
                "Label": label,
                "L4_SRC_PORT": 10_000 + idx,
                "L4_DST_PORT": 20_000 + (idx % 4),
                "IN_BYTES": 100.0 + 5.0 * idx + 500.0 * label,
                "IN_PKTS": 1.0 + (idx % 3) + 2.0 * label,
            }
        )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_toy_v3_netflow_csv(csv_path: Path) -> None:
    fieldnames = [
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "Attack",
        "Label",
        "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS",
        "IN_BYTES",
        "IN_PKTS",
    ]
    ips = [f"10.1.0.{idx}" for idx in range(1, 5)]
    rows: list[dict[str, str | int | float]] = []
    for idx in range(20):
        label = 1 if idx % 2 else 0
        start_ms = 1_700_000_000_000 + idx * 1_000
        rows.append(
            {
                "IPV4_SRC_ADDR": ips[idx % len(ips)],
                "IPV4_DST_ADDR": ips[(idx + 1) % len(ips)],
                "Attack": "Malicious" if label else "Benign",
                "Label": label,
                "FLOW_START_MILLISECONDS": start_ms,
                "FLOW_END_MILLISECONDS": start_ms + 500,
                "IN_BYTES": 50.0 + 10.0 * idx,
                "IN_PKTS": 1.0 + idx,
            }
        )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_dataset(
    root: Path,
    dataset_name: str,
    csv_writer,
    **dataset_kwargs,
) -> dict[str, Path | NetFlowDataset]:
    data_dir = root / "data"
    raw_dir = data_dir / dataset_name
    raw_dir.mkdir(parents=True)
    csv_writer(raw_dir / f"{dataset_name}.csv")

    previous_cwd = Path.cwd()
    os.chdir(root)
    try:
        dataset_init_kwargs = {
            "name": dataset_name,
            "data_dir": str(data_dir),
            "force_reload": True,
            "data_type": "benign",
            "seed": 7,
        }
        dataset_init_kwargs.update(dataset_kwargs)
        dataset = NetFlowDataset(**dataset_init_kwargs)
    finally:
        os.chdir(previous_cwd)

    return {
        "root": root,
        "data_dir": data_dir,
        "dataset": dataset,
    }


def _make_model(dataset: NetFlowDataset) -> GraphIDS:
    torch.manual_seed(0)
    return GraphIDS(
        ndim_in=dataset.num_node_features,
        edim_in=dataset.num_edge_features,
        edim_out=8,
        embed_dim=4,
        num_heads=2,
        num_layers=1,
        window_size=4,
        dropout=0.0,
        ae_dropout=0.0,
        positional_encoding="None",
        agg_type="mean",
        mask_ratio=0.0,
    )


def _make_loader(graph, *, batch_size: int, drop_last: bool):
    return LinkNeighborLoader(
        data=graph,
        num_neighbors=[-1],
        edge_label_index=graph.edge_index,
        edge_label=graph.edge_labels,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=drop_last,
    )


@pytest.fixture(scope="module")
def toy_dataset(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("toy_graphids")
    return _build_dataset(root, "ToyNF", _write_toy_netflow_csv)


@pytest.fixture(scope="module")
def toy_v3_dataset(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("toy_graphids_v3")
    return _build_dataset(root, "ToyNF-v3", _write_toy_v3_netflow_csv)


@pytest.fixture
def dataset_builder(tmp_path: Path):
    def build(
        *,
        dataset_name: str = "ToyNF",
        csv_writer=_write_toy_netflow_csv,
        root_name: str = "toy_graphids_custom",
        **dataset_kwargs,
    ):
        return _build_dataset(
            tmp_path / root_name,
            dataset_name,
            csv_writer,
            **dataset_kwargs,
        )

    return build


@pytest.fixture(scope="module")
def trained_bundle(toy_dataset):
    dataset = toy_dataset["dataset"]
    model = _make_model(dataset)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "weight_decay": 0.0},
            {"params": model.transformer.parameters(), "weight_decay": 0.0},
        ],
        lr=1e-3,
    )
    run = DummyRun()
    checkpoint = toy_dataset["root"] / "toy_graphids.ckpt"

    train_loader = _make_loader(dataset.train_graph, batch_size=8, drop_last=True)
    val_loader = _make_loader(dataset.val_graph, batch_size=2, drop_last=False)
    test_loader = _make_loader(dataset.test_graph, batch_size=2, drop_last=False)

    model, threshold = train(
        model,
        window_size=4,
        step_percent=1.0,
        ae_batch_size=2,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        start_epoch=0,
        num_epochs=1,
        optimizer=optimizer,
        run=run,
        patience=1,
        checkpoint=str(checkpoint),
        device="cpu",
    )

    return {
        "dataset": dataset,
        "model": model,
        "checkpoint": checkpoint,
        "threshold": threshold,
        "run": run,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
