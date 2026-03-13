import pickle
import shutil
from pathlib import Path

import torch

from utils.dataloaders import NetFlowDataset


def test_preprocessing_pipeline_outputs_expected_graphs(toy_dataset) -> None:
    dataset = toy_dataset["dataset"]
    processed_dir = Path(dataset.processed_dir)
    scaler_path = toy_dataset["root"] / "scalers" / "scaler_ToyNF.pkl"

    assert dataset.train_graph.edge_labels.numel() == 8
    assert dataset.val_graph.edge_labels.numel() == 2
    assert dataset.test_graph.edge_labels.numel() == 2
    assert int(dataset.train_graph.edge_labels.sum().item()) == 0
    assert int(dataset.val_graph.edge_labels.sum().item()) == 1
    assert int(dataset.test_graph.edge_labels.sum().item()) == 1

    assert dataset.num_edge_features == 4
    assert dataset.num_node_features == 4
    assert dataset.num_nodes == 6
    assert dataset.train_graph.edge_attr.shape == (8, 4)
    assert torch.isfinite(dataset.train_graph.edge_attr).all()
    assert float(dataset.train_graph.edge_attr.min().item()) >= 0.0
    assert float(dataset.train_graph.edge_attr.max().item()) <= 1.000001
    assert torch.all(dataset.train_graph.x == 1)

    assert scaler_path.exists()
    assert (processed_dir / "train.pt").exists()
    assert (processed_dir / "val.pt").exists()
    assert (processed_dir / "test.pt").exists()
    assert (processed_dir / ".seed").read_text().strip() == "7"


def test_v3_preprocessing_excludes_timestamp_features_and_sorts_flows(
    toy_v3_dataset,
) -> None:
    dataset = toy_v3_dataset["dataset"]
    first_feature = dataset.train_graph.edge_attr[:, 0]

    assert dataset.train_graph.edge_labels.numel() == 8
    assert dataset.val_graph.edge_labels.numel() == 2
    assert dataset.test_graph.edge_labels.numel() == 2
    assert dataset.num_edge_features == 2
    assert dataset.num_node_features == 2
    assert dataset.num_nodes == 4
    assert torch.isfinite(dataset.train_graph.edge_attr).all()
    assert torch.all(torch.diff(first_feature) >= -1e-6)


def test_mixed_preprocessing_keeps_attack_edges_in_training_split(
    dataset_builder,
) -> None:
    dataset = dataset_builder(data_type="mixed", root_name="toy_graphids_mixed")[
        "dataset"
    ]

    assert dataset.train_graph.edge_labels.numel() == 16
    assert dataset.val_graph.edge_labels.numel() == 2
    assert dataset.test_graph.edge_labels.numel() == 2
    assert int(dataset.train_graph.edge_labels.sum().item()) == 8
    assert int(dataset.val_graph.edge_labels.sum().item()) == 1
    assert int(dataset.test_graph.edge_labels.sum().item()) == 1


def test_fraction_sampling_creates_fraction_specific_cache_and_balanced_splits(
    dataset_builder,
) -> None:
    bundle = dataset_builder(
        data_type="mixed",
        fraction=0.8,
        root_name="toy_graphids_fraction",
    )
    dataset = bundle["dataset"]

    assert Path(dataset.processed_dir).name == "ToyNF_0_8"
    assert dataset.train_graph.edge_labels.numel() == 12
    assert dataset.val_graph.edge_labels.numel() == 2
    assert dataset.test_graph.edge_labels.numel() == 2
    assert int(dataset.train_graph.edge_labels.sum().item()) == 6
    assert int(dataset.val_graph.edge_labels.sum().item()) == 1
    assert int(dataset.test_graph.edge_labels.sum().item()) == 1


def test_cached_dataset_warns_on_seed_mismatch_without_reprocessing(
    dataset_builder,
    monkeypatch,
    capsys,
) -> None:
    bundle = dataset_builder(root_name="toy_graphids_cache")
    dataset = bundle["dataset"]
    root = bundle["root"]
    data_dir = bundle["data_dir"]

    capsys.readouterr()
    monkeypatch.chdir(root)
    reloaded = NetFlowDataset(
        name="ToyNF",
        data_dir=str(data_dir),
        force_reload=False,
        data_type="benign",
        seed=11,
    )
    captured = capsys.readouterr()

    assert "Cached data was created with seed=7, but current seed=11" in captured.out
    assert "Run with --reload_dataset to recreate data with the new seed" in captured.out
    assert "Processing dataset ToyNF..." not in captured.out
    assert torch.equal(reloaded.train_graph.edge_labels, dataset.train_graph.edge_labels)
    assert torch.equal(reloaded.val_graph.edge_labels, dataset.val_graph.edge_labels)
    assert torch.equal(reloaded.test_graph.edge_labels, dataset.test_graph.edge_labels)


def test_corrupted_scaler_is_rebuilt_when_reprocessing(
    dataset_builder,
    monkeypatch,
    capsys,
) -> None:
    bundle = dataset_builder(root_name="toy_graphids_scaler")
    dataset = bundle["dataset"]
    root = bundle["root"]
    data_dir = bundle["data_dir"]
    processed_dir = Path(dataset.processed_dir)
    scaler_path = root / "scalers" / "scaler_ToyNF.pkl"

    scaler_path.write_bytes(b"not a pickle")
    shutil.rmtree(processed_dir)

    capsys.readouterr()
    monkeypatch.chdir(root)
    rebuilt = NetFlowDataset(
        name="ToyNF",
        data_dir=str(data_dir),
        force_reload=False,
        data_type="benign",
        seed=7,
    )
    captured = capsys.readouterr()

    assert "Failed to load scaler:" in captured.out
    assert "Processing dataset ToyNF..." in captured.out
    assert scaler_path.exists()
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)

    assert hasattr(scaler, "transform")
    assert rebuilt.train_graph.edge_labels.numel() == 8
