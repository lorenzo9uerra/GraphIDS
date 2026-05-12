import math

import pytest
import torch
from torch_geometric.loader import LinkNeighborLoader

import utils.trainers as trainers_mod
from models.graphids import GraphIDS
from utils.trainers import find_threshold, train, validate
from utils.trainers import test as evaluate_test


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


def _make_model(dataset) -> GraphIDS:
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


def test_training_produces_checkpoint_and_logged_metrics(trained_bundle) -> None:
    checkpoint = trained_bundle["checkpoint"]
    threshold = trained_bundle["threshold"]
    logged_metrics = trained_bundle["run"].logged_metrics
    saved_threshold = torch.load(checkpoint, weights_only=True)["threshold"]

    assert checkpoint.exists()
    assert threshold is not None
    assert math.isfinite(float(threshold))
    assert saved_threshold == threshold

    assert logged_metrics
    latest_metrics = logged_metrics[-1]
    assert {"train_loss", "val_loss", "val_pr_auc"} <= set(latest_metrics)
    assert latest_metrics["train_loss"] >= 0.0
    assert latest_metrics["val_loss"] >= 0.0
    assert 0.0 <= latest_metrics["val_pr_auc"] <= 1.0


def test_validation_and_test_return_aligned_scores_and_metrics(
    trained_bundle,
) -> None:
    dataset = trained_bundle["dataset"]
    model = trained_bundle["model"]
    val_loader = trained_bundle["val_loader"]
    test_loader = trained_bundle["test_loader"]
    threshold = trained_bundle["threshold"]

    val_loss, val_errors, val_labels = validate(
        model,
        val_loader,
        ae_batch_size=2,
        window_size=4,
        device="cpu",
    )
    test_f1, test_pr_auc, test_errors, test_labels, prediction_time = evaluate_test(
        model,
        test_loader,
        ae_batch_size=2,
        window_size=4,
        device="cpu",
        threshold=threshold,
    )

    assert val_errors.shape == val_labels.shape
    assert val_errors.numel() == dataset.val_graph.edge_labels.numel()
    assert torch.equal(val_labels, dataset.val_graph.edge_labels.cpu())
    assert test_errors.shape == test_labels.shape
    assert test_errors.numel() == dataset.test_graph.edge_labels.numel()
    assert torch.equal(test_labels, dataset.test_graph.edge_labels.cpu())
    assert math.isfinite(val_loss)
    assert math.isfinite(test_f1)
    assert 0.0 <= test_pr_auc <= 1.0
    assert prediction_time >= 0.0


def test_find_threshold_supports_validation_f1_and_unsupervised_modes() -> None:
    errors = torch.tensor([0.1, 0.2, 0.8, 0.9], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    validation_threshold = find_threshold(errors, labels, method="validation_f1")
    validation_predictions = (errors > validation_threshold).int()
    unsupervised_threshold = find_threshold(errors, method="unsupervised")

    assert float(errors.min()) <= float(validation_threshold) <= float(errors.max())
    assert torch.equal(validation_predictions, labels)
    assert math.isfinite(float(unsupervised_threshold))
    assert float(unsupervised_threshold) >= float(errors.median())

    with pytest.raises(ValueError):
        find_threshold(errors, method="validation_f1")


def test_train_saves_checkpoint_on_tied_best_pr_auc(
    toy_dataset,
    monkeypatch,
    tmp_path,
) -> None:
    dataset = toy_dataset["dataset"]
    model = _make_model(dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run = DummyRun()

    train_loader = _make_loader(dataset.train_graph, batch_size=8, drop_last=True)
    val_loader = _make_loader(dataset.val_graph, batch_size=2, drop_last=False)
    scripted_scores = torch.tensor([0.1, 0.9], dtype=torch.float32)
    scripted_labels = torch.tensor([0, 1], dtype=torch.long)
    scripted_pr_aucs = iter([0.9, 0.9, 0.8])
    save_epochs: list[int] = []

    def fake_validate(*args, **kwargs):
        return 0.1, scripted_scores, scripted_labels

    original_save_checkpoint = model.save_checkpoint

    def recording_save_checkpoint(path, optimizer=None, epoch=0, threshold=None):
        save_epochs.append(int(epoch))
        original_save_checkpoint(
            path,
            optimizer=optimizer,
            epoch=epoch,
            threshold=threshold,
        )

    monkeypatch.setattr(trainers_mod, "validate", fake_validate)
    monkeypatch.setattr(
        trainers_mod,
        "average_precision_score",
        lambda *args, **kwargs: next(scripted_pr_aucs),
    )
    monkeypatch.setattr(model, "save_checkpoint", recording_save_checkpoint)

    train(
        model,
        window_size=4,
        step_percent=1.0,
        ae_batch_size=2,
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=0,
        num_epochs=3,
        optimizer=optimizer,
        run=run,
        patience=3,
        checkpoint=str(tmp_path / "strict-improvement.ckpt"),
        device="cpu",
    )

    assert save_epochs == [1, 2]
    assert len(run.logged_metrics) == 3


def test_train_early_stops_after_patience_non_improvements(
    toy_dataset,
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    dataset = toy_dataset["dataset"]
    model = _make_model(dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run = DummyRun()

    train_loader = _make_loader(dataset.train_graph, batch_size=8, drop_last=True)
    val_loader = _make_loader(dataset.val_graph, batch_size=2, drop_last=False)

    scripted_scores = torch.tensor([0.1, 0.9], dtype=torch.float32)
    scripted_labels = torch.tensor([0, 1], dtype=torch.long)
    scripted_pr_aucs = iter([0.9, 0.9, 0.8, 0.7])
    validate_calls = 0

    def fake_validate(*args, **kwargs):
        nonlocal validate_calls
        validate_calls += 1
        return 0.1, scripted_scores, scripted_labels

    monkeypatch.setattr(trainers_mod, "validate", fake_validate)
    monkeypatch.setattr(
        trainers_mod,
        "average_precision_score",
        lambda *args, **kwargs: next(scripted_pr_aucs),
    )

    train(
        model,
        window_size=4,
        step_percent=1.0,
        ae_batch_size=2,
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=0,
        num_epochs=5,
        optimizer=optimizer,
        run=run,
        patience=2,
        checkpoint=str(tmp_path / "early-stop.ckpt"),
        device="cpu",
    )
    captured = capsys.readouterr()

    assert "Early stopping!" in captured.out
    assert validate_calls == 3
    assert len(run.logged_metrics) == 3
