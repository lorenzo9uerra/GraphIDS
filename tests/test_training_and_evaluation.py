import math

import pytest
import torch

from utils.trainers import find_threshold, validate
from utils.trainers import test as evaluate_test


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
    assert {"train_loss", "val_loss", "val_pr_auc", "test_f1", "test_pr_auc"} <= set(
        latest_metrics
    )
    assert latest_metrics["train_loss"] >= 0.0
    assert latest_metrics["val_loss"] >= 0.0
    assert 0.0 <= latest_metrics["val_pr_auc"] <= 1.0
    assert 0.0 <= latest_metrics["test_pr_auc"] <= 1.0


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


def test_find_threshold_supports_supervised_and_unsupervised_modes() -> None:
    errors = torch.tensor([0.1, 0.2, 0.8, 0.9], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    supervised_threshold = find_threshold(errors, labels, method="supervised")
    supervised_predictions = (errors > supervised_threshold).int()
    unsupervised_threshold = find_threshold(errors, method="unsupervised")

    assert float(errors.min()) <= float(supervised_threshold) <= float(errors.max())
    assert torch.equal(supervised_predictions, labels)
    assert math.isfinite(float(unsupervised_threshold))
    assert float(unsupervised_threshold) >= float(errors.median())

    with pytest.raises(ValueError):
        find_threshold(errors, method="supervised")
