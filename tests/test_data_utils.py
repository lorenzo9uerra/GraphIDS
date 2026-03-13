import torch

from utils.dataloaders import SequentialDataset, collate_fn
from utils.trainers import calculate_errors
from utils.trainers import test as evaluate_test


def test_sequential_dataset_and_collate_handle_partial_tail_windows() -> None:
    data = torch.arange(10, dtype=torch.float32).view(5, 2)
    dataset = SequentialDataset(data, window=3, step=2, device="cpu")

    first_window, first_mask = dataset[0]
    tail_window, tail_mask = dataset[2]
    batch, mask = collate_fn([dataset[0], dataset[2]])

    assert len(dataset) == 3
    assert first_window.shape == (3, 2)
    assert tail_window.shape == (1, 2)
    assert torch.equal(first_mask, torch.ones_like(first_window, dtype=torch.bool))
    assert torch.equal(tail_mask, torch.ones_like(tail_window, dtype=torch.bool))
    assert batch.shape == (2, 3, 2)
    assert mask.shape == (2, 3, 2)
    assert torch.equal(batch[1, 0], data[-1])
    assert torch.equal(batch[1, 1:], torch.zeros((2, 2)))
    assert torch.equal(
        mask[1],
        torch.tensor(
            [[True, True], [False, False], [False, False]],
            dtype=torch.bool,
        ),
    )


def test_calculate_errors_ignores_padding_and_sanitizes_non_finite_values() -> None:
    outputs = torch.tensor(
        [
            [[1.0, 1.0], [float("nan"), 1.0], [5.0, float("inf")]],
            [[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    batch = torch.tensor(
        [
            [[1.0, 3.0], [1.0, 1.0], [4.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [[True, True], [True, True], [True, True]],
            [[True, True], [False, False], [False, False]],
        ],
        dtype=torch.bool,
    )

    errors = calculate_errors(outputs, batch, mask)

    assert errors.shape == (4,)
    assert torch.equal(
        errors,
        torch.tensor([2.0, 0.0, 1_000_000.0, 2.5], dtype=torch.float32),
    )


def test_evaluation_without_threshold_uses_mean_error_fallback(
    trained_bundle,
    capsys,
) -> None:
    model = trained_bundle["model"]
    test_loader = trained_bundle["test_loader"]

    test_f1, test_pr_auc, test_errors, test_labels, prediction_time = evaluate_test(
        model,
        test_loader,
        ae_batch_size=2,
        window_size=4,
        device="cpu",
        threshold=None,
    )
    captured = capsys.readouterr()

    assert "No threshold provided, using mean of errors for prediction." in captured.out
    assert test_errors.shape == test_labels.shape
    assert test_errors.numel() == trained_bundle["dataset"].test_graph.edge_labels.numel()
    assert 0.0 <= test_pr_auc <= 1.0
    assert prediction_time >= 0.0
    assert test_f1 >= 0.0
