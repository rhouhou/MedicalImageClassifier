import numpy as np

from src.metrics import compute_metrics


def test_compute_metrics_returns_expected_keys():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = compute_metrics(y_true, y_prob)

    expected_keys = {
        "accuracy",
        "auc",
        "pr_auc",
        "brier",
        "confusion_matrix",
    }

    assert expected_keys.issubset(metrics.keys())
    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)