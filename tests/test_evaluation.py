import numpy as np
import pandas as pd
import pytest

from spam_detection.data import duplicate_key, load_training_data
from spam_detection.evaluation import (
    compute_metrics,
    make_repeated_folds,
    select_final_model,
    validation_errors,
)


def test_compute_metrics_on_known_predictions():
    metrics = compute_metrics([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0])
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["spam_precision"] == pytest.approx(0.5)
    assert metrics["spam_recall"] == pytest.approx(0.5)
    assert metrics["ham_fpr"] == pytest.approx(0.25)


def test_no_duplicate_crosses_a_fold_boundary():
    emails = load_training_data()
    keys = duplicate_key(emails["text"])
    for folds in make_repeated_folds(emails):
        for train_index, validation_index in folds:
            assert not set(keys.iloc[train_index]) & set(keys.iloc[validation_index])


def test_selection_maximizes_spam_f1_minus_false_positive_rate():
    comparison = pd.DataFrame(
        {
            "model": ["high_fpr", "balanced", "low_recall"],
            "spam_f1_mean": [0.96, 0.95, 0.85],
            "ham_fpr_mean": [0.05, 0.011, 0.002],
        }
    )
    assert select_final_model(comparison) == "balanced"


def test_validation_errors_are_labeled_by_type():
    emails = pd.DataFrame(
        {"text": ["ok", "fp", "fn"], "label": [0, 0, 1], "source": ["sms"] * 3}
    )
    errors = validation_errors(emails, np.array([0, 1, 0]))
    assert dict(zip(errors["text"], errors["error_type"])) == {
        "fp": "false_positive",
        "fn": "false_negative",
    }
