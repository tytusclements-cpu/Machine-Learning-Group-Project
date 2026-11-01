from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from spam_detection.config import N_REPEATS, N_SPLITS, RANDOM_SEED
from spam_detection.models import Candidate

Fold = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class CrossValidationResult:
    candidate: Candidate
    fold_metrics: pd.DataFrame
    predictions: np.ndarray


def ham_false_positive_rate(y_true, y_pred) -> float:
    true_negatives, false_positives, _, _ = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return false_positives / (false_positives + true_negatives)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "spam_precision": precision_score(y_true, y_pred, zero_division=0),
        "spam_recall": recall_score(y_true, y_pred, zero_division=0),
        "spam_f1": f1_score(y_true, y_pred, zero_division=0),
        "ham_fpr": ham_false_positive_rate(y_true, y_pred),
    }


def make_folds(emails: pd.DataFrame, seed: int) -> list[Fold]:
    strata = emails["source"] + "_" + emails["label"].astype(str)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return list(splitter.split(emails, strata))


def make_repeated_folds(emails: pd.DataFrame) -> list[list[Fold]]:
    seeds = range(RANDOM_SEED, RANDOM_SEED + N_REPEATS)
    return [make_folds(emails, seed) for seed in seeds]


def run_folds(
    candidate: Candidate, emails: pd.DataFrame, folds: list[Fold]
) -> tuple[pd.DataFrame, np.ndarray]:
    texts, labels = emails["text"], emails["label"]
    predictions = np.empty(len(emails), dtype=int)
    fold_metrics = []

    for train_index, validation_index in folds:
        model = candidate.build().fit(texts.iloc[train_index], labels.iloc[train_index])
        predictions[validation_index] = model.predict(texts.iloc[validation_index])
        fold_metrics.append(
            compute_metrics(labels.iloc[validation_index], predictions[validation_index])
        )

    return pd.DataFrame(fold_metrics), predictions


def cross_validate(
    candidate: Candidate, emails: pd.DataFrame, repeated_folds: list[list[Fold]]
) -> CrossValidationResult:
    runs = [run_folds(candidate, emails, folds) for folds in repeated_folds]
    fold_metrics = pd.concat([metrics for metrics, _ in runs], ignore_index=True)
    return CrossValidationResult(candidate, fold_metrics, runs[0][1])


def summarize(results: list[CrossValidationResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        means = result.fold_metrics.mean().add_suffix("_mean")
        stds = result.fold_metrics.std().add_suffix("_std")
        rows.append(
            {
                "model": result.candidate.name,
                "representation": result.candidate.representation,
                "hyperparameters": result.candidate.hyperparameters,
                **means,
                **stds,
            }
        )
    return pd.DataFrame(rows)


def select_final_model(comparison: pd.DataFrame) -> str:
    scores = comparison["spam_f1_mean"] - comparison["ham_fpr_mean"]
    return comparison.loc[scores.idxmax(), "model"]


def metrics_by_source(emails: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            source: compute_metrics(emails["label"][mask], predictions[mask])
            for source, mask in emails.groupby("source").groups.items()
        }
    ).T.rename_axis("source")


def validation_errors(emails: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    is_wrong = emails["label"] != predictions
    errors = emails[is_wrong].copy()
    errors["error_type"] = np.where(errors["label"] == 0, "false_positive", "false_negative")
    return errors[["error_type", "source", "text"]].sort_values(["error_type", "source"])
