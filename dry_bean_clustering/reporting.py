from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dry_bean_clustering.config import ASSIGNMENT_COLUMN, ID_COLUMN

CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def save_cluster_plot(
    points: np.ndarray, variance: np.ndarray, labels: np.ndarray, title: str, path: Path
) -> None:
    figure, axes = plt.subplots(figsize=(7, 5))
    for cluster, color in zip(np.unique(labels), CLUSTER_COLORS):
        members = points[labels == cluster]
        axes.scatter(*members.T, s=3, color=color, alpha=0.6, label=f"Cluster {cluster}")
    axes.set_title(title)
    axes.set_xlabel(f"PC1 ({variance[0]:.1%} of variance)")
    axes.set_ylabel(f"PC2 ({variance[1]:.1%} of variance)")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(markerscale=3, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_assignments(sample_ids: pd.Index, labels: np.ndarray, path: Path) -> None:
    assignments = pd.DataFrame({ID_COLUMN: sample_ids, ASSIGNMENT_COLUMN: labels})
    assignments.to_csv(path, index=False)
