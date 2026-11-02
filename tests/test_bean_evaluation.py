import numpy as np
import pandas as pd
import pytest

from dry_bean_clustering.config import ASSIGNMENT_COLUMN, ID_COLUMN
from dry_bean_clustering.evaluation import (
    explained_variance,
    mean_pairwise_ari,
    profile_clusters,
    project_to_2d,
    select_best,
)
from dry_bean_clustering.models import CONFIGURATIONS
from dry_bean_clustering.reporting import save_assignments


@pytest.fixture
def beans():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))


def test_configurations_cover_both_methods_and_feature_spaces():
    assert len({c.name for c in CONFIGURATIONS}) == 4


def test_mean_pairwise_ari_ignores_cluster_numbering():
    labelings = [np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])]
    assert mean_pairwise_ari(labelings) == pytest.approx(1)


def test_select_best_combines_silhouette_and_stability():
    comparison = pd.DataFrame(
        {
            "configuration": ["unstable", "balanced", "weak"],
            "silhouette_mean": [0.40, 0.33, 0.20],
            "stability": [0.70, 0.99, 1.00],
        }
    )
    assert select_best(comparison) == "balanced"


def test_explained_variance_accumulates_to_one(beans):
    assert explained_variance(beans)["cumulative_variance"].iloc[-1] == pytest.approx(1)


def test_project_to_2d_returns_two_coordinates_per_sample(beans):
    points, variance = project_to_2d(beans)
    assert points.shape == (60, 2)
    assert len(variance) == 2


def test_profile_clusters_reports_sizes():
    beans = pd.DataFrame({"a": [1.0, 3.0, 5.0]})
    profile = profile_clusters(beans, np.array([0, 0, 1]))
    assert profile["size"].tolist() == [2, 1]
    assert profile["a"].tolist() == [2.0, 5.0]


def test_save_assignments_writes_required_columns(tmp_path):
    path = tmp_path / "assignments.csv"
    save_assignments(pd.Index([1, 2]), np.array([3, 4]), path)
    saved = pd.read_csv(path)
    assert saved.columns.tolist() == [ID_COLUMN, ASSIGNMENT_COLUMN]
    assert saved[ASSIGNMENT_COLUMN].tolist() == [3, 4]
