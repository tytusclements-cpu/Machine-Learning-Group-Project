from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dry_bean_clustering.config import SEEDS
from dry_bean_clustering.models import Configuration


@dataclass(frozen=True)
class ClusteringResult:
    configuration: Configuration
    silhouettes: list[float]
    stability: float
    labels: np.ndarray


def fit_clusters(
    configuration: Configuration, beans: pd.DataFrame, seed: int
) -> tuple[np.ndarray, float]:
    pipeline = configuration.build(seed)
    labels = pipeline.fit_predict(beans)
    return labels, silhouette_score(pipeline[:-1].transform(beans), labels)


def mean_pairwise_ari(labelings: list[np.ndarray]) -> float:
    return float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(labelings, 2)]))


def evaluate(configuration: Configuration, beans: pd.DataFrame) -> ClusteringResult:
    labelings, silhouettes = zip(*(fit_clusters(configuration, beans, seed) for seed in SEEDS))
    best_run = int(np.argmax(silhouettes))
    return ClusteringResult(
        configuration, list(silhouettes), mean_pairwise_ari(labelings), labelings[best_run]
    )


def summarize(results: list[ClusteringResult]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "configuration": result.configuration.name,
            "method": result.configuration.method,
            "feature_space": result.configuration.feature_space,
            "silhouette_mean": np.mean(result.silhouettes),
            "silhouette_std": np.std(result.silhouettes, ddof=1),
            "stability": result.stability,
        }
        for result in results
    )


def select_best(comparison: pd.DataFrame) -> str:
    scores = comparison["silhouette_mean"] + comparison["stability"]
    return comparison.loc[scores.idxmax(), "configuration"]


def explained_variance(beans: pd.DataFrame) -> pd.DataFrame:
    pca = make_pipeline(StandardScaler(), PCA(svd_solver="full")).fit(beans)[-1]
    return pd.DataFrame(
        {
            "component": np.arange(1, pca.n_components_ + 1),
            "explained_variance": pca.explained_variance_ratio_,
            "cumulative_variance": pca.explained_variance_ratio_.cumsum(),
        }
    )


def project_to_2d(beans: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2, svd_solver="full"))
    return pipeline.fit_transform(beans), pipeline[-1].explained_variance_ratio_


def profile_clusters(beans: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    groups = beans.groupby(labels)
    return groups.mean().assign(size=groups.size()).rename_axis("cluster")
