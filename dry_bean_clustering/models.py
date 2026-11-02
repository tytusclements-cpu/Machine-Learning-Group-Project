from dataclasses import dataclass

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from dry_bean_clustering.config import N_CLUSTERS, PCA_VARIANCE


def kmeans(seed: int) -> KMeans:
    return KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=seed)


def gaussian_mixture(seed: int) -> GaussianMixture:
    return GaussianMixture(n_components=N_CLUSTERS, n_init=3, random_state=seed)


def standardized() -> list:
    return [StandardScaler()]


def standardized_pca() -> list:
    return [StandardScaler(), PCA(n_components=PCA_VARIANCE, svd_solver="full")]


METHODS = {"kmeans": kmeans, "gaussian_mixture": gaussian_mixture}
FEATURE_SPACES = {"standardized": standardized, "standardized_pca": standardized_pca}


@dataclass(frozen=True)
class Configuration:
    method: str
    feature_space: str

    @property
    def name(self) -> str:
        return f"{self.method}_{self.feature_space}"

    def build(self, seed: int) -> Pipeline:
        return make_pipeline(*FEATURE_SPACES[self.feature_space](), METHODS[self.method](seed))


CONFIGURATIONS = [
    Configuration(method, feature_space) for method in METHODS for feature_space in FEATURE_SPACES
]
