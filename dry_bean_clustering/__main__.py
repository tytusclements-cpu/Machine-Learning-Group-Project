import pandas as pd

from dry_bean_clustering.config import ASSIGNMENTS_PATH, OUTPUT_DIR
from dry_bean_clustering.data import load_beans, summarize_quality
from dry_bean_clustering.evaluation import (
    ClusteringResult,
    evaluate,
    explained_variance,
    profile_clusters,
    project_to_2d,
    select_best,
    summarize,
)
from dry_bean_clustering.models import CONFIGURATIONS
from dry_bean_clustering.reporting import save_assignments, save_cluster_plot
from spam_detection.reporting import print_section


def report_data(beans: pd.DataFrame) -> None:
    quality = summarize_quality(beans)
    variance = explained_variance(beans)

    print(f"Data quality: {quality.to_dict()}")
    print_section("PCA explained variance", variance.head(6))

    quality.to_csv(OUTPUT_DIR / "data_quality.csv", header=["count"])
    variance.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)


def report_method(beans: pd.DataFrame, result: ClusteringResult) -> None:
    method = result.configuration.method
    profile = profile_clusters(beans, result.labels)
    points, variance = project_to_2d(beans)

    print_section(f"Cluster sizes: {result.configuration.name}", profile[["size"]])

    profile.to_csv(OUTPUT_DIR / f"{method}_cluster_profiles.csv")
    save_cluster_plot(
        points,
        variance,
        result.labels,
        f"{result.configuration.name}: 2D PCA view",
        OUTPUT_DIR / f"{method}_pca_clusters.png",
    )


def select_result(results: list[ClusteringResult], comparison: pd.DataFrame) -> ClusteringResult:
    name = select_best(comparison)
    return next(r for r in results if r.configuration.name == name)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    beans = load_beans()
    report_data(beans)

    results = [evaluate(configuration, beans) for configuration in CONFIGURATIONS]
    comparison = summarize(results)
    comparison.to_csv(OUTPUT_DIR / "configuration_comparison.csv", index=False)
    print_section("Configuration comparison", comparison.drop(columns=["method", "feature_space"]))

    for _, method_comparison in comparison.groupby("method"):
        report_method(beans, select_result(results, method_comparison))

    final_result = select_result(results, comparison)
    print(f"\nFinal configuration: {final_result.configuration.name}")
    save_assignments(beans.index, final_result.labels, ASSIGNMENTS_PATH)
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
