# Machine Learning Group Project

## Setup

[pip](https://pip.pypa.io/en/stable/):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Or, if you have [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Tested on `Python 3.14.8` with `matplotlib 3.11.2`, `numpy 2.5.4`, `openpyxl 3.1.5`, `pandas 3.1.0`, `pytest 9.2.0` and `scikit-learn 1.9.1`.

## Project 1: Spam Email Detection

To run:

```bash
.venv/bin/python -m spam_detection
```
or
```bash
uv run python -m spam_detection
```
To test:
```bash
.venv/bin/python -m pytest
```
or
```bash
uv run python -m pytest
```

The full run takes a few minutes. All randomness uses `RANDOM_SEED = 111111` from `spam_detection/config.py`.

### Expected outputs

The console prints the deduplicated training size, the model comparison, the final model and its results by source. These files are written to `outputs/spam/`:

| File | Contents |
| --- | --- |
| `model_comparison.csv` | Representation, hyperparameters, and mean and standard deviation of every metric for each model |
| `confusion_matrix.png` | Out-of-fold confusion matrix for the final model, from the first repeat |
| `final_model_metrics_by_source.csv` | Final model metrics for SMS and email separately |
| `validation_errors.csv` | Every out-of-fold false positive and false negative from the first repeat |
| `Johnson_Spam.txt` | 6,447 test predictions in original order, one per line, no header |

### Method

1. **Loading:** `spam_train1.csv` (SMS) has message text spilled into extra columns, so those columns are joined back together. `spam_train2.csv` holds email. Labels use the provided encoding: ham = 0, spam = 1.
2. **Duplicates:** the two training sets are combined, then duplicates are removed by comparing text with case and whitespace ignored. This happens before any split, reducing 4,296 messages to 4,178. No duplicates had conflicting labels.
3. **Validation:** 5-fold cross-validation is repeated 5 times with consecutive seeds starting at `RANDOM_SEED`, and every model uses the same splits. Folds are stratified on source and label, and reported means and standard deviations cover all 25 folds. Vectorizers are inside each pipeline, so vocabulary and TF-IDF weights are fitted on the training portion of each fold only.
4. **Selection:** the final model has the highest mean spam F1 minus mean ham false positive rate. This reflects the two grading criteria for the hidden test set, and the test set is not used for selection.
5. **Final predictions:** the selected configuration is refitted on all deduplicated training data and then predicts the test set.

## Project 2: Dry Bean Clustering

To run:
```bash
.venv/bin/python -m dry_bean_clustering
```
or
```bash
uv run python -m dry_bean_clustering
```

To test:
```bash
.venv/bin/python -m pytest
```
or
```bash
uv run python -m pytest
```

The full run takes about a minute. All randomness uses the five consecutive seeds starting at `RANDOM_SEED` from `spam_detection/config.py`.

### Expected outputs

The console prints the data checks, explained variance, configuration comparison, cluster sizes and final configuration. These files are written to `outputs/dry_bean/`:

| File | Contents |
| --- | --- |
| `data_quality.csv` | Sample, feature, missing value, duplicate and outlier counts |
| `pca_explained_variance.csv` | Explained and cumulative variance for each principal component |
| `configuration_comparison.csv` | Silhouette mean and standard deviation, and stability, for all four configurations |
| `kmeans_cluster_profiles.csv`, `gaussian_mixture_cluster_profiles.csv` | Size and mean of each original feature per cluster, for each method's selected configuration |
| `kmeans_pca_clusters.png`, `gaussian_mixture_pca_clusters.png` | 2D PCA view of each method's selected configuration |
| `Johnson_DryBeanClusters.csv` | `sample_id` and `Class_id` for all 13,611 samples |

### Method

1. **Data:** `Class` is dropped on load, so it can't affect preprocessing, PCA, clustering or selection. There are no missing values.
2. **Unusual records:** the 68 duplicate rows and 1,124 rows with a feature beyond 3 standard deviations are kept, because every sample needs a cluster ID.
3. **Feature spaces:** every method runs on standardized features and on standardized features reduced by PCA. PCA keeps the fewest components that explain at least 95% of the variance, which is 4 components (95.0%).
4. **Clustering:** K-means and a full-covariance Gaussian mixture, both with 7 clusters, give four configurations.
5. **Evaluation:** each configuration runs on 5 seeds. Silhouette is measured in the space used for clustering, and stability is the mean pairwise Adjusted Rand Index between the seeds' assignments.
6. **Selection:** the best configuration has the highest mean silhouette plus stability. Its best-silhouette run gives the final assignments, and `Class` is empty in the provided data, so ARI against the reference labels can't be used.
7. **Visualization:** the 2D PCA plots are for viewing only and are separate from the PCA used for clustering. They can hide separation that exists in the clustering space.

## Repository layout

```
data/                  Provided datasets
spam_detection/        Project 1 code
dry_bean_clustering/   Project 2 code
tests/                 Unit tests
outputs/               Generated results (after runs)
```
