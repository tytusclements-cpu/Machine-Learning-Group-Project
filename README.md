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

## Project 1: Spam Email Detection

Run these from the repository root:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m spam_detection
```

Or with uv:

```bash
uv run python -m pytest
uv run python -m spam_detection
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

## Repository layout

```
data/              Provided datasets
spam_detection/    Project 1 code
tests/             Unit tests
outputs/           Generated results (after runs)
```
