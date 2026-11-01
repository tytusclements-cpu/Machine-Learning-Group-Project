import pandas as pd

from spam_detection.config import LAST_NAME, OUTPUT_DIR
from spam_detection.data import load_test_messages, load_training_data
from spam_detection.evaluation import (
    CrossValidationResult,
    cross_validate,
    make_repeated_folds,
    metrics_by_source,
    select_final_model,
    summarize,
    validation_errors,
)
from spam_detection.models import CANDIDATES
from spam_detection.reporting import (
    print_section,
    save_confusion_matrix,
    save_predictions,
)


def compare_candidates(emails: pd.DataFrame) -> list[CrossValidationResult]:
    repeated_folds = make_repeated_folds(emails)
    return [cross_validate(candidate, emails, repeated_folds) for candidate in CANDIDATES]


def report_final_model(emails: pd.DataFrame, result: CrossValidationResult) -> None:
    by_source = metrics_by_source(emails, result.predictions)
    errors = validation_errors(emails, result.predictions)

    print_section("Validation metrics by source", by_source)
    print(f"\nValidation errors: {errors['error_type'].value_counts().to_dict()}")

    by_source.to_csv(OUTPUT_DIR / "final_model_metrics_by_source.csv")
    errors.to_csv(OUTPUT_DIR / "validation_errors.csv", index=False)
    save_confusion_matrix(
        emails["label"],
        result.predictions,
        f"{result.candidate.name}: out-of-fold predictions",
        OUTPUT_DIR / "confusion_matrix.png",
    )


def predict_test_set(emails: pd.DataFrame, result: CrossValidationResult) -> None:
    test_messages = load_test_messages()
    model = result.candidate.build().fit(emails["text"], emails["label"])
    save_predictions(model.predict(test_messages), OUTPUT_DIR / f"{LAST_NAME}_Spam.txt")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emails = load_training_data()
    print(f"Training emails after deduplication: {len(emails)}")

    results = compare_candidates(emails)
    comparison = summarize(results)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print_section("Model comparison", comparison.drop(columns="representation"))

    final_name = select_final_model(comparison)
    print(f"\nFinal model: {final_name}")
    final_result = next(r for r in results if r.candidate.name == final_name)

    report_final_model(emails, final_result)
    predict_test_set(emails, final_result)
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
