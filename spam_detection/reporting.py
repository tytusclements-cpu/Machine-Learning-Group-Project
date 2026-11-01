from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["Ham", "Spam"],
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=200)
    plt.close(display.figure_)


def save_predictions(predictions: Iterable[int], path: Path) -> None:
    path.write_text("".join(f"{label}\n" for label in predictions))


def print_section(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}\n{table.round(4).to_string()}")
