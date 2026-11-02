import pandas as pd

from dry_bean_clustering.config import CLASS_COLUMN, DATA_PATH, ID_COLUMN, OUTLIER_Z_SCORE


def load_beans() -> pd.DataFrame:
    beans = pd.read_excel(DATA_PATH, index_col=ID_COLUMN)
    return beans.drop(columns=CLASS_COLUMN)


def summarize_quality(beans: pd.DataFrame) -> pd.Series:
    z_scores = (beans - beans.mean()) / beans.std()
    return pd.Series(
        {
            "samples": len(beans),
            "features": beans.shape[1],
            "missing_values": beans.isna().sum().sum(),
            "duplicate_rows": beans.duplicated().sum(),
            "rows_with_outlier": z_scores.abs().gt(OUTLIER_Z_SCORE).any(axis=1).sum(),
        }
    )
