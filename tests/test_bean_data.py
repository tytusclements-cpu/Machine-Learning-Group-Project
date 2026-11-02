import pandas as pd

from dry_bean_clustering.config import CLASS_COLUMN
from dry_bean_clustering.data import load_beans, summarize_quality


def test_load_beans_keeps_every_sample_and_only_features():
    beans = load_beans()
    assert beans.shape == (13611, 16)
    assert beans.index.is_unique
    assert CLASS_COLUMN not in beans.columns


def test_summarize_quality_counts_missing_values_and_duplicates():
    beans = pd.DataFrame({"a": [1.0, 1.0, 2.0, None], "b": [3.0, 3.0, 4.0, 5.0]})
    quality = summarize_quality(beans)
    assert quality["missing_values"] == 1
    assert quality["duplicate_rows"] == 1
