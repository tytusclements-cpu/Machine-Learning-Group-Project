import pandas as pd

from spam_detection.data import (
    deduplicate,
    join_spilled_text,
    load_test_messages,
    load_training_data,
)


def test_join_spilled_text_restores_commas_and_skips_empty_cells():
    row = pd.Series(["Hello", " world", "", ""])
    assert join_spilled_text(row) == "Hello, world"


def test_deduplicate_ignores_case_and_whitespace():
    emails = pd.DataFrame({"text": ["Win  NOW", "win now\r\n", "Lunch?"], "label": [1, 1, 0]})
    assert deduplicate(emails)["text"].tolist() == ["Win  NOW", "Lunch?"]


def test_training_data_is_fully_labeled():
    emails = load_training_data()
    assert set(emails["label"]) == {0, 1}
    assert set(emails["source"]) == {"sms", "email"}


def test_test_set_keeps_every_row():
    assert len(load_test_messages()) == 6447
