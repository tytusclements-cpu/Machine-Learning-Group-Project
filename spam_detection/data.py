import pandas as pd

from spam_detection.config import DATA_DIR, LABEL_ENCODING


def read_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename, dtype=str, keep_default_na=False)


def join_spilled_text(row: pd.Series) -> str:
    return ",".join(cell for cell in row if cell)


def load_sms_training() -> pd.DataFrame:
    raw = read_csv("spam_train1.csv")
    text_columns = [column for column in raw.columns if column != "v1"]
    return pd.DataFrame(
        {
            "text": raw[text_columns].apply(join_spilled_text, axis=1),
            "label": raw["v1"].map(LABEL_ENCODING),
            "source": "sms",
        }
    )


def load_email_training() -> pd.DataFrame:
    raw = read_csv("spam_train2.csv")
    return pd.DataFrame(
        {
            "text": raw["text"],
            "label": raw["label"].map(LABEL_ENCODING),
            "source": "email",
        }
    )


def load_test_messages() -> pd.Series:
    return read_csv("spam_test.csv")["message"]


def duplicate_key(texts: pd.Series) -> pd.Series:
    return texts.str.casefold().str.split().str.join(" ")


def deduplicate(emails: pd.DataFrame) -> pd.DataFrame:
    is_duplicate = duplicate_key(emails["text"]).duplicated()
    return emails[~is_duplicate].reset_index(drop=True)


def load_training_data() -> pd.DataFrame:
    combined = pd.concat([load_sms_training(), load_email_training()], ignore_index=True)
    return deduplicate(combined)
