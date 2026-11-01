from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Spam_Email_Detection"
OUTPUT_DIR = ROOT / "outputs" / "spam"

LAST_NAME = "Johnson"
LABEL_ENCODING = {"ham": 0, "spam": 1}

RANDOM_SEED = 111111
N_SPLITS = 5
N_REPEATS = 5
