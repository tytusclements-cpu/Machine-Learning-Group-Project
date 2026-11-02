from spam_detection.config import LAST_NAME, RANDOM_SEED, ROOT

DATA_PATH = ROOT / "data" / "Dry_Bean_Clustering" / "Dry_Bean_Dataset.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "dry_bean"
ASSIGNMENTS_PATH = OUTPUT_DIR / f"{LAST_NAME}_DryBeanClusters.csv"

ID_COLUMN = "sample_id"
CLASS_COLUMN = "Class"
ASSIGNMENT_COLUMN = "Class_id"

SEEDS = range(RANDOM_SEED, RANDOM_SEED + 5)
N_CLUSTERS = 7
PCA_VARIANCE = 0.95
OUTLIER_Z_SCORE = 3
