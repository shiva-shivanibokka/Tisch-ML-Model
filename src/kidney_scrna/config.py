"""Central configuration: paths, constants, and label maps."""
from __future__ import annotations
import os
from pathlib import Path

RANDOM_SEED = 42
SUBSET_SIZE = 20_000
CAP = 1000
TEST_SIZE = 0.20
TOP_N_CLASSES = 10
MIN_CLASS_SIZE = 100
VT_THRESHOLD = 0.01
NULL_THRESHOLD = 0.90

METADATA_COLS = ["Cell_ID", "nCount_RNA", "nFeature_RNA", "StudyOrigin_Author",
                 "percent.mt", "Sex", "Sampling_Location", "Age", "Cell_Labels"]
TARGET_COL = "Cell_Labels"
LABEL_MAP = {"PT": "Proximal Tubule", "DT": "Distal Convoluted Tubule",
             "LH": "Loop of Henle and Parietal Epithelium",
             "PC": "Collecting Duct Principal", "IC": "Collecting Duct Intercalated",
             "P": "Glomerular Epithelium and Podocytes"}

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "Tisch24_MergedscRNA_80-85PctVAR.csv"
# Artifacts live at <repo>/artifacts by default; override for non-editable installs
# or custom deploys with KIDNEY_ARTIFACTS_DIR.
ARTIFACTS_DIR = Path(os.environ.get("KIDNEY_ARTIFACTS_DIR", ROOT / "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
EXAMPLES_PATH = ARTIFACTS_DIR / "examples.json"


def ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    return (f"kidney_scrna | seed={RANDOM_SEED} subset={SUBSET_SIZE} "
            f"cap={CAP} top_n={TOP_N_CLASSES}")
