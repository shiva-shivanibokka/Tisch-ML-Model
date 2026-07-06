"""Load and clean the merged kidney scRNA-seq dataset."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config


def load_raw() -> pd.DataFrame:
    return pd.read_csv(config.RAW_CSV, low_memory=False)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.TARGET_COL] = df[config.TARGET_COL].replace(config.LABEL_MAP)
    counts = df[config.TARGET_COL].value_counts()
    keep = counts[counts >= config.MIN_CLASS_SIZE].index
    df = df[df[config.TARGET_COL].isin(keep)].copy()
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(str)
    return df


def select_top_classes(df: pd.DataFrame, n: int = config.TOP_N_CLASSES) -> pd.DataFrame:
    top = df[config.TARGET_COL].value_counts().head(n).index
    return df[df[config.TARGET_COL].isin(top)].copy()


def split_features_target(df: pd.DataFrame):
    gene_cols = [c for c in df.columns if c not in config.METADATA_COLS]
    return df[gene_cols].copy(), df[config.TARGET_COL].copy()


def subsample(X, y, size: int = config.SUBSET_SIZE, seed: int = config.RANDOM_SEED):
    if len(X) <= size:
        return X, y
    _, X_sub, _, y_sub = train_test_split(
        X, y, test_size=size, random_state=seed, stratify=y)
    return X_sub, y_sub


def make_split(X, y, test_size: float = config.TEST_SIZE, seed: int = config.RANDOM_SEED):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
