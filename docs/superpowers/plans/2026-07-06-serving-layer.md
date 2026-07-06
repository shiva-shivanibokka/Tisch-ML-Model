# Serving Layer + Interactive Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `src/kidney_scrna` package, a headless `train.py` that exports a deployable model, a FastAPI serving layer with an interactive multiclass demo, tests, and Docker + Fly.io deploy config — mirroring the Cumida-ML-Model repo, then deploy live.

**Architecture:** Extract the four notebooks' pipeline into a small importable package (config/data/features/models/evaluate/serve). `train.py` runs it headless and saves `artifacts/{model.joblib, metrics.json, examples.json}`. `serve.py` (FastAPI) loads the model bundle and serves an interactive demo + JSON API. Docker image bakes the artifacts; Fly.io hosts it.

**Tech Stack:** Python 3.10+, scikit-learn, imbalanced-learn, scikit-optimize, FastAPI, uvicorn, pydantic, joblib, pytest, Docker, Fly.io.

## Global Constraints

- Python 3.10+; CPU-only; no cuML/GPU/Colab.
- `RANDOM_SEED = 42`, `SUBSET_SIZE = 20_000`, `CAP = 1000`, top-10 classes.
- Data-leakage rule preserved: all feature reduction fit on train only; SMOTE + RandomUnderSampler only inside CV folds / training, never on test.
- Dataset CSV (`Tisch24_MergedscRNA_80-85PctVAR.csv`, 292 MB) stays git-ignored and is never copied into the Docker image.
- No mention of Claude/AI anywhere in code, comments, commits, or docs.
- Commit to `main`. Package importable as `kidney_scrna`.

---

### Task 1: Package scaffold + config

**Files:**
- Create: `pyproject.toml`
- Create: `src/kidney_scrna/__init__.py`
- Create: `src/kidney_scrna/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.RANDOM_SEED`, `config.SUBSET_SIZE`, `config.CAP`, `config.TOP_N_CLASSES`, `config.MIN_CLASS_SIZE`, `config.METADATA_COLS`, `config.TARGET_COL`, `config.LABEL_MAP`, paths `RAW_CSV`, `ARTIFACTS_DIR`, `MODEL_PATH`, `METRICS_PATH`, `EXAMPLES_PATH`; functions `ensure_artifacts_dir()`, `describe() -> str`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "kidney-scrna"
version = "1.0.0"
description = "Kidney scRNA-seq cell-type classifier: pipeline, serving API, and demo."
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0", "numpy>=1.24", "scikit-learn>=1.3", "scipy>=1.10",
    "imbalanced-learn>=0.11", "scikit-optimize>=0.9", "joblib>=1.3",
    "fastapi>=0.110", "uvicorn[standard]>=0.29", "pydantic>=2.6",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]
notebooks = ["matplotlib>=3.7", "seaborn>=0.12"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Write `src/kidney_scrna/config.py`**

```python
"""Central configuration: paths, constants, and label maps."""
from __future__ import annotations
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
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
EXAMPLES_PATH = ARTIFACTS_DIR / "examples.json"


def ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    return (f"kidney_scrna | seed={RANDOM_SEED} subset={SUBSET_SIZE} "
            f"cap={CAP} top_n={TOP_N_CLASSES}")
```

- [ ] **Step 3: Write `src/kidney_scrna/__init__.py`**

```python
"""Kidney scRNA-seq cell-type classifier package."""
__version__ = "1.0.0"
```

- [ ] **Step 4: Write `tests/test_config.py`**

```python
from kidney_scrna import config

def test_constants():
    assert config.RANDOM_SEED == 42
    assert config.TOP_N_CLASSES == 10
    assert config.TARGET_COL == "Cell_Labels"
    assert len(config.METADATA_COLS) == 9

def test_describe():
    assert "kidney_scrna" in config.describe()
```

- [ ] **Step 5: Install editable + run tests**

Run: `pip install -e ".[dev]"` then `pytest tests/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/kidney_scrna/__init__.py src/kidney_scrna/config.py tests/test_config.py
git commit -m "Add kidney_scrna package scaffold and config"
```

---

### Task 2: data.py (load / clean / subsample / split)

**Files:**
- Create: `src/kidney_scrna/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `load_raw() -> pd.DataFrame`
  - `basic_clean(df) -> pd.DataFrame` (harmonise labels, drop classes < MIN_CLASS_SIZE)
  - `select_top_classes(df, n=TOP_N_CLASSES) -> pd.DataFrame`
  - `split_features_target(df) -> (X, y)` (X = gene cols, y = Cell_Labels)
  - `subsample(X, y, size=SUBSET_SIZE, seed=RANDOM_SEED) -> (X_sub, y_sub)` (stratified)
  - `make_split(X, y) -> (X_train, X_test, y_train, y_test)` (stratified, 80/20)

- [ ] **Step 1: Write `tests/test_data.py`** (synthetic frame; no big CSV needed)

```python
import numpy as np, pandas as pd
from kidney_scrna import data, config

def _fake_df(n=400):
    rng = np.random.RandomState(0)
    genes = {f"G{i}": rng.randint(0, 5, n) for i in range(6)}
    meta = {c: 0 for c in config.METADATA_COLS if c != config.TARGET_COL}
    labels = (["PT"] * 200 + ["Proximal Tubule"] * 60 + ["T"] * 60 +
              ["Myeloid"] * 50 + ["Rare"] * 30)  # Rare has 30 < 100
    df = pd.DataFrame({**meta, **genes})
    df[config.TARGET_COL] = labels[:n]
    return df

def test_basic_clean_harmonises_and_drops_rare():
    out = data.basic_clean(_fake_df())
    assert "PT" not in out[config.TARGET_COL].unique()          # harmonised
    assert "Proximal Tubule" in out[config.TARGET_COL].unique()
    assert "Rare" not in out[config.TARGET_COL].unique()        # dropped (<100)

def test_split_features_target_excludes_metadata():
    X, y = data.split_features_target(data.basic_clean(_fake_df()))
    assert not any(c in X.columns for c in config.METADATA_COLS)
    assert y.name == config.TARGET_COL

def test_make_split_is_stratified_and_8020():
    df = data.basic_clean(_fake_df())
    X, y = data.split_features_target(df)
    Xtr, Xte, ytr, yte = data.make_split(X, y)
    assert abs(len(Xtr) / (len(Xtr) + len(Xte)) - 0.8) < 0.02
    assert set(ytr.unique()) == set(yte.unique())
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_data.py -v` → FAIL (module missing).

- [ ] **Step 3: Write `src/kidney_scrna/data.py`**

```python
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
```

- [ ] **Step 4: Run tests** — `pytest tests/test_data.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/kidney_scrna/data.py tests/test_data.py
git commit -m "Add data loading, cleaning, and splitting module"
```

---

### Task 3: features.py (leakage-free feature reduction)

**Files:**
- Create: `src/kidney_scrna/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces:
  - `drop_zero_variance(X_train, X_test) -> (X_train, X_test, removed)`
  - `drop_high_null(X_train, X_test, thr=NULL_THRESHOLD) -> (X_train, X_test, removed)`
  - `scale(X_train, X_test) -> (X_train_s, X_test_s, scaler)`
  - `variance_filter(X_train_s, X_test_s, thr=VT_THRESHOLD) -> (X_train_v, X_test_v, kept_cols)`
  - `rank_genes(X_train_v, y_train, seed) -> order (np.ndarray of column positions, best first)`
  - `sweep_k(X_train_v, y_train, order, seed) -> (feature_counts, f1_scores, best_k)`
  - `select(X_train_v, X_test_v, order, k) -> (X_train_final, X_test_final, selected_genes)`

- [ ] **Step 1: Write `tests/test_features.py`**

```python
import numpy as np, pandas as pd
from kidney_scrna import features

def _xy(n=300, p=40):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(n, p), columns=[f"G{i}" for i in range(p)])
    X["ZERO"] = 0.0                       # zero-variance column
    y = pd.Series(rng.randint(0, 4, n))
    return X, y

def test_drop_zero_variance_removes_constant_col():
    X, y = _xy()
    Xtr, Xte, removed = features.drop_zero_variance(X, X)
    assert "ZERO" in removed and "ZERO" not in Xtr.columns

def test_rank_and_select_returns_k_genes():
    X, y = _xy()
    Xtr, Xte, _ = features.drop_zero_variance(X, X.copy())
    Xtr, Xte, _ = features.drop_high_null(Xtr, Xte)
    Xtr, Xte, scaler = features.scale(Xtr, Xte)
    Xtr, Xte, kept = features.variance_filter(Xtr, Xte)
    order = features.rank_genes(Xtr, y, seed=42)
    Xf_tr, Xf_te, genes = features.select(Xtr, Xte, order, k=10)
    assert Xf_tr.shape[1] == 10 and len(genes) == 10
```

- [ ] **Step 2: Run test to verify it fails.**

- [ ] **Step 3: Write `src/kidney_scrna/features.py`** (ports the validated notebook-2 logic)

```python
"""Leakage-free feature reduction: fit on train, apply to test."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from . import config


def drop_zero_variance(X_train, X_test):
    removed = X_train.var()[lambda s: s == 0].index.tolist()
    return X_train.drop(columns=removed), X_test.drop(columns=removed), removed


def drop_high_null(X_train, X_test, thr=config.NULL_THRESHOLD):
    removed = X_train.isnull().mean()[lambda s: s > thr].index.tolist()
    return X_train.drop(columns=removed), X_test.drop(columns=removed), removed


def scale(X_train, X_test):
    cols = X_train.columns.tolist()
    sc = StandardScaler()
    Xtr = pd.DataFrame(sc.fit_transform(X_train), columns=cols, index=X_train.index)
    Xte = pd.DataFrame(sc.transform(X_test), columns=cols, index=X_test.index)
    return Xtr, Xte, sc


def variance_filter(X_train, X_test, thr=config.VT_THRESHOLD):
    vt = VarianceThreshold(thr).fit(X_train)
    kept = X_train.columns[vt.get_support()].tolist()
    return X_train[kept], X_test[kept], kept


def rank_genes(X_train, y_train, seed=config.RANDOM_SEED):
    y_enc = LabelEncoder().fit_transform(y_train)
    ranker = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=1500, tol=1e-3, random_state=seed)
    if len(X_train) > 4000:
        X_rank, _, y_rank, _ = train_test_split(
            X_train, y_enc, train_size=4000, random_state=seed, stratify=y_enc)
    else:
        X_rank, y_rank = X_train, y_enc
    rfe = RFE(ranker, n_features_to_select=1, step=0.3).fit(X_rank, y_rank)
    return np.argsort(rfe.ranking_)


def sweep_k(X_train, y_train, order, seed=config.RANDOM_SEED):
    y_enc = LabelEncoder().fit_transform(y_train)
    n = X_train.shape[1]
    counts, k = [], n // 4
    while k >= 1:
        counts.append(k); k //= 2
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    ev = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=1500, tol=1e-3, random_state=seed)
    scores = []
    for k in counts:
        cols = X_train.columns[order[:k]]
        scores.append(cross_val_score(ev, X_train[cols], y_enc, cv=cv,
                                      scoring="f1_weighted", n_jobs=-1).mean())
    best_k = counts[int(np.argmax(scores))]
    return counts, scores, best_k


def select(X_train, X_test, order, k):
    genes = X_train.columns[order[:k]].tolist()
    return X_train[genes], X_test[genes], genes
```

- [ ] **Step 4: Run tests** → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/kidney_scrna/features.py tests/test_features.py
git commit -m "Add leakage-free feature reduction module"
```

---

### Task 4: models.py (resamplers, searches, deployable model)

**Files:**
- Create: `src/kidney_scrna/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `make_resamplers(cap=CAP, seed) -> list[(name, sampler)]`
  - `knn_search(X_train, y_train, seed) -> fitted RandomizedSearchCV`
  - `svm_search(X_train, y_train, seed, n_iter=15) -> fitted BayesSearchCV`
  - `build_deployable_svm(X_train_raw, y_train, C, gamma, cap, seed) -> sklearn Pipeline(StandardScaler->SVC)` fitted on SMOTE+undersampled scaled data; no sampler in the returned pipeline.

- [ ] **Step 1: Write `tests/test_models.py`**

```python
import numpy as np, pandas as pd
from kidney_scrna import models

def _xy(n=600, p=8):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(n, p), columns=[f"G{i}" for i in range(p)])
    y = pd.Series((["A"] * 400 + ["B"] * 120 + ["C"] * 80)[:n])
    return X, y

def test_make_resamplers_two_steps():
    steps = models.make_resamplers(cap=100, seed=42)
    assert [s[0] for s in steps] == ["under", "over"]

def test_deployable_svm_predicts_and_has_no_sampler():
    X, y = _xy()
    pipe = models.build_deployable_svm(X, y, C=10, gamma=0.01, cap=100, seed=42)
    assert list(pipe.named_steps) == ["scaler", "svc"]        # no sampler at serve time
    proba = pipe.predict_proba(X.iloc[:3])
    assert proba.shape == (3, len(set(y)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails.**

- [ ] **Step 3: Write `src/kidney_scrna/models.py`**

```python
"""Model search pipelines (leakage-free) and the compact deployable model."""
from __future__ import annotations
from collections import Counter
from scipy.stats import randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from . import config


def make_resamplers(cap: int = config.CAP, seed: int = config.RANDOM_SEED):
    under = RandomUnderSampler(
        sampling_strategy=lambda y: {c: cap for c, v in Counter(y).items() if v > cap},
        random_state=seed)
    over = SMOTE(
        sampling_strategy=lambda y: {c: cap for c, v in Counter(y).items() if v < cap},
        random_state=seed, k_neighbors=5)
    return [("under", under), ("over", over)]


def knn_search(X_train, y_train, seed: int = config.RANDOM_SEED, n_iter: int = 20):
    pipe = ImbPipeline(make_resamplers(seed=seed) + [("knn", KNeighborsClassifier())])
    params = {"knn__n_neighbors": randint(1, 31),
              "knn__weights": ["uniform", "distance"],
              "knn__metric": ["euclidean", "manhattan", "chebyshev"]}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    s = RandomizedSearchCV(pipe, params, n_iter=n_iter, scoring="f1_weighted",
                           cv=cv, random_state=seed, n_jobs=-1)
    return s.fit(X_train, y_train)


def svm_search(X_train, y_train, seed: int = config.RANDOM_SEED, n_iter: int = 15):
    pipe = ImbPipeline(make_resamplers(seed=seed) +
                       [("svc", SVC(kernel="rbf", random_state=seed, cache_size=800))])
    space = {"svc__C": Real(1e-2, 1e2, prior="log-uniform"),
             "svc__gamma": Real(1e-4, 1e-1, prior="log-uniform")}
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    s = BayesSearchCV(pipe, space, n_iter=n_iter, scoring="f1_weighted",
                      cv=cv, random_state=seed, n_jobs=-1)
    return s.fit(X_train, y_train)


def build_deployable_svm(X_train_raw, y_train, C, gamma,
                         cap: int = config.CAP, seed: int = config.RANDOM_SEED) -> Pipeline:
    """Fit StandardScaler->SVC on raw selected-gene values. SMOTE+undersample are
    applied to the scaled training data before fitting the SVC, but the RETURNED
    pipeline contains only scaler+svc (samplers do nothing at predict time)."""
    scaler = StandardScaler().fit(X_train_raw)
    Xs = scaler.transform(X_train_raw)
    Xr, yr = Xs, y_train
    for _, sampler in make_resamplers(cap=cap, seed=seed):
        Xr, yr = sampler.fit_resample(Xr, yr)
    svc = SVC(kernel="rbf", C=C, gamma=gamma, probability=True,
              random_state=seed, cache_size=800).fit(Xr, yr)
    return Pipeline([("scaler", scaler), ("svc", svc)])
```

- [ ] **Step 4: Run tests** → 2 passed (uses tiny data; fast).

- [ ] **Step 5: Commit**

```bash
git add src/kidney_scrna/models.py tests/test_models.py
git commit -m "Add model search pipelines and deployable-model builder"
```

---

### Task 5: evaluate.py (metrics)

**Files:**
- Create: `src/kidney_scrna/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `metrics(y_true, y_pred, y_prob, classes) -> dict{weighted_f1, roc_auc, precision, recall}`
  - `per_class_f1(y_true, y_pred, classes) -> dict{class: {f1, support}}`
  - `winner_by_f1(m_knn, m_svm) -> "KNN" | "SVM"`

- [ ] **Step 1: Write `tests/test_evaluate.py`**

```python
import numpy as np
from kidney_scrna import evaluate

def test_metrics_perfect():
    y = np.array(["A", "B", "A", "B"]); classes = ["A", "B"]
    prob = np.array([[.9,.1],[.1,.9],[.8,.2],[.2,.8]])
    m = evaluate.metrics(y, y, prob, classes)
    assert m["weighted_f1"] == 1.0 and m["roc_auc"] == 1.0

def test_winner():
    assert evaluate.winner_by_f1({"weighted_f1": .6}, {"weighted_f1": .8}) == "SVM"
```

- [ ] **Step 2: Run test to verify it fails.**

- [ ] **Step 3: Write `src/kidney_scrna/evaluate.py`**

```python
"""Evaluation metrics for the multiclass classifier."""
from __future__ import annotations
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, classification_report)
from sklearn.preprocessing import label_binarize


def metrics(y_true, y_pred, y_prob, classes) -> dict:
    yb = label_binarize(y_true, classes=classes)
    return {
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "roc_auc": float(roc_auc_score(yb, y_prob, multi_class="ovr", average="weighted")),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_f1(y_true, y_pred, classes) -> dict:
    rep = classification_report(y_true, y_pred, labels=classes,
                                target_names=classes, output_dict=True, zero_division=0)
    return {c: {"f1": rep[c]["f1-score"], "support": int(rep[c]["support"])} for c in classes}


def winner_by_f1(m_knn: dict, m_svm: dict) -> str:
    return "SVM" if m_svm["weighted_f1"] >= m_knn["weighted_f1"] else "KNN"
```

- [ ] **Step 4: Run tests** → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/kidney_scrna/evaluate.py tests/test_evaluate.py
git commit -m "Add evaluation metrics module"
```

---

### Task 6: train.py (headless pipeline CLI)

**Files:**
- Create: `train.py`

**Interfaces:**
- Consumes: `data`, `features`, `models`, `evaluate`, `config`.
- Produces: `artifacts/model.joblib` (bundle: `{model, genes, model_type, classes}`),
  `artifacts/metrics.json`, `artifacts/examples.json`.

- [ ] **Step 1: Write `train.py`**

```python
#!/usr/bin/env python
"""Headless training CLI: raw CSV -> deployable model + metrics + demo samples."""
from __future__ import annotations
import argparse, json, time, warnings
import joblib
from kidney_scrna import config, data, features, models, evaluate

warnings.filterwarnings("ignore")


def _log(m): print(f"[train] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the kidney cell-type classifier.")
    ap.add_argument("--svm-iters", type=int, default=15)
    ap.add_argument("--knn-iters", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="tiny search for a smoke test")
    args = ap.parse_args()
    if args.quick:
        args.svm_iters, args.knn_iters = 6, 6

    t0 = time.time(); print(config.describe()); config.ensure_artifacts_dir()

    _log("loading + cleaning ...")
    df = data.select_top_classes(data.basic_clean(data.load_raw()))
    X, y = data.split_features_target(df)
    X, y = data.subsample(X, y)
    Xtr, Xte, ytr, yte = data.make_split(X, y)
    _log(f"split: {len(Xtr)} train / {len(Xte)} test")

    # leakage-free reduction (fit on train)
    Xtr, Xte, _ = features.drop_zero_variance(Xtr, Xte)
    Xtr, Xte, _ = features.drop_high_null(Xtr, Xte)
    Xtr_s, Xte_s, _ = features.scale(Xtr, Xte)
    Xtr_v, Xte_v, _ = features.variance_filter(Xtr_s, Xte_s)
    order = features.rank_genes(Xtr_v, ytr)
    _, _, best_k = features.sweep_k(Xtr_v, ytr, order)
    genes = features.select(Xtr_v, Xte_v, order, best_k)[2]
    _log(f"selected {len(genes)} genes")

    classes = sorted(ytr.unique())
    knn = models.knn_search(Xtr_v[genes], ytr, n_iter=args.knn_iters)
    svm = models.svm_search(Xtr_v[genes], ytr, n_iter=args.svm_iters)

    def _eval(search):
        yp = search.predict(Xte_v[genes]); pp = search.predict_proba(Xte_v[genes])
        return evaluate.metrics(yte, yp, pp, classes)
    m_knn, m_svm = _eval(knn), _eval(svm)
    winner = evaluate.winner_by_f1(m_knn, m_svm)
    _log(f"KNN F1={m_knn['weighted_f1']:.4f} | SVM F1={m_svm['weighted_f1']:.4f} | winner={winner}")

    # Deployable model: raw selected-gene values -> scaler -> SVC (the winner is SVM).
    bp = {k.replace("svc__", ""): float(v) for k, v in dict(svm.best_params_).items()}
    deployable = models.build_deployable_svm(Xtr[genes], ytr, C=bp["C"], gamma=bp["gamma"])

    joblib.dump({"model": deployable, "genes": list(genes), "model_type": "SVM (RBF)",
                 "classes": classes}, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(
        {"knn": {**m_knn, "cv_f1": float(knn.best_score_)},
         "svm": {**m_svm, "cv_f1": float(svm.best_score_), "best_params": bp},
         "winner": winner, "n_train": len(Xtr), "n_test": len(Xte),
         "n_genes": len(genes), "per_class_svm": evaluate.per_class_f1(
             yte, svm.predict(Xte_v[genes]), classes)}, indent=2))

    # Demo samples: real held-out cells, up to 2 per class, raw selected-gene values.
    demo = {"genes": list(genes), "classes": classes, "model_type": "SVM (RBF)",
            "metrics": {"weighted_f1": round(m_svm["weighted_f1"], 4),
                        "roc_auc": round(m_svm["roc_auc"], 4)}, "samples": []}
    yte_r = yte.reset_index(drop=True); Xte_r = Xte[genes].reset_index(drop=True)
    for cls in classes:
        for i in yte_r.index[yte_r == cls][:2]:
            demo["samples"].append({"label": cls,
                "features": {g: round(float(Xte_r.iloc[i][g]), 4) for g in genes}})
    config.EXAMPLES_PATH.write_text(json.dumps(demo, indent=1))
    _log(f"saved artifacts in {time.time()-t0:.0f}s -> {config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test** — `python train.py --quick`
Expected: prints split sizes, selected genes, a KNN/SVM F1 line and `winner=SVM`, and writes `artifacts/model.joblib`, `metrics.json`, `examples.json`. (Needs the 292 MB CSV present.)

- [ ] **Step 3: Verify artifacts + model size**

Run: `python -c "import joblib,os;b=joblib.load('artifacts/model.joblib');print(len(b['genes']),'genes',b['model_type']);print(round(os.path.getsize('artifacts/model.joblib')/1e6,1),'MB')"`
Decision: if model.joblib > ~25 MB, switch `build_deployable_svm` to a compact linear model (LogisticRegression(max_iter=2000) on the scaled selected genes) and re-run; otherwise keep RBF-SVM. Record the choice in the commit message.

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "Add headless training CLI that exports the deployable model"
```

---

### Task 7: serve.py (FastAPI app + interactive demo)

**Files:**
- Create: `src/kidney_scrna/serve.py`

**Interfaces:**
- Consumes: `config.MODEL_PATH`, `config.EXAMPLES_PATH`, the bundle keys `model/genes/model_type/classes`.
- Produces: FastAPI `app` with `GET /`, `GET /health`, `GET /model`, `POST /predict`.
  `/predict` request `{features: {gene: float}}`; response `{prediction, confidence, top3:[{label,prob}], model_type}`.

- [ ] **Step 1: Write `src/kidney_scrna/serve.py`**

```python
"""FastAPI serving layer for the kidney cell-type classifier."""
from __future__ import annotations
import json, logging, sys, time
from typing import Any
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from . import config

logger = logging.getLogger("kidney_scrna.serve")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout); h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h); logger.setLevel(logging.INFO)

def _log(event: str, **f: Any) -> None:
    logger.info(json.dumps({"event": event, **f}))

_BUNDLE = None
def load_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(f"No model at {config.MODEL_PATH}. Run `python train.py`.")
        _BUNDLE = joblib.load(config.MODEL_PATH)
        _log("model_loaded", model_type=_BUNDLE["model_type"], n_genes=len(_BUNDLE["genes"]))
    return _BUNDLE

def load_examples() -> dict:
    if config.EXAMPLES_PATH.exists():
        return json.loads(config.EXAMPLES_PATH.read_text())
    return {"samples": [], "genes": [], "classes": []}

class PredictRequest(BaseModel):
    features: dict[str, float] = Field(..., description="gene -> expression value; all genes from /model required")

class TopClass(BaseModel):
    label: str
    prob: float

class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    prediction: str
    confidence: float
    top3: list[TopClass]
    model_type: str

app = FastAPI(title="Kidney Cell-Type Classifier",
              description="Classifies human kidney cells into 10 types from scRNA-seq expression.",
              version="1.0.0")

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_available": config.MODEL_PATH.exists()}

@app.get("/model")
def model_info() -> dict:
    b = load_bundle()
    md = {}
    if config.METRICS_PATH.exists():
        md = json.loads(config.METRICS_PATH.read_text())
    return {"model_type": b["model_type"], "classes": b["classes"],
            "n_genes": len(b["genes"]), "genes": b["genes"], "metrics": md}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    b = load_bundle(); genes = b["genes"]; model = b["model"]
    missing = [g for g in genes if g not in req.features]
    if missing:
        raise HTTPException(422, f"Missing {len(missing)} gene(s), e.g. {missing[:5]}")
    row = [[float(req.features[g]) for g in genes]]
    t0 = time.time()
    proba = model.predict_proba(row)[0]
    order = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)
    classes = list(model.classes_)
    top3 = [TopClass(label=classes[i], prob=round(float(proba[i]), 4)) for i in order[:3]]
    pred = top3[0]
    _log("prediction", prediction=pred.label, confidence=pred.prob,
         latency_ms=round((time.time() - t0) * 1000, 2))
    return PredictResponse(prediction=pred.label, confidence=pred.prob,
                           top3=top3, model_type=b["model_type"])

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    try:
        b = load_bundle()
        model_line = f"{b['model_type']} on {len(b['genes'])} selected gene probes ({len(b['classes'])} cell types)"
    except Exception:
        model_line = "model not loaded - run `python train.py`"
    ex = load_examples()
    buttons = "\n".join(
        f"""<button class="sample" data-label="{s['label']}" data-features='{json.dumps(s['features'])}'>
              Kidney cell {i+1} <span class="truth">(actual: {s['label']})</span></button>"""
        for i, s in enumerate(ex.get("samples", [])))
    return _PAGE.replace("{{MODEL_LINE}}", model_line).replace(
        "{{BUTTONS}}", buttons or "<p><i>No demo samples. Run <code>python train.py</code>.</i></p>")
```

- [ ] **Step 2: Add the HTML template constant `_PAGE`** at the end of `serve.py` (self-contained, theme-aware, multiclass result with top-3 bars):

```python
_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kidney Cell-Type Classifier</title>
<style>
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  max-width: 720px; margin: 5vh auto; padding: 0 1.2rem; line-height: 1.6; }
h1 { margin-bottom:.2rem; } .sub { opacity:.75; margin-top:0; }
code { background: rgba(128,128,128,.18); padding:.1rem .35rem; border-radius:4px; }
a { color:#2f81f7; }
.card { border:1px solid rgba(128,128,128,.3); border-radius:10px; padding:1rem 1.2rem; margin:1rem 0; }
button.sample { display:block; width:100%; text-align:left; cursor:pointer; margin:.5rem 0;
  padding:.7rem 1rem; border-radius:8px; font-size:1rem; border:1px solid rgba(128,128,128,.4);
  background: rgba(128,128,128,.08); }
button.sample:hover { background: rgba(47,129,247,.15); }
.truth { opacity:.6; font-size:.85em; }
#result { margin-top:1rem; padding:1.1rem 1.2rem; border-radius:10px; display:none;
  border:1px solid rgba(128,128,128,.4); }
.row { display:flex; align-items:center; gap:.6rem; margin:.3rem 0; }
.row .lab { width:230px; font-size:.9rem; } .bar { flex:1; height:10px; border-radius:5px;
  background: rgba(128,128,128,.25); overflow:hidden; } .bar>div { height:100%; background:#4c9be8; }
.ok { color:#3fb950; } .no { color:#e85c5c; }
</style></head><body>
<h1>&#129516; Kidney Cell-Type Classifier</h1>
<p class="sub">Predicts which of 10 human kidney cell types a single cell is, from its scRNA-seq gene expression.</p>
<div class="card"><b>Model:</b> {{MODEL_LINE}} &nbsp;&middot;&nbsp; <b>Status:</b> healthy</div>
<h3>&#9654; Try it - classify a real cell</h3>
<p class="sub">These are actual held-out cells the model never trained on. Click one to send its gene-expression values to the live model.</p>
{{BUTTONS}}
<div id="result"></div>
<h3>API</h3>
<ul><li><a href="/docs">/docs</a> - interactive API docs</li>
<li><a href="/health">/health</a> &middot; <a href="/model">/model</a> &middot; <code>POST /predict</code></li></ul>
<p class="sub">Source: <a href="https://github.com/shiva-shivanibokka/Tisch-ML-Model">github.com/shiva-shivanibokka/Tisch-ML-Model</a></p>
<script>
document.querySelectorAll('button.sample').forEach(function(btn){
  btn.addEventListener('click', async function(){
    const box=document.getElementById('result'); box.style.display='block'; box.innerHTML='Predicting...';
    try{
      const features=JSON.parse(btn.getAttribute('data-features'));
      const truth=btn.getAttribute('data-label');
      const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({features:features})});
      const d=await r.json(); const correct=d.prediction===truth;
      let rows=d.top3.map(function(t){const pct=(t.prob*100).toFixed(1);
        return '<div class="row"><div class="lab">'+t.label+'</div><div class="bar"><div style="width:'+pct+'%"></div></div><div>'+pct+'%</div></div>';}).join('');
      box.innerHTML='<b>Prediction: '+d.prediction+'</b> <span class="'+(correct?'ok':'no')+'">'+
        (correct?'&#9989; matches actual':'&#10060; actual: '+truth)+'</span>'+rows;
    }catch(e){ box.textContent='Error: '+e; }
  });
});
</script></body></html>"""
```

- [ ] **Step 3: Manual smoke test** — `uvicorn kidney_scrna.serve:app --port 8000` then in another shell `curl localhost:8000/health` → `{"status":"ok",...}`; open `http://localhost:8000/` and click a sample → prediction renders. Stop the server.

- [ ] **Step 4: Commit**

```bash
git add src/kidney_scrna/serve.py
git commit -m "Add FastAPI serving layer with interactive multiclass demo"
```

---

### Task 8: API tests

**Files:**
- Create: `tests/test_api.py`

- [ ] **Step 1: Write `tests/test_api.py`** (uses the real artifacts produced by Task 6)

```python
import json, pytest
from fastapi.testclient import TestClient
from kidney_scrna import config, serve

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(),
                                reason="run `python train.py` first")
client = TestClient(serve.app)

def test_health():
    assert client.get("/health").json()["status"] == "ok"

def test_model_lists_genes_and_classes():
    d = client.get("/model").json()
    assert d["n_genes"] == len(d["genes"]) and len(d["classes"]) == 10

def test_predict_on_baked_sample():
    sample = json.loads(config.EXAMPLES_PATH.read_text())["samples"][0]
    r = client.post("/predict", json={"features": sample["features"]}).json()
    assert r["prediction"] in client.get("/model").json()["classes"]
    assert 0.0 <= r["confidence"] <= 1.0 and len(r["top3"]) == 3

def test_predict_missing_genes_422():
    assert client.post("/predict", json={"features": {"NOPE": 1.0}}).status_code == 422
```

- [ ] **Step 2: Run tests** — `pytest tests/test_api.py -v` → 4 passed (after Task 6 produced artifacts).

- [ ] **Step 3: Commit**

```bash
git add tests/test_api.py
git commit -m "Add serving API tests"
```

---

### Task 9: Docker + Fly config + requirements

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `fly.toml`
- Modify: `requirements.txt` (add serving deps)

- [ ] **Step 1: Append serving deps to `requirements.txt`**

```
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.6
joblib>=1.3
```

- [ ] **Step 2: Write `.dockerignore`**

```
Tisch24_MergedscRNA_80-85PctVAR.csv
*.ipynb
kidney_cells_*.csv
X_train.csv
X_test.csv
y_train.csv
y_test.csv
.git
.venv
__pycache__
.pytest_cache
.ruff_cache
docs
```

- [ ] **Step 3: Write `Dockerfile`** (slim, non-root, bakes artifacts only)

```dockerfile
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir gunicorn
COPY artifacts ./artifacts
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8080
CMD ["uvicorn", "kidney_scrna.serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 4: Write `fly.toml`** (app name confirmed at `fly launch`)

```toml
app = "tisch-kidney-classifier"
primary_region = "iad"

[build]

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[http_service.checks]]
  method = "get"
  path = "/health"
  interval = "15s"
  timeout = "2s"

[[vm]]
  memory = "512mb"
  cpu_kind = "shared"
  cpus = 1
```

- [ ] **Step 5: Build image locally to verify** — `docker build -t kidney-classifier .` (if Docker available); else defer to Fly remote build. Expected: build succeeds.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore fly.toml requirements.txt
git commit -m "Add Docker image, Fly.io config, and serving dependencies"
```

---

### Task 10: Commit artifacts + deploy live to Fly.io

**Files:**
- Modify: `.gitignore` (allow `artifacts/` to be committed so the image can bake it), commit `artifacts/`.

- [ ] **Step 1: Ensure artifacts are committable** — confirm `.gitignore` does not exclude `artifacts/`; if `model.joblib` size is acceptable (Task 6 decision), `git add artifacts/ && git commit -m "Add trained model artifacts for serving"`. (If too large for git, note it and use a Fly volume or build-time train instead — decided against measured size.)

- [ ] **Step 2: User authenticates** — ask the user to run, via the `!` prefix in their session: `! fly auth login` (opens browser). Wait for confirmation.

- [ ] **Step 3: Launch app (no deploy yet)** — `fly launch --no-deploy --copy-config --name tisch-kidney-classifier --region iad` (accept existing `fly.toml`). Expected: app created.

- [ ] **Step 4: Deploy** — `fly deploy` → wait for healthy. Expected: build + release succeed, `/health` check passes.

- [ ] **Step 5: Verify live** — `fly status` and `curl https://tisch-kidney-classifier.fly.dev/health` → `{"status":"ok",...}`. Record the URL.

- [ ] **Step 6: Commit any config tweaks** from launch.

---

### Task 11: Update README via readme-writer skill

- [ ] **Step 1:** Invoke the `readme-writer-skill` to update `README.md` in place, adding: the serving layer (train.py, API endpoints, demo), the live URL, the `src/kidney_scrna` package to Project Structure, a "Serving & Deployment" section, and the new Tech Stack entries (FastAPI, Docker, Fly.io). Keep all existing accurate content.

- [ ] **Step 2: Commit** the README.

- [ ] **Step 3: Push** — `git push origin main`.

---

## Self-Review

- **Spec coverage:** package (T1-5), train.py + artifacts (T6), serve.py + demo (T7), tests (T8), Docker/Fly (T9), deploy live (T10), README (T11). Deployable-model detail (scaler->SVC, no sampler at serve) covered in T4/T6. Model-size fallback covered in T6 step 3 and T10 step 1. All spec sections mapped.
- **Placeholder scan:** none — every code step has complete code.
- **Type consistency:** bundle keys `{model, genes, model_type, classes}` written in T6, read in T7/T8; `make_resamplers`/`build_deployable_svm`/search signatures consistent between T4 and T6; metrics dict keys (`weighted_f1`, `roc_auc`, ...) consistent T5→T6→T7.
