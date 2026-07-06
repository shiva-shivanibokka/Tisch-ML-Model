import numpy as np
import pandas as pd
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
