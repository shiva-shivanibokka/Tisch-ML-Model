"""End-to-end wiring test on synthetic data (no real dataset needed, so it runs in CI):
feature reduction -> gene selection -> deployable model -> prediction."""
import numpy as np
import pandas as pd

from kidney_scrna import data, features, models


def _synthetic(sizes=(40, 40, 20), p=30, seed=0):
    rng = np.random.RandomState(seed)
    blocks, labels = [], []
    for k, n in enumerate(sizes):
        X = rng.rand(n, p)
        X[:, k * 3:(k + 1) * 3] += 2.0        # give each class real signal on a few genes
        blocks.append(X)
        labels += [f"C{k}"] * n
    Xdf = pd.DataFrame(np.vstack(blocks), columns=[f"G{i}" for i in range(p)])
    return Xdf, pd.Series(labels, name="y")


def test_end_to_end_reduce_select_serve():
    X, y = _synthetic()
    Xtr, Xte, ytr, yte = data.make_split(X, y)

    # leakage-free reduction (fit on train)
    Xtr, Xte, _ = features.drop_zero_variance(Xtr, Xte)
    Xtr, Xte, _ = features.drop_high_null(Xtr, Xte)
    Xtr_s, Xte_s, _ = features.scale(Xtr, Xte)
    Xtr_v, Xte_v, _ = features.variance_filter(Xtr_s, Xte_s)
    order = features.rank_genes(Xtr_v, ytr, seed=42)
    _, _, best_k = features.sweep_k(Xtr_v, ytr, order, seed=42)
    genes = features.select(Xtr_v, Xte_v, order, best_k)[2]
    assert 1 <= len(genes) <= Xtr_v.shape[1]

    # deployable model on raw selected genes (SMOTE-up + undersample both exercised)
    model = models.build_deployable_svm(Xtr[genes], ytr, C=10, gamma=0.1, cap=25, seed=42)
    proba = model.predict_proba(Xte[genes].to_numpy())
    assert proba.shape == (len(Xte), y.nunique())
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(model.predict(Xte[genes].to_numpy())).issubset(set(y))
