import numpy as np
import pandas as pd
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
