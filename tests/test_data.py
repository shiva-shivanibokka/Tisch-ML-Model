import numpy as np
import pandas as pd
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
