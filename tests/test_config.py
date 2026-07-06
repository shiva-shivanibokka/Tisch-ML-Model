from kidney_scrna import config


def test_constants():
    assert config.RANDOM_SEED == 42
    assert config.TOP_N_CLASSES == 10
    assert config.TARGET_COL == "Cell_Labels"
    assert len(config.METADATA_COLS) == 9


def test_describe():
    assert "kidney_scrna" in config.describe()
