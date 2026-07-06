import json

import pytest
from fastapi.testclient import TestClient

from kidney_scrna import config, serve

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(), reason="run `python train.py` first")
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


def test_predict_rejects_nan():
    # NaN isn't valid standard JSON, so send a raw body with a literal NaN token
    # (which Python's json parser accepts) to reach the finite-value check.
    sample = json.loads(config.EXAMPLES_PATH.read_text())["samples"][0]
    feats = dict(sample["features"])
    feats[next(iter(feats))] = -987654.321          # unique sentinel
    body = json.dumps({"features": feats}).replace("-987654.321", "NaN")
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422
