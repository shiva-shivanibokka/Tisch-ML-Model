import numpy as np
from kidney_scrna import evaluate


def test_metrics_perfect():
    # multiclass (3 classes) mirrors the real 10-class app path
    y = np.array(["A", "B", "C", "A"]); classes = ["A", "B", "C"]
    prob = np.array([[.9, .05, .05], [.05, .9, .05], [.05, .05, .9], [.8, .1, .1]])
    m = evaluate.metrics(y, y, prob, classes)
    assert m["weighted_f1"] == 1.0 and m["roc_auc"] == 1.0


def test_winner():
    assert evaluate.winner_by_f1({"weighted_f1": .6}, {"weighted_f1": .8}) == "SVM"
