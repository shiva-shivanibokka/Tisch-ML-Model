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
