"""Leakage-free feature reduction: fit on train, apply to test."""
from __future__ import annotations
import numpy as np
import pandas as pd
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
