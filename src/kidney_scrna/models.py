"""Model search pipelines (leakage-free) and the compact deployable model."""
from __future__ import annotations
from collections import Counter
import numpy as np
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
    pipeline contains only scaler+svc (samplers do nothing at predict time).

    The scaler is fit on a nameless array so the served model expects positional
    input (as the API sends) without emitting sklearn feature-name warnings."""
    X = np.asarray(X_train_raw, dtype=float)
    scaler = StandardScaler().fit(X)
    Xr, yr = scaler.transform(X), y_train
    for _, sampler in make_resamplers(cap=cap, seed=seed):
        Xr, yr = sampler.fit_resample(Xr, yr)
    svc = SVC(kernel="rbf", C=C, gamma=gamma, probability=True,
              random_state=seed, cache_size=800).fit(Xr, yr)
    return Pipeline([("scaler", scaler), ("svc", svc)])
