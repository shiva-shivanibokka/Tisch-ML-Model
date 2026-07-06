#!/usr/bin/env python
"""Headless training CLI: raw CSV -> deployable model + metrics + demo samples.

Runs the same pipeline as the notebooks, but reproducible and headless:

    raw CSV -> clean/harmonise/drop-rare -> top-10 -> stratified subsample
            -> train/test split -> leakage-free feature reduction (~293 genes)
            -> tune KNN (RandomizedSearchCV) and SVM (BayesSearchCV), SMOTE in folds
            -> evaluate on the held-out test set -> pick the winner
            -> fit a compact StandardScaler->SVC on the selected raw genes
            -> save artifacts/{model.joblib, metrics.json, examples.json}

Usage:
    python train.py            # full run (~20-25 min on CPU)
    python train.py --quick    # tiny search for a fast smoke test
"""
from __future__ import annotations
import argparse
import json
import time
import warnings

import joblib

from kidney_scrna import config, data, features, models, evaluate

warnings.filterwarnings("ignore")


def _log(m: str) -> None:
    print(f"[train] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the kidney cell-type classifier.")
    ap.add_argument("--svm-iters", type=int, default=15)
    ap.add_argument("--knn-iters", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="tiny search for a smoke test")
    args = ap.parse_args()
    if args.quick:
        args.svm_iters, args.knn_iters = 6, 6

    t0 = time.time()
    print(config.describe())
    config.ensure_artifacts_dir()

    _log("loading + cleaning ...")
    df = data.select_top_classes(data.basic_clean(data.load_raw()))
    X, y = data.split_features_target(df)
    X, y = data.subsample(X, y)
    Xtr, Xte, ytr, yte = data.make_split(X, y)
    _log(f"split: {len(Xtr)} train / {len(Xte)} test")

    # leakage-free reduction (fit on train). Xtr/Xte stay RAW (unscaled) here.
    Xtr, Xte, _ = features.drop_zero_variance(Xtr, Xte)
    Xtr, Xte, _ = features.drop_high_null(Xtr, Xte)
    Xtr_s, Xte_s, _ = features.scale(Xtr, Xte)
    Xtr_v, Xte_v, _ = features.variance_filter(Xtr_s, Xte_s)
    _log("ranking genes (slow step) ...")
    order = features.rank_genes(Xtr_v, ytr)
    _, _, best_k = features.sweep_k(Xtr_v, ytr, order)
    genes = features.select(Xtr_v, Xte_v, order, best_k)[2]
    _log(f"selected {len(genes)} genes (best_k={best_k})")

    classes = sorted(ytr.unique())

    _log(f"tuning KNN (RandomizedSearchCV, {args.knn_iters} iters) ...")
    knn = models.knn_search(Xtr_v[genes], ytr, n_iter=args.knn_iters)
    knn_pred = knn.predict(Xte_v[genes])
    knn_prob = knn.predict_proba(Xte_v[genes])
    m_knn = evaluate.metrics(yte, knn_pred, knn_prob, classes)
    _log(f"  KNN test F1={m_knn['weighted_f1']:.4f}  (CV {knn.best_score_:.4f})")

    _log(f"tuning SVM (BayesSearchCV, {args.svm_iters} iters) ...")
    svm = models.svm_search(Xtr_v[genes], ytr, n_iter=args.svm_iters)
    bp = {k.replace("svc__", ""): float(v) for k, v in dict(svm.best_params_).items()}

    # The deployable model IS what we evaluate + serve: raw selected genes ->
    # StandardScaler -> RBF-SVC(probability=True). (The search used probability=False
    # for speed, so we score the SVM through this probability-enabled model.)
    _log("fitting deployable SVM (scaler -> RBF-SVC) ...")
    deployable = models.build_deployable_svm(Xtr[genes], ytr, C=bp["C"], gamma=bp["gamma"])
    svm_pred = deployable.predict(Xte[genes])
    svm_prob = deployable.predict_proba(Xte[genes])
    m_svm = evaluate.metrics(yte, svm_pred, svm_prob, classes)
    _log(f"  SVM test F1={m_svm['weighted_f1']:.4f}  (CV {svm.best_score_:.4f})")

    winner = evaluate.winner_by_f1(m_knn, m_svm)
    _log(f"winner by weighted F1: {winner}")

    joblib.dump({"model": deployable, "genes": list(genes), "model_type": "SVM (RBF)",
                 "classes": classes}, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(
        {"knn": {**m_knn, "cv_f1": float(knn.best_score_),
                 "best_params": {k.replace("knn__", ""): (v.item() if hasattr(v, "item") else v)
                                 for k, v in knn.best_params_.items()}},
         "svm": {**m_svm, "cv_f1": float(svm.best_score_), "best_params": bp},
         "winner": winner, "n_train": len(Xtr), "n_test": len(Xte),
         "n_genes": len(genes),
         "per_class_svm": evaluate.per_class_f1(yte, svm_pred, classes)}, indent=2))

    # Demo samples: real held-out cells (up to 2 per class), raw selected-gene values.
    demo = {"genes": list(genes), "classes": classes, "model_type": "SVM (RBF)",
            "metrics": {"weighted_f1": round(m_svm["weighted_f1"], 4),
                        "roc_auc": round(m_svm["roc_auc"], 4)}, "samples": []}
    yte_r = yte.reset_index(drop=True)
    Xte_r = Xte[genes].reset_index(drop=True)
    for cls in classes:
        for i in yte_r.index[yte_r == cls][:2]:
            demo["samples"].append({"label": cls,
                "features": {g: round(float(Xte_r.iloc[i][g]), 4) for g in genes}})
    config.EXAMPLES_PATH.write_text(json.dumps(demo, indent=1))

    _log(f"saved artifacts in {time.time() - t0:.0f}s -> {config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
