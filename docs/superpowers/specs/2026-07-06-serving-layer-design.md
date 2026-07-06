# Design: Serving Layer + Interactive Demo for Tisch-ML-Model

**Date:** 2026-07-06
**Status:** Approved (brainstorming), pending implementation plan

## Goal

Add a production-style serving layer and interactive demo to the (currently
notebook-only) kidney scRNA-seq cell-type classifier, mirroring the architecture
already shipped in the sibling `Cumida-ML-Model` repo, adapted for this project's
**multiclass** problem (10 kidney cell types, 293 selected genes, RBF-SVM). Deploy
it live to Fly.io this session.

## Non-goals

- No change to the four teaching notebooks' results or methodology.
- No new modelling approach; the served model is the existing winning SVM.
- No auth, database, or multi-service infrastructure.

## Architecture

Extract the notebook pipeline into a small importable package so `train.py`, the
notebooks, and the API share one implementation.

```
src/kidney_scrna/
  __init__.py
  config.py     # paths, RANDOM_SEED=42, SUBSET_SIZE=20_000, CAP=1000, class labels, describe()
  data.py       # load_raw, harmonise_labels, drop_rare, select_top_classes, subsample, make_split
  features.py   # drop_zero_variance, drop_high_null, variance_filter, rfe_rank (SGD), select_k
  models.py     # make_resamplers, knn_search, svm_search, build_deployable_model
  evaluate.py   # metrics (weighted F1/AUC/precision/recall), per_class, winner_by_f1
  serve.py      # FastAPI app + interactive demo page
train.py        # headless CLI orchestrating the pipeline, saves artifacts/
tests/
  test_api.py   # /health, /model, /predict smoke tests using a baked sample
artifacts/      # model.joblib, metrics.json, examples.json  (baked into the image)
Dockerfile, fly.toml, pyproject.toml, .dockerignore
```

**Why a package:** the notebooks currently duplicate logic; a shared package removes
drift between "what the notebooks do" and "what is served," and makes the pipeline
testable and importable. This matches the Cumida repo's structure for portfolio
consistency.

## The deployable model (key multiclass detail)

The notebooks scale *all* ~2,350 genes, then select 293. To serve **raw** expression
values for just the 293 selected genes, `train.py` builds a compact, self-contained
sklearn `Pipeline`:

```
StandardScaler(fit on the 293 raw selected-gene columns of training data)
  -> SVC(kernel='rbf', C, gamma, probability=True)   # best params from the search
```

Class balancing (SMOTE + RandomUnderSampler to CAP=1000/class) is applied **only while
training** the SVC — it is *not* part of the served pipeline (samplers do nothing at
predict time). The served artifact therefore takes raw gene values and returns
calibrated class probabilities.

- **Served model:** the winner by weighted F1 (currently SVM). `/model` reports both
  models' scores for transparency.
- **Model-size risk:** an RBF-SVM bundle can be several MB. After training, verify the
  `model.joblib` size. If it is too large to comfortably commit/bake into the image,
  fall back to a compact linear model (LogisticRegression / calibrated LinearSVC on the
  293 genes) for serving and note the choice. Decision made against the measured size.

## Serving API (`serve.py`, FastAPI)

| Endpoint | Method | Contract |
|---|---|---|
| `/` | GET | HTML interactive demo page |
| `/health` | GET | `{status, model_available}` liveness probe |
| `/model` | GET | `{model_type, classes[10], n_genes, genes[293], metrics}` |
| `/predict` | POST | `{features: {gene: value, ...}}` -> `{prediction, confidence, top3:[{label,prob}], model_type}` |

- `/predict` validates that all 293 required genes are present (HTTP 422 with the
  missing ones otherwise), orders features exactly as the model expects, and returns
  the predicted cell type, its probability, and the top-3 classes (richer than a binary
  bar because there are 10 classes).
- Every prediction logs a structured JSON line (event, prediction, confidence,
  latency_ms) for observability.

## Demo UI

Self-contained inline HTML/CSS/JS (no external assets), theme-aware. Real held-out test
cells are baked into `examples.json` (1-2 per class ≈ 10-20 buttons), each carrying its
293 raw gene values and true label. Clicking a cell POSTs to `/predict` and renders the
predicted cell type, confidence, the top-3 bar, and ✅/❌ vs. the actual label. Includes
links to `/docs`, `/health`, `/model`, and the GitHub repo.

## `train.py` CLI

Runs the pipeline headless and reproducibly (`RANDOM_SEED=42`):

```
raw CSV -> clean/harmonise/drop-rare -> top-10 -> stratified subsample (20k)
        -> train/test split -> leakage-free feature reduction (293 genes)
        -> tune KNN (RandomizedSearchCV) and SVM (BayesSearchCV), SMOTE inside folds
        -> evaluate both on held-out test -> pick winner
        -> fit compact deployable model on selected genes
        -> save artifacts/{model.joblib, metrics.json, examples.json}
```

Flags: `--quick` (fewer search iterations for a fast smoke test). Prints a KNN-vs-SVM
summary table. Full run is ~15-25 min on CPU; `--quick` is a couple of minutes.

## Testing

`tests/test_api.py` (pytest + FastAPI `TestClient`): asserts `/health` is ok, `/model`
lists 293 genes and 10 classes, and `/predict` on a baked sample returns a valid class
label with probabilities that sum to ~1. Runnable via `pytest`.

## Deployment (Fly.io, this session)

- `Dockerfile`: slim Python base, non-root user, install package + deps, copy
  `artifacts/`, run `uvicorn kidney_scrna.serve:app`. `.dockerignore` excludes the
  292 MB dataset, notebooks' large outputs, and intermediate CSVs.
- `fly.toml`: single small machine, internal port 8080, `/health` check.
- Live deploy: user runs `fly auth login` interactively (via the `!` prefix); the
  assistant drives `fly launch --no-deploy` (or reuse config) and `fly deploy`, then
  reports the public URL. Model + examples are baked into the image, so no data or
  training happens at deploy time.

## Dependencies added

`fastapi`, `uvicorn[standard]`, `pydantic`, `joblib` (added to `requirements.txt` and a
`pyproject.toml` for the installable package). Existing: scikit-learn, imbalanced-learn,
scikit-optimize, pandas, numpy, scipy, matplotlib, seaborn.

## README

After the serving layer is built, tested, and deployed, re-run the readme-writer skill
so the README documents `train.py`, the API contract, the demo, the live URL, and the
updated project structure — in a single pass (the README is otherwise already current).

## Risks / open items

- **Model size** (see above) — measured decision after first `train.py` run.
- **Training time** — full `train.py` re-runs the SVM search (~15-25 min). Acceptable
  for a one-off artifact build; `--quick` exists for smoke tests.
- **Fly.io image size** — image bakes `model.joblib` + `examples.json` only (small); the
  292 MB dataset is never copied in.
