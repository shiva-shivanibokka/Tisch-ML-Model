# Kidney Cell Type Classification — scRNA-seq Machine Learning Project

This project builds machine learning models to classify human kidney cell types using single-cell RNA sequencing (scRNA-seq) data. The data comes from five published research studies, merged into a single dataset of over 60,000 individual kidney cells. Each cell is described by the expression levels of 2,358 genes, and the goal is to predict which type of kidney cell each one is.

The project is structured as four sequential Jupyter notebooks, each handling one stage of the pipeline. The notebooks are designed for students learning applied machine learning, and every step is explained in plain language alongside the code.

---

## Table of Contents

- [Background](#background)
- [The Dataset](#the-dataset)
- [The Five Studies](#the-five-studies)
- [Label Harmonisation](#label-harmonisation)
- [Cell Types in This Dataset](#cell-types-in-this-dataset)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Notebook 1 — EDA and Loading](#notebook-1--eda-and-loading)
- [Notebook 2 — Preprocessing and Feature Selection](#notebook-2--preprocessing-and-feature-selection)
- [Notebook 3 — K-Nearest Neighbours](#notebook-3--k-nearest-neighbours)
- [Notebook 4 — Support Vector Machine and Model Comparison](#notebook-4--support-vector-machine-and-model-comparison)
- [Key Design Decisions](#key-design-decisions)
- [Libraries Used](#libraries-used)
- [Results Summary](#results-summary)

---

## Background

### What is scRNA-seq?

Single-cell RNA sequencing (scRNA-seq) is a laboratory technique that allows scientists to measure gene expression in individual cells. Every cell in your body contains the same DNA, but different cell types activate different genes. A kidney cell and a blood cell have the same genetic code, but they express different genes — and it is that pattern of gene expression that makes them different.

scRNA-seq works by isolating individual cells, capturing the RNA molecules inside each one (RNA is the messenger that carries instructions from DNA to make proteins), and sequencing them to get a count of how many copies of each gene's RNA were present. The result is a table where each row is a single cell and each column is a gene, and the values are the RNA count for that gene in that cell.

### Why machine learning?

Traditionally, scientists would label cell types by hand by looking at known marker genes (genes that are known to be highly expressed in one particular cell type). This is accurate but slow and requires expert knowledge. With datasets containing tens of thousands of cells from multiple studies, machine learning offers a way to automate this process — learning the gene expression patterns that distinguish each cell type and applying those patterns to classify new cells automatically.

### Why kidney cells?

The kidney is made up of many specialised cell types, each performing a different function — filtering blood, reabsorbing nutrients, secreting waste, regulating blood pressure, and more. Understanding which cells are affected in kidney diseases like diabetic nephropathy or chronic kidney disease requires knowing exactly what type each cell is. This dataset merges cells from five independent human kidney studies, covering both healthy and diseased tissue.

---

## The Dataset

**File:** `Tisch24_MergedscRNA_80-85PctVAR.csv`  
**Size:** 292 MB  
**Rows:** 60,725 (one row per cell)  
**Columns:** 2,367 total — 9 metadata columns + 2,358 gene expression columns

The filename refers to the TISCH2 database (Tumor Immune Single-cell Hub, 2024 edition), and the "80–85% VAR" part means the dataset was filtered to retain only the genes that explain 80–85% of the total variance across all cells. This pre-filtering was done to keep the file size manageable while preserving the most biologically informative genes.

### Metadata Columns

| Column | Type | Description |
|---|---|---|
| `Cell_ID` | Text | Unique identifier for each cell (e.g. `NK1_GGGAACGCGCCA`) |
| `nCount_RNA` | Integer | Total number of RNA molecules detected in the cell |
| `nFeature_RNA` | Integer | Number of distinct genes detected in the cell |
| `StudyOrigin_Author` | Text | Which research study the cell came from |
| `percent.mt` | Decimal | Percentage of RNA from mitochondrial genes (a quality control metric) |
| `Sex` | Text | Patient sex (Male / Female / missing) |
| `Sampling_Location` | Text | Where in the kidney the tissue was sampled from |
| `Age` | Text | Patient age range (e.g. "60 – 69") |
| `Cell_Labels` | Text | **The target variable — the cell type label** |

### Gene Expression Columns

The remaining 2,358 columns are named after human genes (e.g. `ABCG2`, `ABO`, `AANAT`). Each value is an integer count representing how many RNA molecules from that gene were detected in that particular cell. The vast majority of these values are zero — see [Sparsity](#sparsity) below.

### Sparsity

One of the most important characteristics of scRNA-seq data is its **sparsity**. In any given cell, roughly 96–97% of gene columns are zero. This is not a data quality problem — it is a biological reality. In a single cell at any given moment, most genes are simply silent. Only a small subset of genes are actively producing RNA. On average, each cell in this dataset has fewer than 100 genes with non-zero expression out of 2,358 total gene columns.

This sparsity has direct consequences for machine learning. Many genes will have near-zero variance across all cells (because they are almost always zero), making them uninformative for classification. The Variance Threshold step in Notebook 2 removes these genes efficiently.

---

## The Five Studies

The dataset merges cells from five independent human kidney studies:

| Study | Cells | Notes |
|---|---|---|
| Menon | 22,264 | Largest contributor — healthy donor kidneys |
| Liao | 14,880 | Includes immune cell populations |
| Lake | 13,255 | Broad cell type coverage across cortex and medulla |
| Young | 6,067 | Includes rare immune cell types |
| Wu | 4,259 | Diabetic kidney disease study; used abbreviated cell type labels |

Each study was conducted independently, using different laboratory protocols, different sequencing depths, and different cell type naming conventions. This creates two key challenges:

1. **Batch effects** — systematic differences between studies caused by the lab conditions rather than biology. Cells from the same type but different studies may look slightly different in the data.
2. **Label inconsistency** — the Wu study used abbreviated cell type names while the others used full descriptive names for the same cell types.

### About the Wu Study

Wu et al. profiled kidney cells from patients with **diabetic kidney disease** alongside healthy donor kidneys. Diabetic kidney disease is one of the leading causes of kidney failure worldwide, and understanding which cell types are most affected — and how their gene expression changes — is critical for developing treatments. Their study used short abbreviations in the cell type annotations (e.g. `PT` for proximal tubule cells), which was the convention in their original publication.

---

## Label Harmonisation

Because the Wu study used different names for the same cell types, the labels must be standardised before training. Without this step, a machine learning model would treat `PT` (Wu's label for proximal tubule cells) and `Proximal Tubule` (everyone else's label) as two completely different classes — which would be biologically incorrect and would corrupt the training process.

The following six mappings are applied in Notebook 1:

| Wu Label | Standardised Label |
|---|---|
| `PT` | `Proximal Tubule` |
| `DT` | `Distal Convoluted Tubule` |
| `LH` | `Loop of Henle and Parietal Epithelium` |
| `PC` | `Collecting Duct Principal` |
| `IC` | `Collecting Duct Intercalated` |
| `P` | `Glomerular Epithelium and Podocytes` |

After harmonisation, the number of unique cell type labels is reduced from **31 to 25**.

---

## Cell Types in This Dataset

After harmonisation and removal of rare classes (see below), the dataset contains **22 kidney cell types**. These span the major functional regions of the kidney:

**Tubular cells** (handle filtration and reabsorption along the nephron):
- Proximal Tubule — the most abundant cell type in the dataset (~37% of all cells); responsible for reabsorbing the majority of filtered nutrients
- Distal Convoluted Tubule
- Ascending Thin Limb
- Descending Thin Limb
- Thick Ascending Limb
- Loop of Henle and Parietal Epithelium
- Connecting Tubule

**Collecting duct cells** (responsible for final urine concentration):
- Collecting Duct Principal
- Collecting Duct Intercalated

**Vascular cells** (form the blood vessels running through the kidney):
- Endothelium
- Glomerular Endothelium
- Ascending Vasa Recta
- Descending Vasa Recta

**Glomerular cells** (involved in blood filtration at the start of the nephron):
- Glomerular Epithelium and Podocytes
- Mesangium and Vascular Smooth Muscle and Pericytes

**Immune cells** (resident immune population):
- T cells
- Myeloid cells
- Natural Killer cells
- Natural Killer and T cells
- B cells

**Stromal cells:**
- Fibroblasts

**Other:**
- Urothelium (lining the urinary tract)

### Rare Classes Removed

Three cell types were removed from the dataset before training because they had fewer than 100 cells each. Machine learning models cannot reliably learn patterns from such a small number of examples, and these classes would also cause problems during stratified train/test splitting.

| Removed Class | Cell Count |
|---|---|
| Plasmacytoid | 19 |
| Mast | 22 |
| Neutrophil | 83 |

After removal, the final working dataset contains **22 classes**.

### Class Imbalance

The dataset is significantly imbalanced. Proximal Tubule cells make up approximately 37% of all cells, while the smallest retained classes have only a few hundred cells. This imbalance is handled differently in each model notebook:

- **KNN (Notebook 3):** Uses `weights='distance'`, which gives closer neighbours more influence and partially compensates for the majority class outnumbering minority class cells among a cell's nearest neighbours.
- **SVM (Notebook 4):** Uses `class_weight='balanced'`, which automatically assigns higher penalty to misclassifying cells from smaller classes.

---

## Project Structure

```
Tisch-ML-Model/
│
├── Tisch24_MergedscRNA_80-85PctVAR.csv   ← raw dataset (292 MB)
│
├── 01_eda_loading.ipynb                  ← data loading, cleaning, EDA
├── 02_preprocessing.ipynb                ← feature selection, train/test split
├── 03_knn.ipynb                          ← K-Nearest Neighbours classifier
└── 04_svm.ipynb                          ← Support Vector Machine + comparison
```

The notebooks must be run **in order**. Each notebook saves its outputs to Google Drive, and the next notebook reads them as inputs.

```
01_eda_loading.ipynb
    └── saves: kidney_cells_clean.csv
                kidney_cells_top_classes.csv

02_preprocessing.ipynb
    └── reads: kidney_cells_clean.csv  (or  kidney_cells_top_classes.csv)
    └── saves: X_train.csv
                X_test.csv
                y_train.csv
                y_test.csv

03_knn.ipynb
    └── reads: X_train.csv, X_test.csv, y_train.csv, y_test.csv

04_svm.ipynb
    └── reads: X_train.csv, X_test.csv, y_train.csv, y_test.csv
```

---

## How to Run

### Requirements

- A Google account with Google Drive
- Google Colab (free, runs in your browser — no local installation needed)
- The dataset file `Tisch24_MergedscRNA_80-85PctVAR.csv` uploaded to a folder on your Google Drive

### Setup Steps

1. Create a folder on your Google Drive, for example: `My Drive/kidney_scrna_data/`
2. Upload `Tisch24_MergedscRNA_80-85PctVAR.csv` into that folder
3. Open each notebook in Google Colab (File → Open notebook → Google Drive, or upload from GitHub)
4. In **each notebook**, find the configuration cell at the top and update `data_dir` to match your folder path:

```python
data_dir = Path('/content/drive/MyDrive/kidney_scrna_data')
```

5. Run the notebooks in order: `01` → `02` → `03` → `04`

### Choosing Between Full Dataset and Top Classes

In **Notebook 2**, there is a toggle at the top of the configuration cell:

```python
# Options:
#   'kidney_cells_clean.csv'       — all 22 classes
#   'kidney_cells_top_classes.csv' — top 10 prominent classes only
INPUT_FILE = 'kidney_cells_clean.csv'
```

- Use `kidney_cells_clean.csv` to classify all 22 cell types
- Use `kidney_cells_top_classes.csv` to focus only on the 10 most well-represented cell types — this gives higher overall accuracy and trains faster, and is a good starting point if you want to understand the models before tackling the full 22-class problem

Both output files are created by Notebook 1.

---

## Notebook 1 — EDA and Loading

**File:** `01_eda_loading.ipynb`  
**Input:** `Tisch24_MergedscRNA_80-85PctVAR.csv`  
**Outputs:** `kidney_cells_clean.csv`, `kidney_cells_top_classes.csv`

This notebook handles all data loading, cleaning, and exploratory analysis. No machine learning happens here — the goal is to understand what is in the data before doing anything with it.

### What it does, step by step

**Step 1 — Load the raw dataset**  
Reads the 292 MB CSV file. Prints the number of rows (cells), columns, and gene columns.

**Step 2 — Inspect the data**  
Prints the data type of each metadata column, reports which columns have missing values and what percentage is missing, and shows how many unique values are in each column. Then prints every raw cell type label with its cell count and percentage of the total.

**Step 3 — Label harmonisation**  
Applies the 6 Wu label mappings described above, reducing from 31 unique labels to 25.

**Step 4 — Remove rare cell types**  
Removes cells belonging to any class with fewer than 100 cells. Prints which classes were removed and the final cell and class counts.

**Step 5 — EDA plots**  
Produces the following visualisations:

| Plot | What it shows |
|---|---|
| Cell Type Distribution | Horizontal bar chart of all 22 cell types with cell counts and percentages |
| Cell Type Proportions | Pie chart showing the top 8 cell types and an "Other" slice |
| Missing Values | Bar chart of missing percentage per metadata column, with reference lines at 20% and 50% |
| Top 10 Cell Types by Count | Bar chart focusing on the 10 most abundant classes |
| Top 10 vs Remaining Classes | Pie chart comparing the prominent classes to the rest |
| Genes Detected per Cell | Histogram of `nFeature_RNA` with median and mean reference lines |
| Median Genes by Cell Type | Bar chart showing median gene detection count for each cell type |
| Gene Expression Sparsity per Cell | Histogram showing what fraction of each cell's gene columns are zero |
| Median Sparsity by Cell Type | Bar chart showing how sparse each cell type tends to be |

**Step 6 — Save outputs**  
Saves `kidney_cells_clean.csv` (all 22 classes) and `kidney_cells_top_classes.csv` (top 10 only).

---

## Notebook 2 — Preprocessing and Feature Selection

**File:** `02_preprocessing.ipynb`  
**Input:** `kidney_cells_clean.csv` or `kidney_cells_top_classes.csv`  
**Outputs:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

This notebook prepares the data for machine learning. It takes the cleaned dataset, reduces the 2,358 gene features down to a smaller informative subset, and splits the data into training and test sets.

### Data Leakage

A key principle throughout this notebook is **preventing data leakage**. Data leakage happens when information from the test set is used during preprocessing or training. For example, if you calculate the mean and standard deviation of the full dataset (including the test set) and use those values to scale the data, the model has indirectly "seen" the test set before evaluation. This produces unrealistically optimistic results.

To prevent this, every preprocessing step in this notebook is **fitted on the training set only** and then **applied** to both the training and test sets. The test set remains completely untouched until the final evaluation in Notebooks 3 and 4.

### What it does, step by step

**Step 1 — Separate features and target**  
Creates `X` (the 2,358 gene expression columns) and `y` (the `Cell_Labels` column). Metadata columns like age, sex, and study origin are excluded because cell type is determined by gene expression, not patient demographics.

**Step 2 — Stratified sample of 10,000 cells**  
The full ~60,000-cell dataset is too large to run Recursive Feature Elimination (RFE) on in a reasonable time. A sample of 10,000 cells is taken using **stratified sampling**, which preserves the class proportions from the full dataset. For example, if Proximal Tubule makes up 37% of the full dataset, it will also make up approximately 37% of the 10,000-cell sample.

Two sampling methods are shown side by side:
- **Method A** — a custom function that calculates the proportional sample size for each class explicitly
- **Method B** — the same result using sklearn's `train_test_split` with `test_size=10000` and `stratify=y`

Both methods produce the same outcome. Method A is shown first because it makes the proportional logic explicit.

**Step 3 — Train/test split**  
The 10,000-cell sample is split 80% training / 20% test using `train_test_split` with `stratify=y_sub` to maintain class proportions in both sets, and `random_state=42` to make the split reproducible.

**Step 4 — Feature reduction pipeline**

The 2,358 gene features are reduced in four steps:

| Step | Method | Threshold / Setting |
|---|---|---|
| 1 | Zero-variance removal | Exact variance = 0 |
| 2 | High-null removal | > 90% missing values |
| 3 | StandardScaler | mean = 0, std = 1 |
| 4 | VarianceThreshold | threshold = 0.01 |
| 5 | RFE | best k found by sweep |

**Zero-variance removal:** Any gene that has the exact same value in every single training cell is removed. It carries no information whatsoever.

**StandardScaler:** Rescales each gene's values so they have a mean of 0 and a standard deviation of 1 across all training cells. This ensures that genes with naturally high count values do not dominate the variance calculation in the next step.

**VarianceThreshold:** Even after scaling, genes that are near-zero in most cells will still have very low variance. `VarianceThreshold(threshold=0.01)` removes any gene whose variance across all training cells falls below 0.01. The method `get_support()` returns a True/False array (True = keep, False = remove), and this mask is applied identically to both training and test sets.

**RFE (Recursive Feature Elimination):** RFE ranks genes by how useful they are for classification and removes the least useful ones, repeating the process until a target number of features (`k`) is reached. To find the best `k`, a sweep is run over candidate values generated by starting at `N // 4` (one quarter of the features remaining after VarianceThreshold) and halving each time. For each `k`, a lightweight Random Forest is used to select the features and a slightly larger Random Forest is used to evaluate them. The result is plotted as a line chart of weighted F1 score vs number of features, with the best `k` marked by a red dashed vertical line. The final RFE is then re-run with the best `k`.

**Why is a Random Forest used inside RFE if the final models are KNN and SVM?**  
RFE needs a model that can rank features by importance. KNN computes distances rather than importance scores, so it cannot be placed inside RFE. SVM with an RBF kernel also does not produce feature importance scores in a form that RFE can use directly. A lightweight Random Forest is used purely as a feature ranker inside RFE — it is fast and produces reliable gene importance scores. The genes selected by this process are then passed to KNN and SVM for the actual classification. This is standard practice in applied machine learning.

**Step 5 — Save outputs**  
Saves all four files: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.

---

## Notebook 3 — K-Nearest Neighbours

**File:** `03_knn.ipynb`  
**Input:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

### How KNN works

K-Nearest Neighbours is one of the most intuitive machine learning algorithms. It does not learn a model in the traditional sense — it simply stores the entire training set. To classify a new cell, it:

1. Calculates the distance between the new cell and every cell in the training set, using the gene expression values as coordinates in a high-dimensional space
2. Finds the `k` closest training cells (the nearest neighbours)
3. Looks at what cell type those `k` neighbours are labelled as
4. Assigns the class that appears most frequently among them

The idea behind using KNN for gene expression data is that cells of the same type tend to express the same genes at similar levels. In other words, they cluster together in the gene expression space. KNN exploits this directly.

**`weights='distance'`:** By default, all `k` neighbours get an equal vote. With `weights='distance'`, closer neighbours have more influence than distant ones. A cell at distance 0.1 has more voting power than one at distance 2.0. This helps with the class imbalance in this dataset — without distance weighting, the majority class cells can outnumber minority class cells among the neighbours and dominate the vote.

### Hyperparameter tuning — RandomizedSearchCV

The Sepsis project (a companion project in this series) used `GridSearchCV`, which tests every possible combination of hyperparameter values. Here we use **RandomizedSearchCV**, which randomly samples a fixed number of combinations from the search space.

For KNN, this is the better choice because it allows us to search a much wider range for `n_neighbors` (any integer from 1 to 30, rather than just a few hand-picked values) without having to test every single one. With `n_iter=20`, we test 20 randomly drawn combinations — far fewer than a full grid, but usually enough to find a good configuration.

**Search space:**

| Hyperparameter | Range | Description |
|---|---|---|
| `n_neighbors` | 1 to 30 (integers) | How many nearest neighbours vote on the class — smaller k follows the data more closely, larger k gives smoother boundaries |
| `weights` | 'uniform', 'distance' | Whether all neighbours vote equally or closer ones count more |
| `metric` | 'euclidean', 'manhattan', 'chebyshev' | How distance between cells is calculated |

**Total fits:** 20 combinations × 5 cross-validation folds = 100 model fits

### Evaluation

After tuning, the best model is evaluated on the held-out test set. The following outputs are produced:

- **Classification report:** Precision, recall, F1 score, and support for every class individually, plus weighted averages
- **Confusion matrix:** A grid showing how many cells of each true type were predicted as each type — diagonal cells are correct predictions, off-diagonal cells are misclassifications
- **ROC curves (one-vs-rest):** For each of the top 10 most frequent classes, a curve showing the trade-off between true positive rate and false positive rate. AUC (Area Under the Curve) close to 1.0 means the model distinguishes that class well from all others
- **Per-class F1 bar chart:** A horizontal bar chart showing the F1 score for each class individually, sorted from lowest to highest, with a line showing the mean F1 score across all classes — this immediately reveals which cell types the model finds hardest to classify

---

## Notebook 4 — Support Vector Machine and Model Comparison

**File:** `04_svm.ipynb`  
**Input:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

### How SVM works

A Support Vector Machine finds the decision boundary — called a **hyperplane** — that best separates the classes from each other. Unlike KNN which uses all training cells, SVM focuses specifically on the training cells that sit closest to the decision boundary. These are the **support vectors**, and the key idea is to find the boundary that maximises the gap (margin) between the nearest cells of each class.

For multiclass problems, sklearn's SVC uses a **one-vs-one (OvO)** strategy: it trains a separate binary classifier for every pair of classes (e.g. Proximal Tubule vs T cells, Proximal Tubule vs Endothelium, T cells vs Endothelium, and so on), then combines all their votes to make the final prediction.

**`kernel` parameter:** The kernel determines the shape of the decision boundary.
- `'rbf'` (Radial Basis Function) — maps the data into a higher-dimensional space where a linear boundary can separate classes that were not linearly separable in the original space. Generally the most powerful option for complex data.
- `'linear'` — finds a straight-line (or hyperplane) boundary. This actually works very well for high-dimensional data like gene expression, where the large number of features already gives the model a lot of flexibility even with a linear boundary.

**`class_weight='balanced'`:** Automatically adjusts the penalty for misclassifying each class based on how frequent it is. Cells from rare classes receive a higher penalty when misclassified, so the model cannot ignore them in favour of the dominant Proximal Tubule class.

**`probability=True`:** Enables `predict_proba()`, which is needed to plot ROC curves. SVM does not naturally produce probabilities — enabling this setting applies an additional step called Platt scaling, which takes the SVM's raw scores and converts them to probabilities. This adds a small amount of training time.

### Hyperparameter tuning — Bayesian Optimisation

While Notebook 3 used `RandomizedSearchCV` (random sampling), Notebook 4 uses **Bayesian Optimisation** via `BayesSearchCV` from the `scikit-optimize` library.

The fundamental difference is that RandomizedSearchCV has no memory — each trial is sampled independently, with no information from previous trials. Bayesian optimisation is smarter:

1. It runs a few initial random trials to gather early information about the search space
2. It builds a **surrogate model** — a probabilistic model that estimates how the score is likely to vary across the search space based on the results so far
3. It uses the surrogate model to decide which combination to try next, focusing on regions of the search space that look promising
4. After each new trial, it updates the surrogate model and repeats

This is particularly valuable for SVM because `C` and `gamma` are **continuous parameters** — they can take any positive value, and good values typically span several orders of magnitude (e.g. 0.01, 0.1, 1.0, 10, 100). Bayesian search navigates this continuous space intelligently, while random search just samples blindly.

**Search space:**

| Hyperparameter | Range | Scale | Description |
|---|---|---|---|
| `C` | 0.01 – 100.0 | Log-uniform | Regularisation strength — higher C = the model tries harder to correctly classify every training cell, at the risk of overfitting |
| `kernel` | 'rbf', 'linear' | Categorical | Type of decision boundary |
| `gamma` | 0.0001 – 1.0 | Log-uniform | RBF kernel only — how far the influence of each training cell reaches; small gamma = broad influence, large gamma = narrow influence |

Log-uniform means values are sampled evenly on a log scale, so 0.01, 0.1, 1.0, and 10.0 are all equally likely to be tried. This is the appropriate prior when good values can span several orders of magnitude.

**Total fits:** 20 Bayesian trials × 5 cross-validation folds = 100 model fits

**Installation:** `scikit-optimize` is not pre-installed in Google Colab. A pip install cell at the start of the tuning section handles this automatically.

### Evaluation

The same evaluation suite as Notebook 3 is produced:
- Classification report
- Confusion matrix (orange colour scheme)
- ROC curves (one-vs-rest, top 10 classes)
- Per-class F1 bar chart

### Model Comparison — KNN vs SVM

The final section of Notebook 4 compares both models side by side. KNN is re-run using the best hyperparameters found in Notebook 3 (which you paste in manually), and then four metrics are compared:

| Metric | What it measures |
|---|---|
| Weighted F1 | Harmonic mean of precision and recall, averaged across all classes weighted by class size — the primary metric |
| ROC-AUC | Area under the ROC curve (weighted one-vs-rest) — measures overall discriminative ability |
| Precision | Of all cells the model predicted as a given type, how many actually were that type |
| Recall | Of all cells that actually were a given type, how many did the model correctly identify |

The comparison is shown in:
1. A printed table with all four metrics for both models
2. A grouped bar chart with blue bars for KNN and orange bars for SVM
3. Overlaid ROC curves for the largest class only (Proximal Tubule), showing both models on the same plot for a direct head-to-head comparison

---

## Key Design Decisions

### Why these two models?

**KNN** was chosen because it is intuitive (the concept of finding nearest neighbours in gene expression space maps directly to the biology), it requires no assumptions about the data distribution, and it contrasts clearly with SVM. It also connects naturally to the idea of cells of the same type clustering together.

**SVM** was chosen because it is genuinely powerful for high-dimensional data. With hundreds of gene features after RFE, SVM's ability to find complex decision boundaries in high-dimensional space makes it well-suited to this problem. It also introduces students to Bayesian hyperparameter search, which is more sophisticated than the random search used for KNN.

### Why not use the full 60,000 cells?

RFE trains and evaluates a model many times internally — once for each candidate `k` value, and once inside each cross-validation fold. With 60,000 training cells this would take many hours on Google Colab. The stratified 10,000-cell sample preserves the class proportions, meaning the model still sees the same relative balance of cell types — just less of each.

### Why weighted F1 and not accuracy?

The dataset is heavily imbalanced. Proximal Tubule cells make up 37% of the data. A model that always predicted "Proximal Tubule" would achieve 37% accuracy without learning anything meaningful. Weighted F1 score accounts for this by averaging F1 scores across all classes, weighted by class size, so the model is judged on how well it performs across all cell types.

### Why is the same train/test split used for both models?

Using the same `X_train.csv`, `X_test.csv`, `y_train.csv`, and `y_test.csv` files for both Notebook 3 and Notebook 4 ensures that the comparison between KNN and SVM is fair. Both models see exactly the same training data and are evaluated on exactly the same test data.

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `pandas` | any recent | Data loading, manipulation, and saving |
| `numpy` | any recent | Numerical operations and array handling |
| `matplotlib` | any recent | All plots and visualisations |
| `seaborn` | any recent | Plot styling |
| `scikit-learn` | ≥ 1.0 | KNN, SVM, preprocessing, feature selection, metrics |
| `scipy` | any recent | `randint` distribution for RandomizedSearchCV |
| `scikit-optimize` | ≥ 0.9 | `BayesSearchCV` for Bayesian hyperparameter search (Notebook 4 only) |

All libraries except `scikit-optimize` are pre-installed in Google Colab. `scikit-optimize` is installed automatically by a cell in Notebook 4 at the start of the tuning section.

---

## Results Summary

Results will vary depending on which input file is used (`kidney_cells_clean.csv` vs `kidney_cells_top_classes.csv`), the random seed, and the hyperparameters found by the search. The tables below will be filled in after running the notebooks.

### Feature Reduction (Notebook 2)

| Stage | Features Remaining |
|---|---|
| Original gene columns | 2,358 |
| After zero-variance removal | — |
| After high-null removal (>90%) | — |
| After VarianceThreshold (0.01) | — |
| After RFE | — |

### Model Performance (Notebooks 3 and 4)

| Model | Weighted F1 | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| KNN (baseline) | — | — | — | — |
| KNN (tuned, RandomizedSearch) | — | — | — | — |
| SVM (baseline) | — | — | — | — |
| SVM (tuned, BayesSearch) | — | — | — | — |

---

*This project is part of a series of applied machine learning projects designed for students. A companion project applying Random Forest and XGBoost to sepsis prediction is available in the Sepsis-ML-Model repository.*
