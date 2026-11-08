# Project 2: Wine Quality

**Data:** UCI Wine Quality (red and white wine, 11 features, quality score 0–10)

- **Problem 1:** Is this red wine good? (quality ≥ 7 = good)
- **Problem 2:** Is this white wine good? (quality ≥ 7 = good)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The data split is already done. **Don't re-split the data.** We both need the same validation set.

## Files

| File | What it does |
|---|---|
| `common.py` | Shared code: settings, CV, plots, timing, scores |
| `01_prepare_data.py` | Clean data, make the "good" label, hold out 10% |
| `02_part_a_eda.py` | Part A plots |
| `03_part_c_pca.py` | Part C: PCA (11 features → 5) |
| `model_logreg.py` | Example model: Logistic Regression |
| `model_TEMPLATE.py` | Copy this to add a new model |

## Who does what

| Nina | Bobby |
|---|---|
| Logistic Regression (`C`) ✅ | KNN (`n_neighbors`) |
| Decision Tree (`max_depth`) | SVM (`C`) |

## Add a model

1. `cp model_TEMPLATE.py model_knn.py`
2. Fill in the 4 TODOs
3. `.venv/bin/python model_knn.py`

This runs Part B and Part C for both problems. It saves plots to `figures/`, results to `results/`, and updates `results/summary.csv`.

## Key facts

- 6497 wines, 5320 after removing duplicates (red 1359, white 3961)
- Good wines: red 13.5%, white 20.8%. Classes are unbalanced, so we pick hyperparameters by **F1**, not accuracy.
- Train / validation: red 1223 / 136, white 3564 / 397
- PCA with 5 components keeps 80% (red) and 73% (white) of the variance.
