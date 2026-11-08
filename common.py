"""Shared code for Project 2.

We both import this file, so every model uses the same data, folds, plots and scores.
Tell your partner before you change the settings below.
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

# ---------------------------------------------------------------- settings
RANDOM_STATE = 42      # same seed everywhere
VAL_FRACTION = 0.10    # validation set: never used for training or tuning
N_FOLDS = 5            # k in k-fold CV
GOOD_THRESHOLD = 7     # quality >= 7 means good wine (label 1)
N_COMPONENTS = 5       # Part C: PCA goes from 11 features to 5
SELECT_METRIC = "f1"   # pick hyperparameters by CV F1 (classes are unbalanced)

TARGET = "good"
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
PROBLEMS = {
    "red": "Problem 1: Is this red wine good?",
    "white": "Problem 2: Is this white wine good?",
}

# ---------------------------------------------------------------- plot style
WINE_COLORS = {"red": "#e34948", "white": "#eda100"}   # entity colors, always with a legend
SERIES = ["#2a78d6", "#eb6834"]                        # F1 line, accuracy line
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def slug(text):
    return text.lower().replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------- data
def load_split(problem, part="B"):
    """Return X_train, y_train, X_val, y_val for 'red' or 'white'.

    Part B and Part C use the same files. For Part C, PCA is inside the
    pipeline, so it only learns from the training folds.
    """
    train = pd.read_csv(SPLITS / f"{problem}_train.csv")
    val = pd.read_csv(SPLITS / f"{problem}_val.csv")
    return train[FEATURES], train[TARGET], val[FEATURES], val[TARGET]


def make_pipeline(model, part="B"):
    """Standardize -> (PCA for Part C) -> model."""
    steps = [("scale", StandardScaler())]
    if part == "C":
        steps.append(("pca", PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


# ---------------------------------------------------------------- cross-validation
def cv_grid(build_model, param_name, values, X, y, part="B"):
    """k-fold CV for every hyperparameter value. Returns one row per value."""
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for v in values:
        scores = cross_validate(
            make_pipeline(build_model(v), part), X, y,
            cv=folds, scoring=["accuracy", "f1"], n_jobs=-1,
        )
        rows.append({
            param_name: v,
            "f1_mean": scores["test_f1"].mean(),
            "f1_std": scores["test_f1"].std(),
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "fit_time_mean_s": scores["fit_time"].mean(),
        })
    df = pd.DataFrame(rows)
    df[param_name] = pd.Series(list(values), dtype=object)  # keep 1 as 1 (not 1.0) and None as None
    return df


def pick_best(cv_df, param_name):
    """Pick the value with the best CV F1. If tied, take the first (simpler) one."""
    best_row = cv_df.loc[cv_df[f"{SELECT_METRIC}_mean"].idxmax()]
    return best_row[param_name], best_row


def plot_cv(cv_df, param_name, model_name, problem, part, path):
    """Mean +/- 1 std of CV F1 and accuracy for each hyperparameter value."""
    labels = [str(v) for v in cv_df[param_name]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for metric, color, name in [("f1", SERIES[0], "F1 (good wine)"),
                                ("accuracy", SERIES[1], "Accuracy")]:
        m, s = cv_df[f"{metric}_mean"].to_numpy(), cv_df[f"{metric}_std"].to_numpy()
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, m, color=color, marker="o", markersize=6, label=f"{name}, mean ± 1 std")
        ax.annotate(name, (x[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)

    best_value, best_row = pick_best(cv_df, param_name)
    bi = labels.index(str(best_value))
    ax.plot(x[bi], best_row["f1_mean"], "o", markersize=13, markerfacecolor="none",
            markeredgecolor=INK, markeredgewidth=1.5)
    ax.annotate(f"best {param_name} = {best_value}\nF1 = {best_row['f1_mean']:.3f}",
                (x[bi], best_row["f1_mean"]), xytext=(0, -34), textcoords="offset points",
                ha="center", fontsize=9, color=INK)

    ax.set_xticks(x, labels)
    ax.set_xlabel(param_name)
    ax.set_ylabel(f"score ({N_FOLDS}-fold CV)")
    ax.set_ylim(0, 1.02)
    stage = "Part B, all 11 features" if part == "B" else f"Part C, PCA → {N_COMPONENTS} components"
    ax.set_title(f"{model_name}: {PROBLEMS[problem]}\n{stage}", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- final fit + validation
def final_evaluate(build_model, best_value, X_tr, y_tr, X_val, y_val, part="B"):
    """Train on all training data with the best value, then test on the validation set."""
    pipe = make_pipeline(build_model(best_value), part)

    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = pipe.predict(X_val)
    test_time = time.perf_counter() - t0

    train_pred = pipe.predict(X_tr)
    return {
        "train_time_s": train_time,
        "test_time_s": test_time,
        "train_accuracy": accuracy_score(y_tr, train_pred),
        "train_f1": f1_score(y_tr, train_pred, zero_division=0),
        "val_accuracy": accuracy_score(y_val, pred),
        "val_balanced_accuracy": balanced_accuracy_score(y_val, pred),
        "val_precision": precision_score(y_val, pred, zero_division=0),
        "val_recall": recall_score(y_val, pred, zero_division=0),
        "val_f1": f1_score(y_val, pred, zero_division=0),
        "val_confusion_matrix [[TN, FP], [FN, TP]]": confusion_matrix(y_val, pred).tolist(),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_val)),
    }


# ---------------------------------------------------------------- one call per model
def run_model(model_name, build_model, param_name, values,
              problems=("red", "white"), parts=("B", "C")):
    """For each problem and part: run CV, plot it, pick the best value, then train and test.

    Saves plots to figures/, results to results/, and updates results/summary.csv.
    """
    for part in parts:
        for problem in problems:
            X_tr, y_tr, X_val, y_val = load_split(problem, part)
            cv_df = cv_grid(build_model, param_name, values, X_tr, y_tr, part)
            best_value, best_row = pick_best(cv_df, param_name)

            fig_path = FIGURES / f"part{part}" / f"{problem}_{slug(model_name)}_cv.png"
            plot_cv(cv_df, param_name, model_name, problem, part, fig_path)
            final = final_evaluate(build_model, best_value, X_tr, y_tr, X_val, y_val, part)

            record = {
                "model": model_name, "problem": problem, "part": part,
                "hyperparameter": param_name,
                "values_tested": [str(v) for v in values],
                "best_value": str(best_value),
                "best_cv_f1": best_row["f1_mean"],
                "best_cv_accuracy": best_row["accuracy_mean"],
                "cv_table": cv_df.astype({param_name: str}).to_dict(orient="records"),
                **final,
            }
            RESULTS.mkdir(exist_ok=True)
            out = RESULTS / f"part{part}_{problem}_{slug(model_name)}.json"
            out.write_text(json.dumps(record, indent=2, default=float))

            print(f"[Part {part} | {problem:5s} | {model_name}] best {param_name}={best_value}  "
                  f"CV F1={best_row['f1_mean']:.3f}  |  val acc={final['val_accuracy']:.3f}  "
                  f"val F1={final['val_f1']:.3f}  train {final['train_time_s']*1000:.1f} ms  "
                  f"test {final['test_time_s']*1000:.2f} ms")
    summarize_results()


def summarize_results():
    """Put all results/*.json (from both of us) into results/summary.csv."""
    rows = []
    for f in sorted(RESULTS.glob("part*_*.json")):
        r = json.loads(f.read_text())
        rows.append({k: r[k] for k in [
            "part", "problem", "model", "hyperparameter", "best_value", "best_cv_f1",
            "val_accuracy", "val_balanced_accuracy", "val_precision", "val_recall",
            "val_f1", "train_time_s", "test_time_s"]})
    if rows:
        df = pd.DataFrame(rows).sort_values(["part", "problem", "model"])
        df.round(4).to_csv(RESULTS / "summary.csv", index=False)
    return rows
