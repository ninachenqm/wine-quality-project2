"""Step 3: Part C, PCA (11 features -> 5).

Run:  .venv/bin/python 03_part_c_pca.py
Makes:
  data/splits_pca/{red,white}_{train,val}_pca.csv -> Part C Q3.1 / Q4.1
  figures/partC/pca_explained_variance.png        -> Part C Q1
  results/pca_loadings_{red,white}.csv            -> which features each PC uses

PCA learns from the training data only, then is applied to the validation data.

Function call (Part C Q2):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5).fit(StandardScaler().fit_transform(X_train))
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import FEATURES, FIGURES, INK_2, N_COMPONENTS, RANDOM_STATE, RESULTS, ROOT, TARGET, WINE_COLORS, load_split

out_dir = ROOT / "data" / "splits_pca"
out_dir.mkdir(parents=True, exist_ok=True)
(FIGURES / "partC").mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(6.4, 3.8))
for color in ["red", "white"]:
    X_tr, y_tr, X_val, y_val = load_split(color)
    scaler = StandardScaler().fit(X_tr)

    full = PCA(random_state=RANDOM_STATE).fit(scaler.transform(X_tr))
    cum = np.cumsum(full.explained_variance_ratio_)
    k = np.arange(1, len(cum) + 1)
    ax.plot(k, cum, marker="o", markersize=6, color=WINE_COLORS[color], label=f"{color} wine")
    ax.annotate(f"{cum[N_COMPONENTS - 1]:.0%}", (N_COMPONENTS, cum[N_COMPONENTS - 1]),
                xytext=(-10, 8) if color == "red" else (10, -14), textcoords="offset points",
                ha="right" if color == "red" else "left", fontsize=9, color=INK_2)

    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE).fit(scaler.transform(X_tr))
    cols = [f"PC{i}" for i in range(1, N_COMPONENTS + 1)]
    for name, X, y in [("train", X_tr, y_tr), ("val", X_val, y_val)]:
        pcs = pd.DataFrame(pca.transform(scaler.transform(X)), columns=cols)
        pcs[TARGET] = y.to_numpy()
        pcs.to_csv(out_dir / f"{color}_{name}_pca.csv", index=False)

    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=cols).round(3)
    loadings.to_csv(RESULTS / f"pca_loadings_{color}.csv")
    print(f"{color}: {N_COMPONENTS} components keep {cum[N_COMPONENTS - 1]:.1%} of the variance")
    for pc in cols[:2]:
        top = loadings[pc].abs().sort_values(ascending=False).index[:3].tolist()
        print(f"   {pc} mostly mixes: {', '.join(top)}")

ax.axvline(N_COMPONENTS, color=INK_2, linestyle="--", linewidth=1)
ax.text(N_COMPONENTS + 0.15, 0.3, f"keep {N_COMPONENTS} of 11", fontsize=9, color=INK_2)
ax.set_xticks(range(1, len(FEATURES) + 1))
ax.set_xlabel("number of principal components")
ax.set_ylabel("cumulative variance explained")
ax.set_ylim(0, 1.02)
ax.set_title("PCA: how much information 5 components keep", loc="left")
ax.legend(loc="lower right", fontsize=8)
fig.savefig(FIGURES / "partC" / "pca_explained_variance.png")
plt.close(fig)
