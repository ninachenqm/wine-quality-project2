"""Step 2: Part A plots.

Run:  .venv/bin/python 02_part_a_eda.py
Makes:
  figures/partA/feature_histograms.png    -> Part A Q3
  figures/partA/quality_distribution.png  -> Part A Q3 / Q4.1
  figures/partA/feature_vs_good.png       -> for the presentation
  results/partA_feature_summary.csv       -> skewness, to answer "normally distributed?"
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FEATURES, FIGURES, GOOD_THRESHOLD, INK_2, RESULTS, ROOT, TARGET, WINE_COLORS, load_split

out = FIGURES / "partA"
out.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "wine_clean_all.csv")
colors = ["red", "white"]

# ------------------------------------------------ 1. histograms of every variable
fig, axes = plt.subplots(3, 4, figsize=(13, 8.5))
for ax, col in zip(axes.flat, FEATURES + ["quality"]):
    lo, hi = df[col].quantile([0.005, 0.995])           # cut extreme values so the plot is easy to read
    if col == "quality":
        levels = np.arange(df[col].min(), df[col].max() + 1)
        w = 0.4
        for i, c in enumerate(colors):
            share = df[df.color == c][col].value_counts(normalize=True).reindex(levels, fill_value=0)
            ax.bar(levels + (i - 0.5) * w, share, width=w - 0.04, color=WINE_COLORS[c], label=f"{c} wine")
        ax.axvline(GOOD_THRESHOLD - 0.5, color=INK_2, linestyle="--", linewidth=1)
        ax.text(GOOD_THRESHOLD - 0.4, ax.get_ylim()[1] * 0.9, "good ≥ 7", fontsize=8, color=INK_2)
        ax.set_ylabel("share of wines")
    else:
        bins = np.linspace(lo, hi, 35)
        for c in colors:
            ax.hist(df[df.color == c][col].clip(lo, hi), bins=bins, density=True,
                    histtype="step", linewidth=2, color=WINE_COLORS[c], label=f"{c} wine")
        ax.set_yticks([])
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_title(col, loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
axes.flat[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Distribution of each variable, red vs white wine (density; extreme 0.5% tails trimmed)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(out / "feature_histograms.png")
plt.close(fig)

# ------------------------------------------------ 2. quality / class balance
fig, ax = plt.subplots(figsize=(6.4, 3.6))
levels = np.arange(3, 10)
w = 0.4
for i, c in enumerate(colors):
    counts = df[df.color == c]["quality"].value_counts().reindex(levels, fill_value=0)
    bars = ax.bar(levels + (i - 0.5) * w, counts, width=w - 0.04, color=WINE_COLORS[c], label=f"{c} wine")
    ax.bar_label(bars, fontsize=7, color=INK_2, padding=2)
top = df["quality"].value_counts().max()
ax.set_ylim(0, top * 1.22)
ax.axvline(GOOD_THRESHOLD - 0.5, color=INK_2, linestyle="--", linewidth=1)
ax.text(GOOD_THRESHOLD - 0.4, top * 1.13, "label = good (1)  →", fontsize=8, color=INK_2)
ax.text(GOOD_THRESHOLD - 0.6, top * 1.13, "←  not good (0)", fontsize=8, color=INK_2, ha="right")
ax.set_xticks(levels)
ax.set_xlabel("quality score (median of expert ratings, 0–10)")
ax.set_ylabel("number of wines")
ax.set_title("Quality scores and the 'good wine' cut-off", loc="left")
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.85), fontsize=8)
ax.grid(axis="x", visible=False)
fig.savefig(out / "quality_distribution.png")
plt.close(fig)

# ------------------------------------------------ 3. which features separate good wine?
# Use training data only, so we don't touch the validation set.
corr = {}
for c in colors:
    X_tr, y_tr, _, _ = load_split(c)
    corr[c] = X_tr.apply(lambda s: s.corr(y_tr))
corr = pd.DataFrame(corr).sort_values("red")
fig, ax = plt.subplots(figsize=(6.8, 4.6))
y = np.arange(len(corr))
h = 0.38
for i, c in enumerate(colors):
    ax.barh(y + (0.5 - i) * h, corr[c], height=h - 0.04, color=WINE_COLORS[c], label=f"{c} wine")
ax.axvline(0, color=INK_2, linewidth=1)
ax.set_yticks(y, corr.index)
ax.set_xlabel("correlation with 'good' label (training data)")
ax.set_title("What makes a good wine? Red and white don't fully agree", loc="left")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", visible=False)
fig.savefig(out / "feature_vs_good.png")
plt.close(fig)

# ------------------------------------------------ 4. summary table for Q1 / Q3
rows = []
for c in colors:
    sub = df[df.color == c]
    for col in FEATURES + ["quality"]:
        s = sub[col].skew()
        shape = ("roughly normal" if abs(s) < 0.5 else
                 "moderately skewed" if abs(s) < 1 else "strongly skewed")
        rows.append({"color": c, "variable": col, "mean": sub[col].mean(), "std": sub[col].std(),
                     "min": sub[col].min(), "max": sub[col].max(), "skewness": s, "shape": shape})
summary = pd.DataFrame(rows).round(3)
summary.to_csv(RESULTS / "partA_feature_summary.csv", index=False)

print(f"Samples: {len(df)} (red {sum(df.color == 'red')}, white {sum(df.color == 'white')}); "
      f"features: {len(FEATURES)} inputs + quality (+ color)")
print(summary.pivot(index="variable", columns="color", values="skewness"))
print("Correlation with good:\n", corr.round(3))
