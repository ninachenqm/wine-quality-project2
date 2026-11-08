"""Step 1: clean the data, make the two problems, hold out 10%.

Run once:  .venv/bin/python 01_prepare_data.py
Makes:
  data/wine_clean_all.csv                     -> Part A Q4.2
  data/splits/red_train.csv, red_val.csv      -> Part B Q3.1
  data/splits/white_train.csv, white_val.csv  -> Part B Q4.1
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA_RAW, GOOD_THRESHOLD, RANDOM_STATE, ROOT, SPLITS, TARGET, VAL_FRACTION

SPLITS.mkdir(parents=True, exist_ok=True)
all_clean = []

for color in ["red", "white"]:
    df = pd.read_csv(DATA_RAW / f"winequality-{color}.csv", sep=";")
    n_raw = len(df)

    # Drop duplicate rows so the same wine can't be in both train and validation.
    df = df.drop_duplicates().reset_index(drop=True)

    # 1 = good, 0 = not good. "quality" stays in the file, but it is NOT a
    # feature (it gives away the answer).
    df[TARGET] = (df["quality"] >= GOOD_THRESHOLD).astype(int)

    train, val = train_test_split(
        df, test_size=VAL_FRACTION, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    train.to_csv(SPLITS / f"{color}_train.csv", index=False)
    val.to_csv(SPLITS / f"{color}_val.csv", index=False)

    all_clean.append(df.assign(color=color))
    print(f"{color:5s}: {n_raw} rows -> {len(df)} after dropping {n_raw - len(df)} duplicates | "
          f"good = {df[TARGET].sum()} ({df[TARGET].mean():.1%}) | "
          f"train {len(train)} / validation {len(val)}")

pd.concat(all_clean, ignore_index=True).to_csv(ROOT / "data" / "wine_clean_all.csv", index=False)
print("Saved data/wine_clean_all.csv and data/splits/*.csv")
