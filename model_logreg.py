"""Model: Logistic Regression (example; copy this for your own models).

Hyperparameter: C controls the L2 penalty.
Small C = strong penalty (simpler model). Big C = weak penalty.

Function call (Part B Q2), Python / scikit-learn:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1.0, max_iter=2000)   # L2 penalty is the default
    model.fit(X_train, y_train); model.predict(X_val)
(make_pipeline in common.py adds StandardScaler, and PCA for Part C.)

Run:  .venv/bin/python model_logreg.py
Runs Part B and Part C for both problems.
"""
from sklearn.linear_model import LogisticRegression

from common import RANDOM_STATE, run_model

MODEL_NAME = "Logistic Regression"
PARAM_NAME = "C"
VALUES = [0.001, 0.01, 0.1, 1, 10, 100]   # ordered simple -> complex


def build_model(C):
    # L2 is the default penalty.
    return LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE)


if __name__ == "__main__":
    run_model(MODEL_NAME, build_model, PARAM_NAME, VALUES)
