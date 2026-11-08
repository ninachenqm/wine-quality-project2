"""Model: Decision Tree

Hyperparameter: max_depth = how many levels the tree can have.
Small depth = simple tree. Big depth (or None = no limit) = complex tree that can overfit.

Function call (Part B Q2):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train); model.predict(X_val)

Run:  python model_tree.py
"""
from sklearn.tree import DecisionTreeClassifier

from common import RANDOM_STATE, run_model

MODEL_NAME = "Decision Tree"
PARAM_NAME = "max_depth"
VALUES = [1, 2, 3, 5, 8, 12, None]   # None = no limit


def build_model(value):
    return DecisionTreeClassifier(max_depth=value, random_state=RANDOM_STATE)


if __name__ == "__main__":
    run_model(MODEL_NAME, build_model, PARAM_NAME, VALUES)