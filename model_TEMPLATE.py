"""Model: <NAME>  -- copy this file to model_<name>.py and fill in the 4 TODOs.

Hyperparameter: <which one and what it controls>

Function call (Part B Q2):
    <the sklearn call you use>

Run:  .venv/bin/python model_<name>.py
"""
# TODO 1: import your sklearn model class
# e.g. from sklearn.tree import DecisionTreeClassifier
#      from sklearn.neighbors import KNeighborsClassifier
#      from sklearn.svm import SVC

from common import RANDOM_STATE, run_model

MODEL_NAME = "TODO 2: display name, e.g. Decision Tree"
PARAM_NAME = "TODO 3: hyperparameter name, e.g. max_depth"
VALUES = []  # TODO 3: values to test, ordered simple -> complex


def build_model(value):
    # TODO 4: return the model with the hyperparameter set to `value`
    # (pass random_state=RANDOM_STATE if the model has one)
    raise NotImplementedError


if __name__ == "__main__":
    run_model(MODEL_NAME, build_model, PARAM_NAME, VALUES)
