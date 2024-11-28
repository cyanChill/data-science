import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y):
    """Determine the MI scores of discrete features.

    Parameters
    ----------
    X : DataFrame
      DataFrame containing the all the available features.
    y : Series
      Series containing data in the target feature.

    Returns
    -------
    Series
      Discrete features in descending order based on their MI score.
    """
    X = X.copy()
    # Label encode categorical variables.
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # Create mapping on whether a given feature is discrete (less restrictive
    # compared to `X.dtypes == int`).
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]

    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    """Plot MI scores in a horizontal bar chart.

    Parameters
    ----------
    score : Series returned from `make_mi_scores`.
    """
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
