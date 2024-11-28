import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

def score_dataset(X, y, model=XGBRegressor()):
    """Use cross-validation and returns the Root Mean Squared Log Error.

    Parameters
    ----------
    X : DataFrame
      DataFrame containing the all the available features.
    y : Series
      Series containing data in the target feature.
    model : Model
      Model we want to test.

    Returns
    -------
    Float
      Root Mean Squared Log Error from using the model.
    """
    X_copy = X.copy()
    # Encode categorical labels.
    for colname in X_copy.select_dtypes(["category", "object"]):
        X_copy[colname], _ = X_copy[colname].factorize()
    # Use cross-validation and compute RMSLE.
    score = cross_val_score(model, X, y, cv=5, scoring="negative_mean_squared_log_error")
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score
