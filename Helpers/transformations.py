import pandas as pd

def classify_categorical_columns(train_data, test_data):
    """Identifies and classifies categorical columns in a DataFrame, useful for Ordinal Encoding.

    Parameters
    ----------
    train_data : DataFrame
      DataFrame containing the training data.
    test_data : DataFrame
      DataFrame containing the testing data.

    Returns
    -------
    tuple (categorical_columns, safe_columns, unsafe_columns)
      A tuple containing the classifications of the categorical columns
      in a DataFrame relevant to Ordinal Encoding.
      
      categorical_columns : list of str
        List of all categorical columns in the training data.
      safe_columns : list of str
        List of all categorical columns that can be used for ordinal
        encoding as its values are present in both the training & testing
        data.
      unsafe_columns : list of str
        List of all categorical columns that can't be used for ordinal
        encoding due to values in the columns not being present in the
        training data.
    """
    categorical_cols = [col for col in train_data.columns if train_data[col].dtype == "object"]
    safe_cols = [col for col in categorical_cols if set(test_data[col]).issubset(set(train_data[col]))]
    unsafe_cols = list(set(categorical_cols) - set(safe_cols))
    return (categorical_cols, safe_cols, unsafe_cols)

def use_OH_encoding(OH_encoder, df, columns, training=True):
    """Applies One-Hot Encoding to the specified columns in a DataFrame.

    Parameters
    ----------
    OH_encoder : OneHotEncoder
      OneHotEncoder instance containing your configurations.
    df : DataFrame
      DataFrame which we want to apply One-Hot Encoding to.
    columns : list of str
      List containing the names of the categorical columns we want to apply
      One-Hot Encoding to.
    training : bool, default=True
      If this is used on a training dataset. If true, we use `fit_transform()`
      on the DataFrame, otherwise, we use `transform()`.

    Returns
    -------
    DataFrame
      DataFrame with the specified columns One-Hot Encoded (with the encoded
      columns removed).
    """
    encoded_cols = pd.DataFrame(
        OH_encoder.fit_transform(df[columns]) if training
        else OH_encoder.transform(df[columns])
    )
    # Re-add the index.
    encoded_cols.index = df.index
    # Remove columns that were replaced.
    numeric_df = df.drop(columns, axis=1)
    # Combine the hot-encoded & numeric columns.
    OH_X = pd.concat([numeric_df, encoded_cols], axis=1)
    # Ensure all columns have string type.
    OH_X.columns = OH_X.columns.astype(str)
    return OH_X
