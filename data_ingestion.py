# data_ingestion.py

import pandas as pd

def load_csv(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def split_features_target(df, target_column):
    """
    Splits the DataFrame into features (X) and target (y) based on the target_column.
    """
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    else:
        raise ValueError("Target column not found in DataFrame.")
