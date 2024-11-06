# data_transformation.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def handle_missing_values(df, method='drop', value=None):
    """Handles missing values in the DataFrame."""
    if method == 'drop':
        df = df.dropna()
    elif method == 'fill':
        if value is not None:
            df = df.fillna(value)
        else:
            raise ValueError("Value required for filling.")
    elif method == 'interpolate':
        df = df.interpolate()
    return df

def scale_data(df, columns):
    """Scales specified columns using StandardScaler."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def encode_categorical(df, columns):
    """Encodes categorical columns using LabelEncoder."""
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def remove_outliers(df, z_threshold=3):
    """Removes outliers based on Z-score threshold."""
    return df[(df.apply(lambda x: (x - x.mean()).abs() / x.std() < z_threshold).all(axis=1))]
