# ml_models.py

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score

def train_linear_regression(X, y):
    """Trains a linear regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_logistic_regression(X, y):
    """Trains a logistic regression model."""
    model = LogisticRegression()
    model.fit(X, y)
    return model

def train_kmeans(X, n_clusters=3):
    """Trains a KMeans clustering model."""
    model = KMeans(n_clusters=n_clusters)
    model.fit(X)
    return model

def evaluate_model(model, X_test, y_test, task_type='regression'):
    """Evaluates model performance for regression or classification tasks."""
    y_pred = model.predict(X_test)
    if task_type == 'regression':
        return mean_squared_error(y_test, y_pred)
    elif task_type == 'classification':
        return accuracy_score(y_test, y_pred)
