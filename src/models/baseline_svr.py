import os
import sys

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data import load_data, preprocess_data, get_train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_baseline_svr(X_train, y_train):
    """
    Trains a baseline Support Vector Regressor (SVR) on the provided data.
    Uses GridSearchCV to find basic optimal hyperparameters.
    """
    # Define a basic parameter grid for SVR
    param_grid = {
        'kernel': ['rbf', 'linear', 'poly'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'epsilon': [0.01, 0.1, 0.5]
    }
    
    svr = SVR()
    print("Starting Grid Search for Baseline SVR...")
    
    # 5-fold cross-validation
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name="SVR"):
    """Evaluates the model and prints the MSE."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"--- {model_name} Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    return mse

if __name__ == '__main__':
    print("Loading data for baseline training...")
    df = load_data()
    X, y, scaler = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    best_svr_model = train_baseline_svr(X_train, y_train)
    evaluate_model(best_svr_model, X_test, y_test, "Baseline SVR")
