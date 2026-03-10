import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath='data/data.xlsx'):
    """Loads the dataset from the Excel file."""
    df = pd.read_excel(filepath)
    return df

def preprocess_data(df):
    """
    Preprocesses the data:
    1. Extracts features (X) and target (y).
    2. Applies log10 to the target ('loss') as mentioned in the paper.
    3. Scales the features using StandardScaler.
    """
    # The paper's target is confinement loss. The other columns are features.
    # Note: 'Im(neff)' is directly related to loss mathematically, 
    # so we should likely drop it from features to prevent data leakage.
    
    if 'Im(neff)' in df.columns:
        df = df.drop(columns=['Im(neff)'])
        
    X = df.drop(columns=['loss'])
    # Handle any potential zero or negative loss values before log10
    # To avoid log(0) issues, we can add a very small constant or filter
    y_raw = df['loss']
    y = np.log10(y_raw.clip(lower=1e-10))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Keep feature names for later analysis
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled_df, y, scaler

def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """Splits the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == '__main__':
    df = load_data()
    print("Data loaded successfully. Shape:", df.shape)
    X, y, scaler = preprocess_data(df)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
