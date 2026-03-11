import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath='data/data.xlsx'):
    """Loads the dataset from the Excel file."""
    df = pd.read_excel(filepath)
    return df

def preprocess_data(df):
    """
    Preprocesses the data:
    1. Extracts features (X) and target (y).
    2. Drops constant or redundant features: 'Im(neff)', 'dc (um)', 'Re(eff)'.
    3. Applies log10 to the target ('loss').
    4. Scales the features using MinMaxScaler.
    """
    # 1. Drop redundant/constant features
    cols_to_drop = ['loss', 'Im(neff)', 'dc (um)', 'Re(eff)']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    # Identify geometric configuration for grouping
    config_cols = ['Pitch (um)', 'd1 (um)', 'd2 (um)', 'd3 (um)']
    # Create unique group IDs based on geometric configurations
    df_configs = df[config_cols].drop_duplicates().reset_index(drop=True)
    df_configs['group_id'] = range(len(df_configs))
    
    # Merge group IDs back to the main dataframe
    df = df.merge(df_configs, on=config_cols, how='left')
    group_ids = df['group_id'].values
    
    X = df.drop(columns=existing_cols_to_drop + ['group_id'])
    
    # 2. Handle target: Multiply by 10^8 then log10, exactly as the authors did
    y_raw = df['loss']
    y = np.log10((y_raw * (10**8)).clip(lower=1e-10))
    
    # 3. Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled_df, y, group_ids, scaler

def get_logo_folds(X, y, groups):
    """
    Returns a Leave-One-Group-Out cross-validation generator.
    Allows testing on completely new geometric configurations.
    """
    logo = LeaveOneGroupOut()
    return logo.split(X, y, groups)

if __name__ == '__main__':
    df = load_data()
    print("Data loaded. Shape:", df.shape)
    X, y, groups, scaler = preprocess_data(df)
    print("Features (X):", X.columns.tolist())
    print(f"Number of groups: {len(np.unique(groups))}")
    
    # Simple check for LOGO split
    logo = get_logo_folds(X, y, groups)
    train_idx, test_idx = next(logo)
    print(f"First fold: Train size={len(train_idx)}, Test size={len(test_idx)}")
    
    # Verify that test configuration is not in train
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]
    print(f"Unique train groups: {np.unique(train_groups)}")
    print(f"Unique test groups: {np.unique(test_groups)}")
