import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_and_split_data(parquet_file):
    # Load the parquet file
    df = pd.read_parquet(parquet_file)
    
    # Get unique timestamps and sort them
    timestamps = sorted(df.index.unique())
    total_timestamps = len(timestamps)
    
    # Verify we have 60 timestamps
    assert total_timestamps == 60, f"Expected 60 timestamps, got {total_timestamps}"
    
    # Split timestamps into train/val/test
    train_timestamps = timestamps[:40]  # First 40 timestamps
    val_timestamps = timestamps[40:60]  # Last 20 timestamps
    
    # Split the data
    train_data = df.loc[train_timestamps]
    val_data = df.loc[val_timestamps]
    
    return train_data, val_data

def perform_pca_analysis(train_data, val_data, target_column):
    # Separate features (X) and target (y)
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_val = val_data.drop(columns=[target_column])
    y_val = val_data[target_column]
    
    # Get number of features
    n_features = X_train.shape[1]
    
    # Track RMSE for different PCA component choices
    rmse_per_component = []
    numpc = range(1, n_features + 1)
    
    # Try each candidate number of principal components
    for n_components in numpc:
        # Step 1: Fit PCA on training data
        pca = PCA(n_components=n_components)
        X_train_reduced = pca.fit_transform(X_train)
        X_val_reduced = pca.transform(X_val)
        
        # Step 2: Fit linear regression on reduced training data
        regressor = LinearRegression()
        regressor.fit(X_train_reduced, y_train)
        
        # Step 3: Predict on validation data and calculate RMSE
        y_val_pred = regressor.predict(X_val_reduced)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse_per_component.append(rmse)
    
    # Find optimal number of components
    optimal_idx = np.argmin(rmse_per_component)
    optimal_n_components = numpc[optimal_idx]
    
    # Plot RMSE vs number of components
    plt.figure(figsize=(10, 6))
    plt.plot(numpc, rmse_per_component, 'bo-')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Number of Principal Components')
    plt.axvline(x=optimal_n_components, color='r', linestyle='--', 
                label=f'Optimal components: {optimal_n_components}')
    plt.legend()
    plt.grid(True)
    plt.savefig('pca_rmse_plot.png')
    plt.close()
    
    return optimal_n_components, rmse_per_component

def main():
    # TODO: Replace with your parquet file path
    parquet_file = "path_to_your_parquet_file.parquet"
    target_column = "your_target_column"  # Replace with your target column name
    
    # Load and split data
    train_data, val_data = load_and_split_data(parquet_file)
    
    # Perform PCA analysis
    optimal_n_components, rmse_per_component = perform_pca_analysis(
        train_data, val_data, target_column
    )
    
    print(f"Optimal number of components: {optimal_n_components}")
    print(f"Minimum RMSE achieved: {min(rmse_per_component):.4f}")

if __name__ == "__main__":
    main() 