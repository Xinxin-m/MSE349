import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def split_data(parquet_file, dropna=True):
    """
    Split data into training and validation sets based on timestamps.
    
    Parameters:
    -----------
    parquet_file : str
        Path to the parquet file
    dropna : bool, default=True
        Whether to drop rows with missing values
        
    Returns:
    --------
    tuple
        (train_data, val_data, timestamps)
    """
    # Load the parquet file
    df = pd.read_parquet(parquet_file)
    
    timestamps = df['timestamp'].unique()
    total_timestamps = len(timestamps)
    
    # Verify we have 60 timestamps
    assert total_timestamps == 60, f"Expected 60 timestamps, got {total_timestamps}"
    
    # Split timestamps into train/val
    train_timestamps = timestamps[:40]  # First 40 timestamps
    val_timestamps = timestamps[40:60]  # Last 20 timestamps
    
    if dropna:
        df = df.dropna()
    
    # Split the data
    train_data = df[df['timestamp'].isin(train_timestamps)]
    val_data = df[df['timestamp'].isin(val_timestamps)]
    
    return train_data, val_data, timestamps

def run_pca_analysis(train_data, val_data, target_column):
    """
    Run PCA analysis and return predictions.
    
    Parameters:
    -----------
    train_data, val_data : DataFrame
        Training and validation data
    target_column : str
        Name of the target column
        
    Returns:
    --------
    tuple
        (y_train, y_train_pred, y_val, y_val_pred, optimal_n_components)
    """
    # Separate features (X) and target (y)
    X_train = train_data.drop(columns=[target_column, 'timestamp', 'filename'])
    y_train = train_data[target_column]
    X_val = val_data.drop(columns=[target_column, 'timestamp', 'filename'])
    y_val = val_data[target_column]
    
    # Get number of features
    n_features = X_train.shape[1]
    
    # Track RMSE for different PCA component choices
    rmse_per_component = []
    num_pc = range(1, n_features + 1)
    
    # Try each candidate number of principal components
    for k in num_pc:
        # Step 1: Fit PCA on training data
        pca = PCA(n_components=k)
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
    opt_idx = np.argmin(rmse_per_component)
    opt_k = num_pc[opt_idx]
    
    # Get final predictions using optimal number of components
    pca = PCA(n_components=opt_k)
    X_train_reduced = pca.fit_transform(X_train)
    X_val_reduced = pca.transform(X_val)
    regressor = LinearRegression()
    regressor.fit(X_train_reduced, y_train)
    y_train_pred = regressor.predict(X_train_reduced)
    y_val_pred = regressor.predict(X_val_reduced)
    
    return y_train, y_train_pred, y_val, y_val_pred, opt_k

def analyze_model_performance(y_train, y_train_pred, y_val, y_val_pred, 
                            train_filenames, val_filenames, 
                            train_timestamps, val_timestamps,
                            output_filename):
    """
    Comprehensive analysis of model performance with combined visualization.
    
    Parameters:
    -----------
    y_train, y_train_pred : array-like
        Training actual and predicted values
    y_val, y_val_pred : array-like
        Validation actual and predicted values
    train_filenames, val_filenames : array-like
        Filenames for training and validation data
    train_timestamps, val_timestamps : array-like
        Timestamps for training and validation data
    output_filename : str
        Path to save the combined visualization
    """
    
    # Create DataFrames for analysis
    train_results = pd.DataFrame({
        'filename': train_filenames,
        'timestamp': train_timestamps,
        'actual': y_train,
        'predicted': y_train_pred,
        'error': y_train - y_train_pred,
        'squared_error': (y_train - y_train_pred) ** 2,
        'set': 'train'
    })
    
    val_results = pd.DataFrame({
        'filename': val_filenames,
        'timestamp': val_timestamps,
        'actual': y_val,
        'predicted': y_val_pred,
        'error': y_val - y_val_pred,
        'squared_error': (y_val - y_val_pred) ** 2,
        'set': 'val'
    })
    
    # Combine results
    all_results = pd.concat([train_results, val_results])
    
    # Calculate dataset-wise performance
    dataset_performance = pd.DataFrame({
        'train_mse': train_results.groupby('filename')['squared_error'].mean(),
        'val_mse': val_results.groupby('filename')['squared_error'].mean()
    })
    dataset_performance['mse_ratio'] = dataset_performance['val_mse'] / dataset_performance['train_mse']
    
    # Calculate temporal performance
    temporal_performance = all_results.groupby(['timestamp', 'set'])['squared_error'].mean().unstack()
    
    # Create combined visualization
    plt.figure(figsize=(20, 15))
    
    # 1. Dataset Performance (Top 10 by Validation MSE)
    plt.subplot(2, 2, 1)
    top_datasets = dataset_performance.sort_values('val_mse', ascending=False).head(10)
    top_datasets['val_mse'].plot(kind='bar')
    plt.title('Top 10 Datasets by Validation MSE')
    plt.xticks(rotation=45)
    plt.ylabel('MSE')
    
    # 2. Training vs Validation MSE Ratio
    plt.subplot(2, 2, 2)
    top_ratio = dataset_performance.sort_values('mse_ratio', ascending=False).head(10)
    top_ratio['mse_ratio'].plot(kind='bar')
    plt.title('Top 10 Datasets by MSE Ratio (Val/Train)')
    plt.xticks(rotation=45)
    plt.ylabel('MSE Ratio')
    
    # 3. Temporal Performance
    plt.subplot(2, 2, 3)
    temporal_performance.plot()
    plt.title('MSE Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('MSE')
    plt.legend(['Training', 'Validation'])
    plt.grid(True)
    
    # 4. Error Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(data=all_results, x='error', hue='set', bins=50, kde=True)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    
    # Calculate and return summary statistics
    summary_stats = {
        'dataset_performance': dataset_performance,
        'temporal_performance': temporal_performance,
        'error_stats': {
            'train': {
                'mean': np.mean(train_results['error']),
                'std': np.std(train_results['error']),
                'skew': scipy.stats.skew(train_results['error']),
                'kurtosis': scipy.stats.kurtosis(train_results['error'])
            },
            'val': {
                'mean': np.mean(val_results['error']),
                'std': np.std(val_results['error']),
                'skew': scipy.stats.skew(val_results['error']),
                'kurtosis': scipy.stats.kurtosis(val_results['error'])
            }
        }
    }
    
    return summary_stats

def print_performance_summary(summary_stats):
    """
    Print a summary of the model performance analysis.
    """
    print("\nModel Performance Summary:")
    print("\nTop 5 Datasets by Validation MSE:")
    print(summary_stats['dataset_performance'].sort_values('val_mse', ascending=False).head())
    
    print("\nTop 5 Datasets by MSE Ratio (Val/Train):")
    print(summary_stats['dataset_performance'].sort_values('mse_ratio', ascending=False).head())
    
    print("\nError Statistics:")
    print("\nTraining Set:")
    for stat, value in summary_stats['error_stats']['train'].items():
        print(f"{stat}: {value:.4f}")
    
    print("\nValidation Set:")
    for stat, value in summary_stats['error_stats']['val'].items():
        print(f"{stat}: {value:.4f}")

# Example usage:
if __name__ == "__main__":
    # Example parameters
    parquet_file = "USD_720_PCR/2023-01_L60.parquet"
    target_column = "log_return"
    output_filename = "model_performance_analysis.png"
    
    # Split data
    train_data, val_data, timestamps = split_data(parquet_file)
    
    # Run PCA analysis
    y_train, y_train_pred, y_val, y_val_pred, optimal_n_components = run_pca_analysis(
        train_data, val_data, target_column
    )
    
    # Analyze model performance
    summary_stats = analyze_model_performance(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_val=y_val,
        y_val_pred=y_val_pred,
        train_filenames=train_data['filename'],
        val_filenames=val_data['filename'],
        train_timestamps=timestamps[:40],
        val_timestamps=timestamps[40:],
        output_filename=output_filename
    )
    
    print(f"\nOptimal number of components: {optimal_n_components}")
    print_performance_summary(summary_stats)
    