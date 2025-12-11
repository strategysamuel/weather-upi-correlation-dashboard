"""
Analytics Engine Module

This module provides correlation analysis and anomaly detection functionality
for weather and UPI transaction data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List, Any
import warnings

class CorrelationEngine:
    """
    Handles correlation calculations between weather and transaction variables.
    """
    
    def __init__(self):
        """Initialize the correlation engine."""
        pass
    
    def compute_correlations(self, merged_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Pearson correlation coefficients between weather and transaction variables.
        
        This method calculates pairwise correlations between all weather variables
        (temperature, humidity, rainfall) and UPI transaction metrics (count, value).
        It uses Pearson correlation coefficient which measures linear relationships
        between variables on a scale from -1 to +1.
        
        The correlation calculation process:
        1. Identifies available weather and transaction variables
        2. For each weather-transaction pair:
           - Removes rows with missing values in either variable
           - Calculates Pearson correlation coefficient
           - Handles edge cases (insufficient data points)
        3. Returns dictionary with correlation results
        
        Args:
            merged_df (pd.DataFrame): DataFrame containing both weather and transaction data
                                     Must include columns for weather variables and UPI metrics
            
        Returns:
            Dict[str, float]: Dictionary mapping variable pair names to correlation coefficients
                             Keys format: "{weather_var}_vs_{transaction_var}"
                             Values range from -1.0 to +1.0, or NaN if insufficient data
                             
        Example:
            >>> engine = CorrelationEngine()
            >>> correlations = engine.compute_correlations(merged_df)
            >>> print(correlations['avg_temp_c_vs_total_upi_txn_count'])
            0.234
            
        Requirements: 3.1, 3.2, 3.3 (correlation calculations between weather and payments)
        """
        if merged_df.empty:
            return {}
        
        # Define weather and transaction variables
        weather_vars = ['avg_temp_c', 'humidity_pct', 'rain_mm']
        transaction_vars = ['total_upi_txn_count', 'avg_txn_value_inr']
        
        correlations = {}
        
        # Compute correlations between all weather-transaction pairs
        for weather_var in weather_vars:
            for txn_var in transaction_vars:
                if weather_var in merged_df.columns and txn_var in merged_df.columns:
                    # Remove rows where either variable is NaN to ensure clean correlation calculation
                    # This prevents scipy.stats.pearsonr from failing on missing data
                    clean_data = merged_df[[weather_var, txn_var]].dropna()
                    
                    # Pearson correlation requires at least 2 data points
                    # With fewer points, correlation is mathematically undefined
                    if len(clean_data) >= 2:
                        # Calculate Pearson correlation coefficient and p-value
                        # We only store the coefficient here; p-value handled separately
                        corr_coef, _ = stats.pearsonr(clean_data[weather_var], clean_data[txn_var])
                        correlations[f"{weather_var}_vs_{txn_var}"] = corr_coef
                    else:
                        # Mark as NaN when insufficient data for meaningful correlation
                        correlations[f"{weather_var}_vs_{txn_var}"] = np.nan
        
        return correlations
    
    def compute_correlation_matrix(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a correlation matrix with all weather-payment relationships.
        
        This method creates a comprehensive correlation matrix showing relationships
        between weather variables and UPI transaction metrics. The matrix provides
        a structured view of all pairwise correlations for easy analysis and
        visualization in heatmaps.
        
        Matrix structure:
        - Rows: Weather variables (temperature, humidity, rainfall)
        - Columns: Transaction variables (count, value)
        - Values: Pearson correlation coefficients (-1 to +1)
        
        Args:
            merged_df (pd.DataFrame): DataFrame containing both weather and transaction data
                                     Must include both weather and UPI metric columns
            
        Returns:
            pd.DataFrame: Correlation matrix with weather variables as rows and
                         transaction variables as columns. Empty DataFrame if
                         insufficient data or missing required columns.
                         
        Example:
            >>> engine = CorrelationEngine()
            >>> matrix = engine.compute_correlation_matrix(merged_df)
            >>> print(matrix.loc['avg_temp_c', 'total_upi_txn_count'])
            0.234
            
        Requirements: 3.4 (correlation matrix generation with all relationships)
        """
        if merged_df.empty:
            return pd.DataFrame()
        
        # Define variables for correlation matrix
        weather_vars = ['avg_temp_c', 'humidity_pct', 'rain_mm']
        transaction_vars = ['total_upi_txn_count', 'avg_txn_value_inr']
        
        # Filter to only include available columns
        available_weather = [var for var in weather_vars if var in merged_df.columns]
        available_transaction = [var for var in transaction_vars if var in merged_df.columns]
        
        if not available_weather or not available_transaction:
            return pd.DataFrame()
        
        # Create correlation matrix
        all_vars = available_weather + available_transaction
        correlation_matrix = merged_df[all_vars].corr(method='pearson')
        
        # Extract only weather-transaction correlations
        weather_txn_matrix = correlation_matrix.loc[available_weather, available_transaction]
        
        return weather_txn_matrix
    
    def compute_statistical_significance(self, merged_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute p-values for statistical significance of correlations.
        
        Args:
            merged_df: DataFrame containing both weather and transaction data
            
        Returns:
            Dictionary mapping variable pairs to p-values
            
        Requirements: 3.5
        """
        if merged_df.empty:
            return {}
        
        # Define weather and transaction variables
        weather_vars = ['avg_temp_c', 'humidity_pct', 'rain_mm']
        transaction_vars = ['total_upi_txn_count', 'avg_txn_value_inr']
        
        p_values = {}
        
        # Compute p-values for all weather-transaction pairs
        for weather_var in weather_vars:
            for txn_var in transaction_vars:
                if weather_var in merged_df.columns and txn_var in merged_df.columns:
                    # Remove rows where either variable is NaN
                    clean_data = merged_df[[weather_var, txn_var]].dropna()
                    
                    if len(clean_data) >= 2:  # Need at least 2 points for correlation
                        _, p_value = stats.pearsonr(clean_data[weather_var], clean_data[txn_var])
                        p_values[f"{weather_var}_vs_{txn_var}"] = p_value
                    else:
                        p_values[f"{weather_var}_vs_{txn_var}"] = np.nan
        
        return p_values


class AnomalyDetector:
    """
    Handles anomaly detection and outlier identification.
    """
    
    def __init__(self, z_threshold: float = 2.0):
        """
        Initialize the anomaly detector.
        
        Args:
            z_threshold: Z-score threshold for outlier detection (default: 2.0)
        """
        self.z_threshold = z_threshold
    
    def compute_z_scores(self, data: pd.Series) -> pd.Series:
        """
        Compute z-scores for a data series to identify statistical outliers.
        
        Z-score measures how many standard deviations a data point is from the mean.
        The formula is: z = (x - μ) / σ, where:
        - x is the data point
        - μ is the mean of the dataset
        - σ is the standard deviation
        
        Z-scores interpretation:
        - |z| > 2: Likely outlier (beyond 2 standard deviations)
        - |z| > 3: Very likely outlier (beyond 3 standard deviations)
        - z > 0: Above average
        - z < 0: Below average
        
        Args:
            data (pd.Series): Pandas Series containing numerical data for z-score calculation
                             Missing values (NaN) are handled gracefully
            
        Returns:
            pd.Series: Series of z-scores with same index as input data
                      Returns empty Series if input is empty
                      Returns zeros if standard deviation is zero (constant data)
                      
        Example:
            >>> detector = AnomalyDetector()
            >>> data = pd.Series([10, 12, 11, 25, 13])  # 25 is outlier
            >>> z_scores = detector.compute_z_scores(data)
            >>> print(z_scores[3])  # z-score for value 25
            2.1
            
        Requirements: 4.4 (z-score calculation correctness)
        """
        if data.empty or data.isna().all():
            return pd.Series(dtype=float)
        
        # Remove NaN values for calculation
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return pd.Series(index=data.index, dtype=float)
        
        mean_val = clean_data.mean()
        std_val = clean_data.std()
        
        # Handle edge case where all values are identical (std deviation = 0)
        # In this case, no values are outliers since there's no variation
        if std_val == 0:
            return pd.Series(0.0, index=data.index)
        
        # Apply z-score formula: (x - μ) / σ
        # This standardizes values to show how many standard deviations from mean
        z_scores = (data - mean_val) / std_val
        return z_scores
    
    def detect_outliers(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using z-score analysis with configurable threshold.
        
        This method identifies statistical outliers in both weather and transaction data
        using z-score analysis. Data points with |z-score| > threshold are flagged as
        outliers. The method adds several columns to the DataFrame:
        
        Added columns:
        - '{variable}_z_score': Z-score for each numerical variable
        - 'txn_volume_outlier': Boolean flag for transaction outliers
        - 'weather_outlier': Boolean flag for weather outliers
        
        Outlier detection process:
        1. Calculate z-scores for all numerical variables
        2. Flag records where |z-score| > threshold for any variable
        3. Create separate flags for transaction and weather outliers
        4. Combine individual variable outlier flags into category flags
        
        Args:
            merged_df (pd.DataFrame): DataFrame containing weather and transaction data
                                     Must include numerical columns for analysis
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with original data plus:
                         - Z-score columns for each numerical variable
                         - Boolean outlier flags for transactions and weather
                         - Same number of rows as input DataFrame
                         
        Example:
            >>> detector = AnomalyDetector(z_threshold=2.0)
            >>> enhanced_df = detector.detect_outliers(merged_df)
            >>> outlier_count = enhanced_df['txn_volume_outlier'].sum()
            >>> print(f"Found {outlier_count} transaction outliers")
            
        Requirements: 4.1, 4.2, 4.3 (outlier detection and flagging)
        """
        if merged_df.empty:
            return merged_df.copy()
        
        result_df = merged_df.copy()
        
        # Define variables to check for outliers
        transaction_vars = ['total_upi_txn_count', 'avg_txn_value_inr']
        weather_vars = ['avg_temp_c', 'humidity_pct', 'rain_mm']
        
        # Detect transaction outliers
        txn_outlier_flags = []
        for var in transaction_vars:
            if var in result_df.columns:
                z_scores = self.compute_z_scores(result_df[var])
                outliers = np.abs(z_scores) > self.z_threshold
                txn_outlier_flags.append(outliers)
                result_df[f'{var}_z_score'] = z_scores
        
        # Combine transaction outlier flags
        if txn_outlier_flags:
            result_df['txn_volume_outlier'] = pd.concat(txn_outlier_flags, axis=1).any(axis=1)
        else:
            result_df['txn_volume_outlier'] = False
        
        # Detect weather outliers
        weather_outlier_flags = []
        for var in weather_vars:
            if var in result_df.columns:
                z_scores = self.compute_z_scores(result_df[var])
                outliers = np.abs(z_scores) > self.z_threshold
                weather_outlier_flags.append(outliers)
                result_df[f'{var}_z_score'] = z_scores
        
        # Combine weather outlier flags
        if weather_outlier_flags:
            result_df['weather_outlier'] = pd.concat(weather_outlier_flags, axis=1).any(axis=1)
        else:
            result_df['weather_outlier'] = False
        
        return result_df
    
    def generate_anomaly_summary(self, merged_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of detected outliers with their characteristics.
        
        Args:
            merged_df: DataFrame with outlier flags
            
        Returns:
            Dictionary containing anomaly summary statistics
            
        Requirements: 4.5
        """
        if merged_df.empty:
            return {}
        
        summary = {}
        
        # Count outliers
        if 'txn_volume_outlier' in merged_df.columns:
            summary['transaction_outliers_count'] = int(merged_df['txn_volume_outlier'].sum())
            summary['transaction_outliers_percentage'] = float(
                (merged_df['txn_volume_outlier'].sum() / len(merged_df)) * 100
            )
        
        if 'weather_outlier' in merged_df.columns:
            summary['weather_outliers_count'] = int(merged_df['weather_outlier'].sum())
            summary['weather_outliers_percentage'] = float(
                (merged_df['weather_outlier'].sum() / len(merged_df)) * 100
            )
        
        # Get outlier dates
        if 'txn_volume_outlier' in merged_df.columns and 'date' in merged_df.columns:
            txn_outlier_dates = merged_df[merged_df['txn_volume_outlier']]['date'].tolist()
            summary['transaction_outlier_dates'] = [str(date) for date in txn_outlier_dates]
        
        if 'weather_outlier' in merged_df.columns and 'date' in merged_df.columns:
            weather_outlier_dates = merged_df[merged_df['weather_outlier']]['date'].tolist()
            summary['weather_outlier_dates'] = [str(date) for date in weather_outlier_dates]
        
        # Calculate extreme values
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.endswith('_z_score'):
                var_name = col.replace('_z_score', '')
                max_z = merged_df[col].max()
                min_z = merged_df[col].min()
                if not pd.isna(max_z) and not pd.isna(min_z):
                    summary[f'{var_name}_max_z_score'] = float(max_z)
                    summary[f'{var_name}_min_z_score'] = float(min_z)
        
        return summary


def analyze_weather_upi_correlations(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main function to perform complete correlation and anomaly analysis.
    
    Args:
        merged_df: DataFrame containing merged weather and UPI data
        
    Returns:
        Dictionary containing all analysis results
    """
    if merged_df.empty:
        return {}
    
    # Initialize engines
    corr_engine = CorrelationEngine()
    anomaly_detector = AnomalyDetector()
    
    # Perform correlation analysis
    correlations = corr_engine.compute_correlations(merged_df)
    correlation_matrix = corr_engine.compute_correlation_matrix(merged_df)
    p_values = corr_engine.compute_statistical_significance(merged_df)
    
    # Perform anomaly detection
    enhanced_df = anomaly_detector.detect_outliers(merged_df)
    anomaly_summary = anomaly_detector.generate_anomaly_summary(enhanced_df)
    
    return {
        'correlations': correlations,
        'correlation_matrix': correlation_matrix,
        'p_values': p_values,
        'enhanced_dataframe': enhanced_df,
        'anomaly_summary': anomaly_summary
    }