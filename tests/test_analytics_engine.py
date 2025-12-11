"""
Tests for the Analytics Engine Module

This module contains property-based tests and unit tests for correlation
calculations and anomaly detection functionality.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from scipy import stats
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analytics_engine import CorrelationEngine, AnomalyDetector, analyze_weather_upi_correlations


class TestCorrelationEngine:
    """Test cases for the CorrelationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = CorrelationEngine()
    
    @given(
        data_size=st.integers(min_value=3, max_value=20),
        temp_base=st.floats(min_value=0, max_value=30, allow_nan=False),
        temp_variation=st.floats(min_value=1, max_value=20, allow_nan=False),
        txn_base=st.integers(min_value=1000, max_value=50000),
        txn_variation=st.integers(min_value=100, max_value=10000)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_correlation_calculation_accuracy(self, data_size, temp_base, temp_variation, txn_base, txn_variation):
        """
        # Feature: weather-upi-dashboard, Property 10: Correlation Calculation Accuracy
        Test that correlation calculations produce mathematically correct results.
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        # Generate varied data to avoid constant arrays
        np.random.seed(42)  # For reproducibility
        temp_data = np.random.normal(temp_base, temp_variation, data_size)
        txn_data = np.random.normal(txn_base, txn_variation, data_size).astype(int)
        
        # Ensure no negative transaction values
        txn_data = np.abs(txn_data)
        
        # Create test DataFrame
        df = pd.DataFrame({
            'avg_temp_c': temp_data,
            'total_upi_txn_count': txn_data,
            'humidity_pct': np.random.uniform(30, 80, data_size),  # Varied data
            'rain_mm': np.random.exponential(2, data_size),
            'avg_txn_value_inr': np.random.uniform(50, 200, data_size)
        })
        
        # Compute correlations using our engine
        correlations = self.engine.compute_correlations(df)
        
        # Verify correlation exists for temp vs txn_count
        corr_key = 'avg_temp_c_vs_total_upi_txn_count'
        assert corr_key in correlations
        
        # Compute expected correlation using scipy directly
        try:
            expected_corr, _ = stats.pearsonr(temp_data, txn_data)
            
            # Verify our calculation matches scipy's calculation
            if not np.isnan(expected_corr):
                assert abs(correlations[corr_key] - expected_corr) < 1e-10
            else:
                assert np.isnan(correlations[corr_key])
        except:
            # Handle cases where correlation cannot be computed
            assert np.isnan(correlations[corr_key])
    
    def test_correlation_empty_dataframe(self):
        """Test correlation calculation with empty DataFrame."""
        empty_df = pd.DataFrame()
        correlations = self.engine.compute_correlations(empty_df)
        assert correlations == {}
    
    def test_correlation_missing_columns(self):
        """Test correlation calculation with missing required columns."""
        df = pd.DataFrame({'irrelevant_col': [1, 2, 3]})
        correlations = self.engine.compute_correlations(df)
        assert correlations == {}


class TestCorrelationMatrix:
    """Test cases for correlation matrix functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = CorrelationEngine()
    
    @given(
        data_size=st.integers(min_value=2, max_value=50),
        temp_range=st.tuples(st.floats(min_value=-30, max_value=0), st.floats(min_value=0, max_value=50)),
        humidity_range=st.tuples(st.floats(min_value=0, max_value=50), st.floats(min_value=50, max_value=100)),
        rain_range=st.tuples(st.floats(min_value=0, max_value=10), st.floats(min_value=10, max_value=100))
    )
    @settings(max_examples=100)
    def test_correlation_matrix_completeness(self, data_size, temp_range, humidity_range, rain_range):
        """
        # Feature: weather-upi-dashboard, Property 11: Correlation Matrix Completeness
        Test that correlation matrix contains all weather-payment relationships.
        **Validates: Requirements 3.4**
        """
        # Generate test data
        temp_data = np.random.uniform(temp_range[0], temp_range[1], data_size)
        humidity_data = np.random.uniform(humidity_range[0], humidity_range[1], data_size)
        rain_data = np.random.uniform(rain_range[0], rain_range[1], data_size)
        txn_count_data = np.random.randint(1000, 100000, data_size)
        txn_value_data = np.random.uniform(50, 500, data_size)
        
        df = pd.DataFrame({
            'avg_temp_c': temp_data,
            'humidity_pct': humidity_data,
            'rain_mm': rain_data,
            'total_upi_txn_count': txn_count_data,
            'avg_txn_value_inr': txn_value_data
        })
        
        # Compute correlation matrix
        corr_matrix = self.engine.compute_correlation_matrix(df)
        
        # Verify matrix dimensions
        expected_weather_vars = 3  # temp, humidity, rain
        expected_txn_vars = 2     # count, value
        
        assert corr_matrix.shape == (expected_weather_vars, expected_txn_vars)
        
        # Verify all expected relationships are present
        expected_weather_cols = ['avg_temp_c', 'humidity_pct', 'rain_mm']
        expected_txn_cols = ['total_upi_txn_count', 'avg_txn_value_inr']
        
        assert list(corr_matrix.index) == expected_weather_cols
        assert list(corr_matrix.columns) == expected_txn_cols
        
        # Verify all values are valid correlations (between -1 and 1, allowing for floating point precision)
        for weather_var in expected_weather_cols:
            for txn_var in expected_txn_cols:
                corr_val = corr_matrix.loc[weather_var, txn_var]
                if not np.isnan(corr_val):
                    assert -1.0000001 <= corr_val <= 1.0000001, f"Correlation {corr_val} outside valid range"
    
    def test_correlation_matrix_empty_dataframe(self):
        """Test correlation matrix with empty DataFrame."""
        empty_df = pd.DataFrame()
        corr_matrix = self.engine.compute_correlation_matrix(empty_df)
        assert corr_matrix.empty


class TestCorrelationRangeValidation:
    """Test cases for correlation value range validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = CorrelationEngine()
    
    @given(
        data_size=st.integers(min_value=2, max_value=100),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=100)
    def test_correlation_value_range_validation(self, data_size, seed):
        """
        # Feature: weather-upi-dashboard, Property 18: Correlation Value Range Validation
        Test that all correlation coefficients fall within valid range [-1, 1].
        **Validates: Requirements 6.3**
        """
        np.random.seed(seed)
        
        # Generate diverse test data
        df = pd.DataFrame({
            'avg_temp_c': np.random.normal(25, 10, data_size),
            'humidity_pct': np.random.uniform(0, 100, data_size),
            'rain_mm': np.random.exponential(5, data_size),
            'total_upi_txn_count': np.random.poisson(50000, data_size),
            'avg_txn_value_inr': np.random.lognormal(5, 1, data_size)
        })
        
        # Compute all correlations
        correlations = self.engine.compute_correlations(df)
        
        # Verify all correlation values are within valid range (allowing for floating point precision)
        for corr_name, corr_value in correlations.items():
            if not np.isnan(corr_value):
                assert -1.0000001 <= corr_value <= 1.0000001, f"Correlation {corr_name} = {corr_value} is outside valid range [-1, 1]"
        
        # Also test correlation matrix
        corr_matrix = self.engine.compute_correlation_matrix(df)
        if not corr_matrix.empty:
            for weather_var in corr_matrix.index:
                for txn_var in corr_matrix.columns:
                    corr_val = corr_matrix.loc[weather_var, txn_var]
                    if not np.isnan(corr_val):
                        assert -1.0000001 <= corr_val <= 1.0000001, f"Matrix correlation {weather_var} vs {txn_var} = {corr_val} is outside valid range"
    
    def test_perfect_correlation_cases(self):
        """Test edge cases with perfect correlations."""
        # Perfect positive correlation
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # y = 2*x
        
        df = pd.DataFrame({
            'avg_temp_c': x,
            'total_upi_txn_count': y,
            'humidity_pct': [50] * 5,
            'rain_mm': [0] * 5,
            'avg_txn_value_inr': [100] * 5
        })
        
        correlations = self.engine.compute_correlations(df)
        temp_txn_corr = correlations['avg_temp_c_vs_total_upi_txn_count']
        
        # Should be very close to 1.0
        assert abs(temp_txn_corr - 1.0) < 1e-10
        
        # Perfect negative correlation
        df['total_upi_txn_count'] = [10, 8, 6, 4, 2]  # decreasing as temp increases
        correlations = self.engine.compute_correlations(df)
        temp_txn_corr = correlations['avg_temp_c_vs_total_upi_txn_count']
        
        # Should be very close to -1.0
        assert abs(temp_txn_corr - (-1.0)) < 1e-10


class TestAnomalyDetector:
    """Test cases for the AnomalyDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector(z_threshold=2.0)
    
    @given(
        data_size=st.integers(min_value=5, max_value=50),
        mean_val=st.floats(min_value=0, max_value=1000, allow_nan=False),
        std_val=st.floats(min_value=1, max_value=100, allow_nan=False),
        outlier_multiplier=st.floats(min_value=3, max_value=10, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_outlier_detection_accuracy(self, data_size, mean_val, std_val, outlier_multiplier):
        """
        # Feature: weather-upi-dashboard, Property 13: Outlier Detection Accuracy
        Test that outlier detection correctly identifies values exceeding 2 standard deviations.
        **Validates: Requirements 4.1, 4.2**
        """
        # Generate normal data with a large enough sample to maintain statistics
        np.random.seed(42)
        normal_data = np.random.normal(mean_val, std_val, max(data_size, 20))
        
        # Add extreme outliers that will definitely be detected
        # Use a much larger multiplier to ensure detection
        extreme_multiplier = max(outlier_multiplier, 5.0)  # Ensure at least 5 std devs
        outlier1 = mean_val + extreme_multiplier * std_val  # Positive outlier
        outlier2 = mean_val - extreme_multiplier * std_val  # Negative outlier
        
        # Create DataFrame with outliers
        all_data = np.concatenate([normal_data, [outlier1, outlier2]])
        df = pd.DataFrame({
            'total_upi_txn_count': all_data,
            'avg_txn_value_inr': np.random.uniform(50, 200, len(all_data)),
            'avg_temp_c': np.random.uniform(20, 35, len(all_data)),
            'humidity_pct': np.random.uniform(30, 80, len(all_data)),
            'rain_mm': np.random.exponential(2, len(all_data))
        })
        
        # Detect outliers
        result_df = self.detector.detect_outliers(df)
        
        # Verify outlier detection
        assert 'txn_volume_outlier' in result_df.columns
        assert 'total_upi_txn_count_z_score' in result_df.columns
        
        # Check that the known outliers are detected
        z_scores = result_df['total_upi_txn_count_z_score']
        outlier_flags = result_df['txn_volume_outlier']
        
        # The last two entries should be outliers (our inserted outliers)
        # Since we used extreme multipliers, they should definitely be detected
        assert outlier_flags.iloc[-1] == True, f"outlier2 not detected: z-score = {z_scores.iloc[-1]}"
        assert outlier_flags.iloc[-2] == True, f"outlier1 not detected: z-score = {z_scores.iloc[-2]}"
        
        # Verify z-scores are computed correctly
        assert abs(z_scores.iloc[-1]) > 2.0  # outlier2 z-score
        assert abs(z_scores.iloc[-2]) > 2.0  # outlier1 z-score
    
    def test_outlier_detection_empty_dataframe(self):
        """Test outlier detection with empty DataFrame."""
        empty_df = pd.DataFrame()
        result_df = self.detector.detect_outliers(empty_df)
        assert result_df.empty


class TestZScoreCalculation:
    """Test cases for z-score calculation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector()
    
    @given(
        data_size=st.integers(min_value=3, max_value=100),
        mean_val=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        std_val=st.floats(min_value=0.1, max_value=100, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_z_score_calculation_correctness(self, data_size, mean_val, std_val):
        """
        # Feature: weather-upi-dashboard, Property 15: Z-score Calculation Correctness
        Test that z-scores are computed mathematically correctly.
        **Validates: Requirements 4.4**
        """
        # Generate test data
        np.random.seed(42)
        data = np.random.normal(mean_val, std_val, data_size)
        data_series = pd.Series(data)
        
        # Compute z-scores using our method
        z_scores = self.detector.compute_z_scores(data_series)
        
        # Compute expected z-scores manually using pandas methods (same as our implementation)
        expected_mean = data_series.mean()
        expected_std = data_series.std()
        expected_z_scores = (data_series - expected_mean) / expected_std
        
        # Verify our calculation matches expected calculation
        for i in range(len(z_scores)):
            if not np.isnan(z_scores.iloc[i]) and not np.isnan(expected_z_scores[i]):
                assert abs(z_scores.iloc[i] - expected_z_scores[i]) < 1e-10
    
    def test_z_score_constant_data(self):
        """Test z-score calculation with constant data (std = 0)."""
        constant_data = pd.Series([5.0, 5.0, 5.0, 5.0])
        z_scores = self.detector.compute_z_scores(constant_data)
        
        # All z-scores should be 0 when std = 0
        assert all(z_scores == 0.0)
    
    def test_z_score_empty_series(self):
        """Test z-score calculation with empty series."""
        empty_series = pd.Series(dtype=float)
        z_scores = self.detector.compute_z_scores(empty_series)
        assert z_scores.empty
    
    def test_z_score_with_nan_values(self):
        """Test z-score calculation with NaN values."""
        data_with_nan = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        z_scores = self.detector.compute_z_scores(data_with_nan)
        
        # Should handle NaN values gracefully
        assert len(z_scores) == len(data_with_nan)
        assert pd.isna(z_scores.iloc[2])  # NaN should remain NaN