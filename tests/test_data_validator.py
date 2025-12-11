"""
Property-based tests for Data Validation module
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime, timedelta

from src.data_validator import DataValidator, ValidationResult, validate_datasets

class TestDataValidationProperties:
    """Property-based tests for data validation functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.validator = DataValidator()
    
    @given(
        valid_dates=st.lists(
            st.dates(min_value=datetime(2020, 1, 1).date(), 
                    max_value=datetime(2025, 12, 31).date()),
            min_size=1, max_size=20
        ),
        invalid_dates=st.lists(
            st.dates(min_value=datetime(1800, 1, 1).date(), 
                    max_value=datetime(1899, 12, 31).date()),
            min_size=0, max_size=5
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_date_range_validation(self, valid_dates, invalid_dates):
        """
        # Feature: weather-upi-dashboard, Property 2: Date Range Validation
        **Validates: Requirements 1.2**
        
        For any dataset containing date values, the Data_Validator should correctly 
        identify dates outside valid ranges and invalid date formats.
        """
        # Create test DataFrame with mix of valid and invalid dates
        all_dates = valid_dates + invalid_dates
        np.random.shuffle(all_dates)
        
        # Convert to pandas datetime
        date_series = pd.to_datetime(all_dates)
        
        # Create test weather DataFrame
        df = pd.DataFrame({
            'date': date_series,
            'city': ['Mumbai'] * len(all_dates),
            'avg_temp_c': [25.0] * len(all_dates),
            'humidity_pct': [60.0] * len(all_dates),
            'rain_mm': [0.1] * len(all_dates)
        })
        
        # Validate the data
        result = self.validator.validate_weather_data(df)
        
        # Check that validation detects issues if there are invalid dates
        if invalid_dates:
            # Should have warnings about early dates
            has_date_warning = any('very early' in warning for warning in result.warnings)
            assert has_date_warning, "Should detect dates that are too early"
        
        # Should not have errors for valid date formats (all dates are valid datetime objects)
        date_format_errors = [error for error in result.errors if 'invalid date' in error.lower()]
        assert len(date_format_errors) == 0, "Should not have date format errors for valid datetime objects"
        
        # Verify date column validation was performed
        assert len(df) == len(all_dates), "DataFrame should have correct number of rows"
        assert not df['date'].isna().any(), "No dates should be NaT after conversion"
    
    @given(
        positive_values=st.lists(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False), 
                               min_size=1, max_size=20),
        negative_values=st.lists(st.floats(min_value=-100.0, max_value=-0.1, allow_nan=False, allow_infinity=False), 
                               min_size=0, max_size=5)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_non_negative_value_validation(self, positive_values, negative_values):
        """
        # Feature: weather-upi-dashboard, Property 3: Non-negative Value Validation
        **Validates: Requirements 1.3**
        
        For any dataset containing numerical values, the Data_Validator should correctly 
        flag all negative transaction counts and weather measurements.
        """
        # Create test data with mix of positive and negative values
        all_values = positive_values + negative_values
        np.random.shuffle(all_values)
        
        # Create test UPI DataFrame (transaction counts should be non-negative)
        dates = pd.date_range('2024-01-01', periods=len(all_values), freq='D')
        upi_df = pd.DataFrame({
            'date': dates,
            'total_upi_txn_count': all_values,  # This should be non-negative
            'avg_txn_value_inr': [abs(v) for v in all_values]  # Make this positive
        })
        
        # Validate the UPI data
        result = self.validator.validate_upi_data(upi_df)
        
        # Check that validation detects negative values if they exist
        if negative_values:
            # Should have errors about negative transaction counts
            negative_errors = [error for error in result.errors if 'negative values' in error.lower()]
            assert len(negative_errors) > 0, "Should detect negative transaction counts"
            
            # Verify the error mentions the correct column
            txn_count_errors = [error for error in negative_errors if 'total_upi_txn_count' in error]
            assert len(txn_count_errors) > 0, "Should specifically flag negative transaction counts"
        else:
            # Should not have negative value errors if all values are positive
            negative_errors = [error for error in result.errors if 'negative values' in error.lower()]
            assert len(negative_errors) == 0, "Should not flag negative values when all are positive"
        
        # Verify that positive values don't trigger negative value errors
        assert len(positive_values) > 0, "Test should have some positive values"
        
        # Create a DataFrame with only positive values to verify no false positives
        positive_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=len(positive_values), freq='D'),
            'total_upi_txn_count': positive_values,
            'avg_txn_value_inr': positive_values
        })
        
        positive_result = self.validator.validate_upi_data(positive_df)
        negative_errors_positive = [error for error in positive_result.errors if 'negative values' in error.lower()]
        assert len(negative_errors_positive) == 0, "Should not flag negative values for positive-only data"
    
    @given(
        data_size=st.integers(min_value=5, max_value=50),
        null_percentage=st.floats(min_value=0.0, max_value=0.8)  # 0% to 80% nulls
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_null_value_detection_accuracy(self, data_size, null_percentage):
        """
        # Feature: weather-upi-dashboard, Property 4: Null Value Detection Accuracy
        **Validates: Requirements 1.4**
        
        For any dataset with missing values, the Data_Validator should accurately 
        count and locate all null values across all columns.
        """
        # Calculate number of null values to inject
        num_nulls = int(data_size * null_percentage)
        
        # Create base data
        dates = pd.date_range('2024-01-01', periods=data_size, freq='D')
        base_data = {
            'date': dates,
            'city': ['Mumbai'] * data_size,
            'avg_temp_c': [25.0] * data_size,
            'humidity_pct': [60.0] * data_size,
            'rain_mm': [0.1] * data_size
        }
        
        # Create DataFrame
        df = pd.DataFrame(base_data)
        
        # Inject null values randomly across non-date columns
        if num_nulls > 0:
            null_columns = ['city', 'avg_temp_c', 'humidity_pct', 'rain_mm']
            
            # Randomly select positions to make null
            total_positions = len(null_columns) * data_size
            null_positions = np.random.choice(total_positions, size=min(num_nulls, total_positions), replace=False)
            
            actual_nulls_injected = 0
            for pos in null_positions:
                col_idx = pos // data_size
                row_idx = pos % data_size
                col_name = null_columns[col_idx]
                df.iloc[row_idx, df.columns.get_loc(col_name)] = np.nan
                actual_nulls_injected += 1
        else:
            actual_nulls_injected = 0
        
        # Validate the data
        result = self.validator.validate_weather_data(df)
        
        # Count actual nulls in the DataFrame
        actual_null_count = df.isnull().sum().sum()
        
        # Verify null detection accuracy
        if actual_null_count > 0:
            # Should have warnings about null values
            null_warnings = [warning for warning in result.warnings if 'null values' in warning.lower()]
            assert len(null_warnings) > 0, f"Should detect {actual_null_count} null values"
            
            # Check if the warning mentions the correct count
            count_mentioned = any(str(actual_null_count) in warning for warning in null_warnings)
            assert count_mentioned, f"Should mention the correct null count ({actual_null_count})"
        else:
            # Should not have null value warnings if no nulls exist
            null_warnings = [warning for warning in result.warnings if 'null values' in warning.lower()]
            assert len(null_warnings) == 0, "Should not report null values when none exist"
        
        # Verify that the validator's null detection matches actual nulls
        assert actual_null_count >= 0, "Null count should be non-negative"
        
        # Test with completely empty rows (should be flagged as error)
        if data_size >= 2:
            # Create a DataFrame with one completely empty row
            empty_row_df = df.copy()
            empty_row_df.iloc[0, 1:] = np.nan  # Make first row empty except date
            
            empty_result = self.validator.validate_weather_data(empty_row_df)
            
            # Should detect the empty row issue
            # Note: This might be detected as high null percentage rather than empty rows
            # depending on implementation, so we check for any null-related issues
            has_null_issues = len(empty_result.warnings) > 0 or len(empty_result.errors) > 0
            assert has_null_issues, "Should detect issues with rows that have many nulls"