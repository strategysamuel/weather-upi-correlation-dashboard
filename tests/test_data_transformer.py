"""
Property-based tests for Data Transformation module
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime, timedelta
from typing import List, Dict

from src.data_transformer import DataTransformer, standardize_weather_columns, standardize_upi_columns, merge_weather_upi_data

class TestDataTransformationProperties:
    """Property-based tests for data transformation functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.transformer = DataTransformer()
    
    @given(
        column_variations=st.lists(
            st.sampled_from([
                # Weather column variations
                ('date', 'Date', 'DATE', 'date_col'),
                ('avg_temp_c', 'temperature', 'temp_c', 'Temperature_C', 'avg-temp-c'),
                ('humidity_pct', 'humidity', 'Humidity_PCT', 'relative_humidity', 'humidity-pct'),
                ('rain_mm', 'rainfall', 'precipitation', 'Precipitation_MM', 'rain-mm'),
                ('city', 'City', 'CITY', 'location'),
                ('condition', 'weather_condition', 'Weather-Condition', 'CONDITION')
            ]),
            min_size=5, max_size=6, unique=True
        ),
        data_size=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_column_standardization_consistency(self, column_variations, data_size):
        """
        # Feature: weather-upi-dashboard, Property 6: Column Standardization Consistency
        **Validates: Requirements 2.1**
        
        For any dataset with varying column naming conventions, the Analytics_Pipeline 
        should standardize all column names to a consistent format.
        """
        # Create test DataFrame with varied column names
        test_data = {}
        expected_standard_columns = set()
        
        for i, (standard_name, *variations) in enumerate(column_variations):
            # Pick a random variation for this column
            if variations:
                chosen_variation = np.random.choice([standard_name] + variations)
            else:
                chosen_variation = standard_name
            
            # Generate test data based on column type
            if 'date' in standard_name.lower():
                test_data[chosen_variation] = pd.date_range('2024-01-01', periods=data_size, freq='D')
                expected_standard_columns.add('date')
            elif 'temp' in standard_name.lower():
                test_data[chosen_variation] = np.random.uniform(15, 35, data_size)
                expected_standard_columns.add('avg_temp_c')
            elif 'humidity' in standard_name.lower():
                test_data[chosen_variation] = np.random.uniform(30, 90, data_size)
                expected_standard_columns.add('humidity_pct')
            elif 'rain' in standard_name.lower() or 'precipitation' in standard_name.lower():
                test_data[chosen_variation] = np.random.uniform(0, 50, data_size)
                expected_standard_columns.add('rain_mm')
            elif 'city' in standard_name.lower():
                test_data[chosen_variation] = ['Mumbai'] * data_size
                expected_standard_columns.add('city')
            elif 'condition' in standard_name.lower():
                test_data[chosen_variation] = ['Sunny'] * data_size
                expected_standard_columns.add('condition')
        
        # Ensure we have the minimum required columns for weather data
        required_weather_columns = {'date', 'avg_temp_c', 'humidity_pct', 'rain_mm', 'city'}
        if not required_weather_columns.issubset(expected_standard_columns):
            # Add missing required columns with standard names
            for missing_col in required_weather_columns - expected_standard_columns:
                if missing_col == 'date':
                    test_data['date'] = pd.date_range('2024-01-01', periods=data_size, freq='D')
                elif missing_col == 'avg_temp_c':
                    test_data['avg_temp_c'] = np.random.uniform(15, 35, data_size)
                elif missing_col == 'humidity_pct':
                    test_data['humidity_pct'] = np.random.uniform(30, 90, data_size)
                elif missing_col == 'rain_mm':
                    test_data['rain_mm'] = np.random.uniform(0, 50, data_size)
                elif missing_col == 'city':
                    test_data['city'] = ['Mumbai'] * data_size
                expected_standard_columns.add(missing_col)
        
        # Create DataFrame
        df = pd.DataFrame(test_data)
        
        # Apply column standardization
        standardized_df = self.transformer.standardize_columns(df, 'weather')
        
        # Verify standardization consistency
        # 1. All required standard columns should be present
        required_columns = ['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm']
        for col in required_columns:
            assert col in standardized_df.columns, f"Required standard column '{col}' missing after standardization"
        
        # 2. Column names should follow consistent naming convention (lowercase with underscores)
        for col in standardized_df.columns:
            # Should be lowercase
            assert col.islower() or col == col.lower(), f"Column '{col}' should be lowercase"
            
            # Should not contain spaces or special characters (except underscores)
            import re
            assert re.match(r'^[a-z0-9_]+$', col), f"Column '{col}' contains invalid characters"
        
        # 3. Data integrity should be preserved
        assert len(standardized_df) == data_size, "Row count should be preserved after standardization"
        
        # 4. Standard columns should have expected data types
        assert pd.api.types.is_datetime64_any_dtype(standardized_df['date']), "Date column should be datetime"
        assert pd.api.types.is_numeric_dtype(standardized_df['avg_temp_c']), "Temperature should be numeric"
        assert pd.api.types.is_numeric_dtype(standardized_df['humidity_pct']), "Humidity should be numeric"
        assert pd.api.types.is_numeric_dtype(standardized_df['rain_mm']), "Rain should be numeric"
        
        # 5. Test UPI column standardization as well
        upi_test_data = {
            'Date': pd.date_range('2024-01-01', periods=data_size, freq='D'),
            'Total-UPI-Txn-Count': np.random.randint(1000, 10000, data_size),
            'Avg_Txn_Value_INR': np.random.uniform(100, 1000, data_size)
        }
        
        upi_df = pd.DataFrame(upi_test_data)
        upi_standardized = self.transformer.standardize_columns(upi_df, 'upi')
        
        # Verify UPI standardization
        upi_required_columns = ['date', 'total_upi_txn_count', 'avg_txn_value_inr']
        for col in upi_required_columns:
            assert col in upi_standardized.columns, f"Required UPI column '{col}' missing after standardization"
        
        # Verify UPI column naming consistency
        for col in upi_standardized.columns:
            assert col.islower() or col == col.lower(), f"UPI column '{col}' should be lowercase"
            assert re.match(r'^[a-z0-9_]+$', col), f"UPI column '{col}' contains invalid characters"
    
    @given(
        weather_dates=st.lists(
            st.dates(min_value=datetime(2024, 1, 1).date(), 
                    max_value=datetime(2024, 12, 31).date()),
            min_size=5, max_size=15, unique=True
        ),
        upi_dates=st.lists(
            st.dates(min_value=datetime(2024, 1, 1).date(), 
                    max_value=datetime(2024, 12, 31).date()),
            min_size=5, max_size=15, unique=True
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_date_based_merge_completeness(self, weather_dates, upi_dates):
        """
        # Feature: weather-upi-dashboard, Property 7: Date-based Merge Completeness
        **Validates: Requirements 2.2, 2.4**
        
        For any two datasets with overlapping date ranges, the merge operation should 
        preserve all records with matching dates without data loss.
        """
        # Sort dates to ensure proper ordering
        weather_dates = sorted(weather_dates)
        upi_dates = sorted(upi_dates)
        
        # Create weather DataFrame
        weather_df = pd.DataFrame({
            'date': weather_dates,
            'city': ['Mumbai'] * len(weather_dates),
            'avg_temp_c': np.random.uniform(20, 35, len(weather_dates)),
            'humidity_pct': np.random.uniform(40, 80, len(weather_dates)),
            'rain_mm': np.random.uniform(0, 20, len(weather_dates)),
            'condition': ['Sunny'] * len(weather_dates)
        })
        
        # Create UPI DataFrame
        upi_df = pd.DataFrame({
            'date': upi_dates,
            'total_upi_txn_count': np.random.randint(1000, 10000, len(upi_dates)),
            'avg_txn_value_inr': np.random.uniform(100, 1000, len(upi_dates))
        })
        
        # Calculate expected overlap
        weather_date_set = set(weather_dates)
        upi_date_set = set(upi_dates)
        expected_overlap = weather_date_set.intersection(upi_date_set)
        expected_merge_count = len(expected_overlap)
        
        # Perform merge
        merged_df = self.transformer.merge_datasets(weather_df, upi_df)
        
        # Verify merge completeness
        if expected_merge_count > 0:
            # Should have records for all overlapping dates
            assert len(merged_df) == expected_merge_count, \
                f"Expected {expected_merge_count} merged records, got {len(merged_df)}"
            
            # All dates in merged DataFrame should be in the overlap
            merged_dates = set(merged_df['date'].dt.date)
            assert merged_dates == expected_overlap, \
                "Merged dates should exactly match the expected overlap"
            
            # Verify no data loss - all columns from both datasets should be present
            weather_columns = set(weather_df.columns)
            upi_columns = set(upi_df.columns)
            merged_columns = set(merged_df.columns)
            
            # All original columns should be preserved (possibly with suffixes)
            for col in weather_columns:
                if col == 'date':
                    assert col in merged_columns, f"Date column should be preserved"
                else:
                    # Column might have suffix if there was a conflict
                    col_preserved = (col in merged_columns or 
                                   f"{col}_weather" in merged_columns or
                                   f"{col}_upi" in merged_columns)
                    assert col_preserved, f"Weather column '{col}' not preserved in merge"
            
            for col in upi_columns:
                if col == 'date':
                    continue  # Already checked
                else:
                    col_preserved = (col in merged_columns or 
                                   f"{col}_weather" in merged_columns or
                                   f"{col}_upi" in merged_columns)
                    assert col_preserved, f"UPI column '{col}' not preserved in merge"
            
            # Verify data integrity - values should match original data
            for _, row in merged_df.iterrows():
                date_val = row['date'].date()
                
                # Find corresponding rows in original data
                weather_row = weather_df[weather_df['date'] == date_val].iloc[0]
                upi_row = upi_df[upi_df['date'] == date_val].iloc[0]
                
                # Check that values are preserved (accounting for possible suffixes)
                for col in weather_df.columns:
                    if col == 'date':
                        continue
                    
                    if col in merged_df.columns:
                        merged_val = row[col]
                    elif f"{col}_weather" in merged_df.columns:
                        merged_val = row[f"{col}_weather"]
                    else:
                        pytest.fail(f"Weather column '{col}' not found in merged data")
                    
                    original_val = weather_row[col]
                    
                    # Handle different data types appropriately
                    if pd.api.types.is_numeric_dtype(type(original_val)):
                        assert abs(merged_val - original_val) < 1e-10, \
                            f"Numeric value mismatch for {col}: {merged_val} != {original_val}"
                    else:
                        assert merged_val == original_val, \
                            f"Value mismatch for {col}: {merged_val} != {original_val}"
        else:
            # No overlap expected, merged DataFrame should be empty
            assert len(merged_df) == 0, "Merged DataFrame should be empty when no dates overlap"
    
    @given(
        date_format=st.sampled_from([
            '%Y-%m-%d',      # 2024-01-15 (unambiguous)
            '%Y/%m/%d'       # 2024/01/15 (unambiguous)
        ]),
        base_dates=st.lists(
            st.dates(min_value=datetime(2024, 1, 1).date(),
                    max_value=datetime(2024, 6, 30).date()),
            min_size=3, max_size=10, unique=True
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_8_date_format_standardization(self, date_format, base_dates):
        """
        # Feature: weather-upi-dashboard, Property 8: Date Format Standardization
        **Validates: Requirements 2.3**
        
        For any datasets with different date formats, the Analytics_Pipeline should 
        convert all dates to identical formats before merging.
        """
        # Create test data with a single date format
        test_data = []
        
        for i, date_obj in enumerate(base_dates):
            # Format the date according to the chosen format
            formatted_date = date_obj.strftime(date_format)
            
            test_data.append({
                'date': formatted_date,
                'city': 'Mumbai',
                'avg_temp_c': 25.0 + i,
                'humidity_pct': 60.0 + i,
                'rain_mm': 0.1 * i,
                'condition': 'Sunny'
            })
        
        # Create DataFrame
        df = pd.DataFrame(test_data)
        
        # Apply date normalization
        normalized_df = self.transformer.normalize_dates(df, 'date')
        
        # Verify date format standardization
        # 1. All dates should be successfully parsed (no NaT values)
        assert not normalized_df['date'].isna().any(), \
            "All dates should be successfully parsed after normalization"
        
        # 2. All dates should have the same format (pandas datetime)
        assert pd.api.types.is_datetime64_any_dtype(normalized_df['date']), \
            "Date column should be datetime type after normalization"
        
        # 3. Date values should be preserved for unambiguous formats
        original_date_set = set(base_dates)
        normalized_date_set = set(normalized_df['date'].dt.date)
        
        assert original_date_set == normalized_date_set, \
            f"Original date values should be preserved. Original: {original_date_set}, Normalized: {normalized_date_set}"
        
        # 4. Dates should be in consistent format (no time component)
        for date_val in normalized_df['date']:
            # Should be midnight (no time component)
            assert date_val.time() == datetime.min.time(), \
                "Normalized dates should not have time components"
        
        # 5. Test that different formats can be merged after standardization
        # Create second DataFrame with different format
        different_format = '%Y/%m/%d' if date_format == '%Y-%m-%d' else '%Y-%m-%d'
        
        second_test_data = []
        for i, date_obj in enumerate(base_dates[:3]):  # Use subset for second dataset
            formatted_date = date_obj.strftime(different_format)
            second_test_data.append({
                'date': formatted_date,
                'total_upi_txn_count': 1000 + i,
                'avg_txn_value_inr': 500.0 + i
            })
        
        second_df = pd.DataFrame(second_test_data)
        second_normalized = self.transformer.normalize_dates(second_df, 'date')
        
        # Both DataFrames should have identical date format after normalization
        assert normalized_df['date'].dtype == second_normalized['date'].dtype, \
            "Date columns should have identical types after normalization"
        
        # Should be able to merge without date format conflicts
        if len(second_normalized) > 0 and len(normalized_df) > 0:
            try:
                # This should not raise an error due to date format mismatches
                merged = pd.merge(normalized_df, second_normalized, on='date', how='inner')
                
                # Verify merge worked correctly
                if len(merged) > 0:
                    assert pd.api.types.is_datetime64_any_dtype(merged['date']), \
                        "Merged date column should maintain datetime type"
            except Exception as e:
                pytest.fail(f"Date format standardization failed to enable merging: {e}")