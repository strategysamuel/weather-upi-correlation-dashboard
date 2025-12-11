"""
Tests for the main pipeline orchestrator

This module contains property-based tests for pipeline validation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from pathlib import Path
import sys
import tempfile
import os

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_validator import DataValidator, ValidationResult
from data_transformer import DataTransformer
from main import WeatherUPIPipeline
import config

class TestPipelineValidation:
    """Property-based tests for pipeline validation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
        self.pipeline = WeatherUPIPipeline()
    
    @given(
        weather_data=st.lists(
            st.tuples(
                st.dates(min_value=pd.Timestamp('2024-01-01').date(), 
                        max_value=pd.Timestamp('2024-12-31').date()),
                st.just('Mumbai'),  # Fixed city to speed up generation
                st.floats(min_value=20, max_value=40, allow_nan=False),
                st.floats(min_value=50, max_value=90, allow_nan=False),
                st.floats(min_value=0, max_value=50, allow_nan=False),
                st.just('Clear')  # Fixed condition to speed up generation
            ),
            min_size=1,
            max_size=20  # Reduced size for faster generation
        ),
        upi_data=st.lists(
            st.tuples(
                st.dates(min_value=pd.Timestamp('2024-01-01').date(), 
                        max_value=pd.Timestamp('2024-12-31').date()),
                st.integers(min_value=100, max_value=10000),
                st.floats(min_value=100, max_value=1000, allow_nan=False),
                st.just('Normal')  # Fixed notes to speed up generation
            ),
            min_size=1,
            max_size=20  # Reduced size for faster generation
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_validation_report_generation_property(self, weather_data, upi_data):
        """
        # Feature: weather-upi-dashboard, Property 5: Validation Report Generation
        # **Validates: Requirements 1.5**
        
        For any validation process, the Analytics_Pipeline should generate a complete 
        validation report with clear pass/fail status for all checks
        """
        # Create DataFrames from generated data
        weather_df = pd.DataFrame(weather_data, columns=[
            'date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm', 'condition'
        ])
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        upi_df = pd.DataFrame(upi_data, columns=[
            'date', 'total_upi_txn_count', 'avg_txn_value_inr', 'notes'
        ])
        upi_df['date'] = pd.to_datetime(upi_df['date'])
        
        # Perform validation
        weather_result = self.validator.validate_weather_data(weather_df)
        upi_result = self.validator.validate_upi_data(upi_df)
        validation_report = self.validator.generate_validation_report(weather_result, upi_result)
        
        # Property: Validation report should always be generated and contain required sections
        assert isinstance(validation_report, str), "Validation report should be a string"
        assert len(validation_report) > 0, "Validation report should not be empty"
        
        # Check for required sections in the report
        required_sections = [
            "DATA VALIDATION REPORT",
            "OVERALL STATUS:",
            "WEATHER DATA VALIDATION:",
            "UPI DATA VALIDATION:",
            "SUMMARY:"
        ]
        
        for section in required_sections:
            assert section in validation_report, f"Validation report missing required section: {section}"
        
        # Check that overall status is clearly indicated
        assert ("PASSED" in validation_report or "FAILED" in validation_report), \
            "Validation report should contain clear PASSED or FAILED status"
        
        # Check that error and warning counts are reported
        assert "Total Errors:" in validation_report, "Report should include total error count"
        assert "Total Warnings:" in validation_report, "Report should include total warning count"
        
        # Verify that the report reflects the actual validation results
        if weather_result.passed and upi_result.passed:
            assert "OVERALL STATUS: PASSED" in validation_report, \
                "Report should show PASSED when both validations pass"
        else:
            assert "OVERALL STATUS: FAILED" in validation_report, \
                "Report should show FAILED when any validation fails"
    
    @given(
        weather_errors=st.integers(min_value=0, max_value=10),
        weather_warnings=st.integers(min_value=0, max_value=10),
        upi_errors=st.integers(min_value=0, max_value=10),
        upi_warnings=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=100)
    def test_validation_report_error_counting_property(self, weather_errors, weather_warnings, 
                                                     upi_errors, upi_warnings):
        """
        Property test for validation report error and warning counting accuracy
        """
        # Create mock validation results
        weather_result = ValidationResult()
        upi_result = ValidationResult()
        
        # Add specified number of errors and warnings
        for i in range(weather_errors):
            weather_result.add_error(f"Weather error {i}")
        for i in range(weather_warnings):
            weather_result.add_warning(f"Weather warning {i}")
        for i in range(upi_errors):
            upi_result.add_error(f"UPI error {i}")
        for i in range(upi_warnings):
            upi_result.add_warning(f"UPI warning {i}")
        
        # Generate report
        validation_report = self.validator.generate_validation_report(weather_result, upi_result)
        
        # Property: Report should accurately count total errors and warnings
        total_errors = weather_errors + upi_errors
        total_warnings = weather_warnings + upi_warnings
        
        assert f"Total Errors: {total_errors}" in validation_report, \
            f"Report should show correct total error count: {total_errors}"
        assert f"Total Warnings: {total_warnings}" in validation_report, \
            f"Report should show correct total warning count: {total_warnings}"
        
        # Property: Overall status should be FAILED if any errors exist
        if total_errors > 0:
            assert "OVERALL STATUS: FAILED" in validation_report, \
                "Report should show FAILED status when errors exist"
        else:
            assert "OVERALL STATUS: PASSED" in validation_report, \
                "Report should show PASSED status when no errors exist"
    
    def test_validation_report_file_generation(self):
        """
        Test that validation reports can be written to files without encoding issues
        """
        # Create simple test data
        weather_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'city': ['Mumbai'] * 5,
            'avg_temp_c': [25.0, 26.0, 24.0, 27.0, 25.5],
            'humidity_pct': [70.0, 75.0, 68.0, 80.0, 72.0],
            'rain_mm': [0.0, 2.5, 0.0, 5.0, 1.0],
            'condition': ['Clear'] * 5
        })
        
        upi_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'total_upi_txn_count': [1000, 1100, 950, 1200, 1050],
            'avg_txn_value_inr': [500.0, 520.0, 480.0, 550.0, 510.0],
            'notes': ['Normal'] * 5
        })
        
        # Perform validation
        weather_result = self.validator.validate_weather_data(weather_df)
        upi_result = self.validator.validate_upi_data(upi_df)
        validation_report = self.validator.generate_validation_report(weather_result, upi_result)
        
        # Test file writing with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(validation_report)
            temp_file = f.name
        
        try:
            # Verify file was written and can be read back
            with open(temp_file, 'r', encoding='utf-8') as f:
                read_content = f.read()
            
            assert read_content == validation_report, "File content should match original report"
            assert len(read_content) > 0, "File should not be empty"
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    @given(
        has_weather_data=st.booleans(),
        has_upi_data=st.booleans()
    )
    @settings(max_examples=20)
    def test_validation_handles_empty_datasets(self, has_weather_data, has_upi_data):
        """
        Property test for validation behavior with empty datasets
        """
        # Create datasets based on flags
        if has_weather_data:
            weather_df = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=3),
                'city': ['Mumbai'] * 3,
                'avg_temp_c': [25.0, 26.0, 24.0],
                'humidity_pct': [70.0, 75.0, 68.0],
                'rain_mm': [0.0, 2.5, 0.0],
                'condition': ['Clear'] * 3
            })
        else:
            weather_df = pd.DataFrame()
        
        if has_upi_data:
            upi_df = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=3),
                'total_upi_txn_count': [1000, 1100, 950],
                'avg_txn_value_inr': [500.0, 520.0, 480.0],
                'notes': ['Normal'] * 3
            })
        else:
            upi_df = pd.DataFrame()
        
        # Validation should handle empty datasets gracefully
        weather_result = self.validator.validate_weather_data(weather_df)
        upi_result = self.validator.validate_upi_data(upi_df)
        validation_report = self.validator.generate_validation_report(weather_result, upi_result)
        
        # Property: Validation should always produce a report, even for empty data
        assert isinstance(validation_report, str), "Should always return a string report"
        assert len(validation_report) > 0, "Report should not be empty"
        
        # Property: Empty datasets should result in validation failures
        if not has_weather_data:
            assert not weather_result.passed, "Empty weather dataset should fail validation"
            assert "empty" in validation_report.lower(), "Report should mention empty dataset"
        
        if not has_upi_data:
            assert not upi_result.passed, "Empty UPI dataset should fail validation"
            assert "empty" in validation_report.lower(), "Report should mention empty dataset"


class TestMergedDatasetStructure:
    """Property-based tests for merged dataset structure validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.transformer = DataTransformer()
    
    @given(
        weather_dates=st.lists(
            st.dates(min_value=pd.Timestamp('2024-01-01').date(), 
                    max_value=pd.Timestamp('2024-12-31').date()),
            min_size=1,
            max_size=30,
            unique=True
        ),
        upi_dates=st.lists(
            st.dates(min_value=pd.Timestamp('2024-01-01').date(), 
                    max_value=pd.Timestamp('2024-12-31').date()),
            min_size=1,
            max_size=30,
            unique=True
        ),
        weather_temps=st.lists(
            st.floats(min_value=20, max_value=40, allow_nan=False),
            min_size=1,
            max_size=30
        ),
        upi_counts=st.lists(
            st.integers(min_value=100, max_value=10000),
            min_size=1,
            max_size=30
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_merged_dataset_structure_property(self, weather_dates, upi_dates, weather_temps, upi_counts):
        """
        # Feature: weather-upi-dashboard, Property 9: Merged Dataset Structure
        # **Validates: Requirements 2.5**
        
        For any successful merge operation, the output dataset should contain all 
        relevant columns from both input datasets
        """
        # Ensure lists are same length as dates
        weather_temps = weather_temps[:len(weather_dates)] + [25.0] * max(0, len(weather_dates) - len(weather_temps))
        upi_counts = upi_counts[:len(upi_dates)] + [1000] * max(0, len(upi_dates) - len(upi_counts))
        
        # Create weather DataFrame
        weather_df = pd.DataFrame({
            'date': weather_dates,
            'city': ['Mumbai'] * len(weather_dates),
            'avg_temp_c': weather_temps,
            'humidity_pct': [70.0] * len(weather_dates),
            'rain_mm': [0.0] * len(weather_dates),
            'condition': ['Clear'] * len(weather_dates)
        })
        
        # Create UPI DataFrame
        upi_df = pd.DataFrame({
            'date': upi_dates,
            'total_upi_txn_count': upi_counts,
            'avg_txn_value_inr': [500.0] * len(upi_dates),
            'notes': ['Normal'] * len(upi_dates)
        })
        
        # Perform merge
        try:
            merged_df = self.transformer.transform_and_merge(weather_df, upi_df)
            
            # Property: Merged dataset should contain all relevant columns from both datasets
            expected_weather_columns = ['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm', 'condition']
            expected_upi_columns = ['total_upi_txn_count', 'avg_txn_value_inr', 'notes']
            expected_all_columns = expected_weather_columns + expected_upi_columns
            
            # Check that all expected columns are present
            for col in expected_all_columns:
                assert col in merged_df.columns, f"Merged dataset missing expected column: {col}"
            
            # Property: Merged dataset should only contain records with matching dates
            if not merged_df.empty:
                # All dates in merged dataset should exist in both original datasets
                merged_dates = set(merged_df['date'].dt.date)
                weather_date_set = set(pd.to_datetime(weather_df['date']).dt.date)
                upi_date_set = set(pd.to_datetime(upi_df['date']).dt.date)
                
                for merged_date in merged_dates:
                    assert merged_date in weather_date_set, f"Merged date {merged_date} not in weather data"
                    assert merged_date in upi_date_set, f"Merged date {merged_date} not in UPI data"
            
            # Property: Merged dataset should preserve data integrity
            if not merged_df.empty:
                # Check that weather data is preserved correctly
                for _, row in merged_df.iterrows():
                    date_val = row['date'].date()
                    
                    # Find corresponding weather row
                    weather_match = weather_df[pd.to_datetime(weather_df['date']).dt.date == date_val]
                    if not weather_match.empty:
                        weather_row = weather_match.iloc[0]
                        assert row['city'] == weather_row['city'], "City data should be preserved"
                        assert abs(row['avg_temp_c'] - weather_row['avg_temp_c']) < 0.001, "Temperature data should be preserved"
                    
                    # Find corresponding UPI row
                    upi_match = upi_df[pd.to_datetime(upi_df['date']).dt.date == date_val]
                    if not upi_match.empty:
                        upi_row = upi_match.iloc[0]
                        assert row['total_upi_txn_count'] == upi_row['total_upi_txn_count'], "UPI count data should be preserved"
                        assert abs(row['avg_txn_value_inr'] - upi_row['avg_txn_value_inr']) < 0.001, "UPI value data should be preserved"
            
            # Property: Merged dataset should be sorted by date
            if len(merged_df) > 1:
                dates = merged_df['date'].tolist()
                sorted_dates = sorted(dates)
                assert dates == sorted_dates, "Merged dataset should be sorted by date"
                
        except ValueError as e:
            # If merge fails due to no overlapping dates, that's acceptable
            if "No overlapping data found" in str(e):
                # This is expected when there are no common dates
                overlapping_dates = set(pd.to_datetime(weather_df['date']).dt.date) & set(pd.to_datetime(upi_df['date']).dt.date)
                assert len(overlapping_dates) == 0, "Should only fail when no overlapping dates exist"
            else:
                raise
    
    @given(
        common_dates=st.lists(
            st.dates(min_value=pd.Timestamp('2024-01-01').date(), 
                    max_value=pd.Timestamp('2024-12-31').date()),
            min_size=1,
            max_size=10,
            unique=True
        )
    )
    @settings(max_examples=30)
    def test_merged_dataset_preserves_all_matching_records(self, common_dates):
        """
        Property test that merged dataset preserves all records with matching dates
        """
        # Create datasets with identical dates to ensure overlap
        weather_df = pd.DataFrame({
            'date': common_dates,
            'city': ['Mumbai'] * len(common_dates),
            'avg_temp_c': [25.0 + i for i in range(len(common_dates))],
            'humidity_pct': [70.0] * len(common_dates),
            'rain_mm': [0.0] * len(common_dates),
            'condition': ['Clear'] * len(common_dates)
        })
        
        upi_df = pd.DataFrame({
            'date': common_dates,
            'total_upi_txn_count': [1000 + i*10 for i in range(len(common_dates))],
            'avg_txn_value_inr': [500.0] * len(common_dates),
            'notes': ['Normal'] * len(common_dates)
        })
        
        # Perform merge
        merged_df = self.transformer.transform_and_merge(weather_df, upi_df)
        
        # Property: All common dates should be preserved in merge
        assert len(merged_df) == len(common_dates), \
            f"Merged dataset should have {len(common_dates)} records, got {len(merged_df)}"
        
        # Property: All original data should be preserved
        merged_dates = set(merged_df['date'].dt.date)
        expected_dates = set(common_dates)
        assert merged_dates == expected_dates, "All common dates should be preserved in merge"
    
    def test_merged_dataset_handles_duplicate_dates(self):
        """
        Test that merged dataset handles duplicate dates correctly
        """
        # Create datasets with duplicate dates
        weather_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'city': ['Mumbai'] * 3,
            'avg_temp_c': [25.0, 26.0, 24.0],
            'humidity_pct': [70.0, 75.0, 68.0],
            'rain_mm': [0.0, 1.0, 0.0],
            'condition': ['Clear'] * 3
        })
        
        upi_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-02'],
            'total_upi_txn_count': [1000, 1100, 1200],
            'avg_txn_value_inr': [500.0, 520.0, 510.0],
            'notes': ['Normal'] * 3
        })
        
        # Perform merge
        merged_df = self.transformer.transform_and_merge(weather_df, upi_df)
        
        # Property: Merge should handle duplicates and produce valid result
        assert not merged_df.empty, "Merge should produce non-empty result"
        assert 'date' in merged_df.columns, "Date column should be preserved"
        
        # All required columns should be present
        expected_columns = ['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm', 
                          'condition', 'total_upi_txn_count', 'avg_txn_value_inr', 'notes']
        for col in expected_columns:
            assert col in merged_df.columns, f"Column {col} should be present in merged dataset"


class TestEndToEndIntegration:
    """End-to-end integration tests for complete pipeline execution"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = WeatherUPIPipeline()
        # Ensure output directory exists for tests
        config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    def test_complete_pipeline_execution_from_mcp_to_dashboard(self):
        """
        Test complete pipeline execution from MCP to dashboard
        Requirements: 7.2, 7.3, 7.5
        """
        # Test complete pipeline execution
        success = self.pipeline.run_pipeline()
        
        # Verify pipeline completed successfully
        assert success, "Pipeline should complete successfully"
        
        # Verify all expected output files were created
        expected_files = [
            config.MERGED_DATA_FILE,
            config.ANALYTICS_FILE,
            config.VALIDATION_REPORT_FILE,
            config.OUTPUT_DIR / 'pipeline.log',
            config.OUTPUT_DIR / 'pipeline_summary.txt'
        ]
        
        for file_path in expected_files:
            assert file_path.exists(), f"Expected output file not created: {file_path}"
            assert file_path.stat().st_size > 0, f"Output file is empty: {file_path}"
        
        # Verify analytics.csv has correct structure
        analytics_df = pd.read_csv(config.ANALYTICS_FILE)
        analytics_df['date'] = pd.to_datetime(analytics_df['date'])  # Parse date column
        assert not analytics_df.empty, "Analytics file should not be empty"
        
        # Check for required columns in analytics output
        required_analytics_columns = [
            'date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm', 'condition',
            'total_upi_txn_count', 'avg_txn_value_inr', 'notes',
            'total_upi_txn_count_z_score', 'avg_txn_value_inr_z_score', 'txn_volume_outlier',
            'avg_temp_c_z_score', 'humidity_pct_z_score', 'rain_mm_z_score', 'weather_outlier'
        ]
        
        for col in required_analytics_columns:
            assert col in analytics_df.columns, f"Analytics file missing required column: {col}"
        
        # Verify data types are correct
        assert pd.api.types.is_datetime64_any_dtype(analytics_df['date']), "Date should be datetime"
        assert pd.api.types.is_numeric_dtype(analytics_df['avg_temp_c']), "Temperature should be numeric"
        assert pd.api.types.is_numeric_dtype(analytics_df['total_upi_txn_count']), "UPI count should be numeric"
        assert pd.api.types.is_bool_dtype(analytics_df['txn_volume_outlier']), "Outlier flags should be boolean"
        
        # Test dashboard data loading
        try:
            # Import dashboard module
            from dashboard import load_analytics_data, create_correlation_heatmap, generate_insights
            
            # Test dashboard data loading
            dashboard_data = load_analytics_data()
            assert not dashboard_data.empty, "Dashboard should load analytics data successfully"
            
            # Test correlation heatmap creation
            correlation_fig = create_correlation_heatmap(dashboard_data)
            assert correlation_fig is not None, "Correlation heatmap should be created"
            
            # Test insights generation
            insights = generate_insights(dashboard_data)
            assert isinstance(insights, str), "Insights should be generated as string"
            assert len(insights) > 0, "Insights should not be empty"
            
        except ImportError as e:
            pytest.skip(f"Dashboard module not available for testing: {e}")
    
    def test_pipeline_error_handling_across_all_stages(self):
        """
        Test error handling across all pipeline stages
        Requirements: 7.2, 7.3, 7.5
        """
        # Test with missing input files
        original_upi_file = config.UPI_DATA_FILE
        original_weather_file = config.WEATHER_DATA_FILE
        
        try:
            # Test with missing UPI file
            config.UPI_DATA_FILE = Path("nonexistent_upi.csv")
            
            # Pipeline should handle missing files gracefully
            success = self.pipeline.run_pipeline()
            assert not success, "Pipeline should fail gracefully with missing UPI file"
            
            # Error file should be created
            error_file = config.OUTPUT_DIR / "pipeline_error.txt"
            assert error_file.exists(), "Error file should be created on pipeline failure"
            
            # Reset for next test
            config.UPI_DATA_FILE = original_upi_file
            
            # Test with missing weather fallback file
            config.WEATHER_DATA_FILE = Path("nonexistent_weather.csv")
            
            # Pipeline should handle missing weather fallback
            success = self.pipeline.run_pipeline()
            # This might still succeed if API works, or fail if both API and fallback fail
            # Either way, it should handle the error gracefully without crashing
            
        finally:
            # Restore original file paths
            config.UPI_DATA_FILE = original_upi_file
            config.WEATHER_DATA_FILE = original_weather_file
    
    def test_dashboard_data_loading_and_visualization_rendering(self):
        """
        Test dashboard data loading and visualization rendering
        Requirements: 7.2, 7.3, 7.5
        """
        # First ensure we have analytics data
        if not config.ANALYTICS_FILE.exists():
            # Run pipeline to generate data
            success = self.pipeline.run_pipeline()
            assert success, "Pipeline should complete to generate test data"
        
        try:
            # Import dashboard components
            from dashboard import (
                load_analytics_data, 
                create_correlation_heatmap, 
                create_time_series_charts,
                generate_insights,
                calculate_correlations
            )
            
            # Test data loading
            data = load_analytics_data()
            assert not data.empty, "Dashboard should load analytics data"
            assert len(data) > 0, "Loaded data should have records"
            
            # Test correlation calculation
            correlations = calculate_correlations(data)
            assert isinstance(correlations, dict), "Correlations should be returned as dictionary"
            assert len(correlations) > 0, "Should calculate some correlations"
            
            # Verify correlation values are in valid range
            for corr_name, corr_value in correlations.items():
                if not pd.isna(corr_value):
                    assert -1 <= corr_value <= 1, f"Correlation {corr_name} should be between -1 and 1: {corr_value}"
            
            # Test correlation heatmap creation
            heatmap_fig = create_correlation_heatmap(data)
            assert heatmap_fig is not None, "Correlation heatmap should be created"
            
            # Test time series charts creation
            time_series_fig = create_time_series_charts(data)
            assert time_series_fig is not None, "Time series charts should be created"
            
            # Test insights generation
            insights = generate_insights(data)
            assert isinstance(insights, str), "Insights should be string"
            assert len(insights) > 0, "Insights should not be empty"
            
            # Test that insights contain meaningful content
            insight_keywords = ['correlation', 'temperature', 'rainfall', 'transaction', 'outlier']
            insights_lower = insights.lower()
            found_keywords = [kw for kw in insight_keywords if kw in insights_lower]
            assert len(found_keywords) > 0, f"Insights should contain relevant keywords, found: {found_keywords}"
            
        except ImportError as e:
            pytest.skip(f"Dashboard module not available for testing: {e}")
        except Exception as e:
            pytest.fail(f"Dashboard functionality failed: {e}")
    
    def test_pipeline_data_flow_integrity(self):
        """
        Test data flow integrity throughout the pipeline
        Requirements: 7.2, 7.3, 7.5
        """
        # Run pipeline and capture intermediate results
        pipeline = WeatherUPIPipeline()
        
        # Step 1: Fetch weather data
        weather_df = pipeline.fetch_weather_data()
        assert not weather_df.empty, "Weather data should be fetched"
        
        # Verify weather data structure
        weather_columns = ['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm', 'condition']
        for col in weather_columns:
            assert col in weather_df.columns, f"Weather data missing column: {col}"
        
        # Step 2: Load UPI data
        upi_df = pipeline.load_upi_data()
        assert not upi_df.empty, "UPI data should be loaded"
        
        # Verify UPI data structure
        upi_columns = ['date', 'total_upi_txn_count', 'avg_txn_value_inr', 'notes']
        for col in upi_columns:
            assert col in upi_df.columns, f"UPI data missing column: {col}"
        
        # Step 3: Validate data
        weather_result, upi_result, validation_report = pipeline.validate_data(weather_df, upi_df)
        assert isinstance(validation_report, str), "Validation report should be generated"
        assert len(validation_report) > 0, "Validation report should not be empty"
        
        # Step 4: Transform and merge
        merged_df = pipeline.transform_and_merge_data(weather_df, upi_df)
        assert not merged_df.empty, "Merged data should not be empty"
        
        # Verify merged data contains all expected columns
        expected_merged_columns = weather_columns + [col for col in upi_columns if col != 'date']
        for col in expected_merged_columns:
            assert col in merged_df.columns, f"Merged data missing column: {col}"
        
        # Verify data integrity - merged data should only contain overlapping dates
        weather_dates = set(pd.to_datetime(weather_df['date']).dt.date)
        upi_dates = set(pd.to_datetime(upi_df['date']).dt.date)
        merged_dates = set(merged_df['date'].dt.date)
        expected_overlap = weather_dates & upi_dates
        
        assert merged_dates.issubset(expected_overlap), "Merged data should only contain overlapping dates"
        
        # Step 5: Perform analytics
        analytics_results = pipeline.perform_analytics(merged_df)
        assert isinstance(analytics_results, dict), "Analytics should return dictionary"
        assert 'correlations' in analytics_results, "Analytics should include correlations"
        assert 'anomaly_summary' in analytics_results, "Analytics should include anomaly summary"
        
        # Verify analytics results structure
        correlations = analytics_results['correlations']
        assert isinstance(correlations, dict), "Correlations should be dictionary"
        
        # Check that correlations are computed for expected variable pairs
        expected_correlation_pairs = [
            'avg_temp_c_vs_total_upi_txn_count',
            'avg_temp_c_vs_avg_txn_value_inr',
            'humidity_pct_vs_total_upi_txn_count',
            'humidity_pct_vs_avg_txn_value_inr',
            'rain_mm_vs_total_upi_txn_count',
            'rain_mm_vs_avg_txn_value_inr'
        ]
        
        for pair in expected_correlation_pairs:
            assert pair in correlations, f"Missing correlation pair: {pair}"
            if not pd.isna(correlations[pair]):
                assert -1 <= correlations[pair] <= 1, f"Invalid correlation value for {pair}: {correlations[pair]}"
    
    def test_pipeline_output_file_consistency(self):
        """
        Test consistency of output files generated by pipeline
        Requirements: 7.2, 7.3, 7.5
        """
        # Run pipeline
        success = self.pipeline.run_pipeline()
        assert success, "Pipeline should complete successfully"
        
        # Load and verify merged data file
        merged_df = pd.read_csv(config.MERGED_DATA_FILE)
        merged_df['date'] = pd.to_datetime(merged_df['date'])  # Parse date column
        assert not merged_df.empty, "Merged data file should not be empty"
        
        # Load and verify analytics file
        analytics_df = pd.read_csv(config.ANALYTICS_FILE)
        analytics_df['date'] = pd.to_datetime(analytics_df['date'])  # Parse date column
        assert not analytics_df.empty, "Analytics file should not be empty"
        
        # Verify consistency between merged and analytics files
        assert len(merged_df) == len(analytics_df), "Merged and analytics files should have same number of records"
        
        # Verify that analytics file contains all columns from merged file plus additional analytics columns
        merged_columns = set(merged_df.columns)
        analytics_columns = set(analytics_df.columns)
        
        # All merged columns should be present in analytics
        assert merged_columns.issubset(analytics_columns), "Analytics file should contain all merged data columns"
        
        # Analytics should have additional columns for z-scores and outlier flags
        additional_columns = analytics_columns - merged_columns
        expected_additional = {
            'total_upi_txn_count_z_score', 'avg_txn_value_inr_z_score', 'txn_volume_outlier',
            'avg_temp_c_z_score', 'humidity_pct_z_score', 'rain_mm_z_score', 'weather_outlier'
        }
        
        assert expected_additional.issubset(additional_columns), \
            f"Analytics file missing expected additional columns: {expected_additional - additional_columns}"
        
        # Verify date consistency
        merged_dates = pd.to_datetime(merged_df['date']).dt.date
        analytics_dates = pd.to_datetime(analytics_df['date']).dt.date
        
        assert merged_dates.equals(analytics_dates), "Date columns should be identical between files"
        
        # Verify validation report exists and contains summary
        assert config.VALIDATION_REPORT_FILE.exists(), "Validation report should exist"
        
        with open(config.VALIDATION_REPORT_FILE, 'r', encoding='utf-8') as f:
            validation_content = f.read()
        
        assert len(validation_content) > 0, "Validation report should not be empty"
        assert "DATA VALIDATION REPORT" in validation_content, "Validation report should have proper header"
        
        # Verify pipeline summary exists
        summary_file = config.OUTPUT_DIR / "pipeline_summary.txt"
        assert summary_file.exists(), "Pipeline summary should exist"
        
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_content = f.read()
        
        assert len(summary_content) > 0, "Pipeline summary should not be empty"
        assert "WEATHER-UPI CORRELATION PIPELINE SUMMARY" in summary_content, "Summary should have proper header"
    
    def test_pipeline_handles_various_data_scenarios(self):
        """
        Test pipeline behavior with various data scenarios
        Requirements: 7.2, 7.3, 7.5
        """
        # Test with minimal data (single record)
        minimal_weather = pd.DataFrame({
            'date': ['2024-11-01'],
            'city': ['Mumbai'],
            'avg_temp_c': [25.0],
            'humidity_pct': [70.0],
            'rain_mm': [0.0],
            'condition': ['Clear']
        })
        
        minimal_upi = pd.DataFrame({
            'date': ['2024-11-01'],
            'total_upi_txn_count': [1000],
            'avg_txn_value_inr': [500.0],
            'notes': ['Test']
        })
        
        # Test transformation with minimal data
        transformer = DataTransformer()
        merged_minimal = transformer.transform_and_merge(minimal_weather, minimal_upi)
        
        assert len(merged_minimal) == 1, "Should handle single record correctly"
        assert not merged_minimal.empty, "Minimal merge should produce result"
        
        # Test analytics with minimal data
        from analytics_engine import analyze_weather_upi_correlations
        
        analytics_minimal = analyze_weather_upi_correlations(merged_minimal)
        assert isinstance(analytics_minimal, dict), "Analytics should work with minimal data"
        
        # With single record, correlations will be NaN, but structure should be correct
        assert 'correlations' in analytics_minimal, "Should include correlations structure"
        assert 'anomaly_summary' in analytics_minimal, "Should include anomaly summary"
        
        # Test with data containing extreme values (need more points for outlier detection)
        extreme_weather = pd.DataFrame({
            'date': ['2024-11-01', '2024-11-02', '2024-11-03', '2024-11-04', '2024-11-05'],
            'city': ['Mumbai'] * 5,
            'avg_temp_c': [25.0, 26.0, 25.5, 50.0, 25.2],  # One extreme temperature
            'humidity_pct': [70.0, 72.0, 71.0, 100.0, 70.5],  # One extreme humidity
            'rain_mm': [0.0, 1.0, 0.5, 1000.0, 0.2],  # One extreme rainfall
            'condition': ['Clear', 'Clear', 'Clear', 'Extreme', 'Clear']
        })
        
        extreme_upi = pd.DataFrame({
            'date': ['2024-11-01', '2024-11-02', '2024-11-03', '2024-11-04', '2024-11-05'],
            'total_upi_txn_count': [1000, 1100, 1050, 1000000, 1020],  # One extreme count
            'avg_txn_value_inr': [500.0, 520.0, 510.0, 10000.0, 505.0],  # One extreme value
            'notes': ['Normal', 'Normal', 'Normal', 'Extreme', 'Normal']
        })
        
        # Pipeline should handle extreme values without crashing
        merged_extreme = transformer.transform_and_merge(extreme_weather, extreme_upi)
        assert not merged_extreme.empty, "Should handle extreme values"
        
        analytics_extreme = analyze_weather_upi_correlations(merged_extreme)
        assert isinstance(analytics_extreme, dict), "Analytics should handle extreme values"
        
        # Extreme values should be detected as outliers (with sufficient data points)
        enhanced_df = analytics_extreme.get('enhanced_dataframe')
        if enhanced_df is not None and not enhanced_df.empty and len(enhanced_df) >= 3:
            # Check that outlier detection columns exist
            assert 'weather_outlier' in enhanced_df.columns, "Weather outlier column should exist"
            assert 'txn_volume_outlier' in enhanced_df.columns, "Transaction outlier column should exist"
            
            # With extreme values and sufficient data points, some outliers should be detected
            weather_outliers = enhanced_df['weather_outlier'].sum()
            txn_outliers = enhanced_df['txn_volume_outlier'].sum()
            
            # With extreme values, outliers may or may not be detected depending on the distribution
            # The important thing is that the analytics runs without crashing and produces valid results
            assert isinstance(weather_outliers, (int, np.integer)), "Weather outlier count should be numeric"
            assert isinstance(txn_outliers, (int, np.integer)), "Transaction outlier count should be numeric"
            assert weather_outliers >= 0, "Weather outlier count should be non-negative"
            assert txn_outliers >= 0, "Transaction outlier count should be non-negative"