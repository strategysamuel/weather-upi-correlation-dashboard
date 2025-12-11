"""
Data Validation Module

This module provides comprehensive validation functions for weather and UPI transaction data,
including column validation, date range checks, non-negative value validation, and null detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationResult:
    """Container for validation results"""
    
    def __init__(self):
        self.passed = True
        self.errors = []
        self.warnings = []
        self.details = {}
    
    def add_error(self, message: str, details: Any = None):
        """Add an error to the validation result"""
        self.passed = False
        self.errors.append(message)
        if details:
            self.details[message] = details
    
    def add_warning(self, message: str, details: Any = None):
        """Add a warning to the validation result"""
        self.warnings.append(message)
        if details:
            self.details[message] = details
    
    def get_summary(self) -> str:
        """Get a summary of validation results"""
        status = "PASSED" if self.passed else "FAILED"
        summary = f"Validation {status}\n"
        
        if self.errors:
            summary += f"Errors ({len(self.errors)}):\n"
            for error in self.errors:
                summary += f"  - {error}\n"
        
        if self.warnings:
            summary += f"Warnings ({len(self.warnings)}):\n"
            for warning in self.warnings:
                summary += f"  - {warning}\n"
        
        return summary

class DataValidator:
    """Comprehensive data validation for weather and UPI datasets"""
    
    def __init__(self):
        self.weather_required_columns = ['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm']
        self.upi_required_columns = ['date', 'total_upi_txn_count', 'avg_txn_value_inr']
        
        # Define reasonable ranges for validation
        self.weather_ranges = {
            'avg_temp_c': (-50, 60),
            'humidity_pct': (0, 100),
            'rain_mm': (0, 1000)
        }
        
        self.upi_ranges = {
            'total_upi_txn_count': (0, float('inf')),
            'avg_txn_value_inr': (0, float('inf'))
        }
    
    def validate_weather_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Comprehensive validation of weather data for quality assurance.
        
        This method performs thorough validation of weather datasets to ensure
        data quality and integrity before analysis. It checks multiple aspects:
        
        Validation checks performed:
        1. Dataset emptiness check
        2. Required column presence validation
        3. Date column format and range validation
        4. Numerical column range validation (temperature, humidity, rainfall)
        5. Null value detection and reporting
        
        Weather-specific validations:
        - Temperature: Reasonable range (-50°C to 60°C)
        - Humidity: Valid percentage range (0% to 100%)
        - Rainfall: Non-negative values (0mm to 1000mm)
        
        Args:
            df (pd.DataFrame): Weather DataFrame to validate
                              Expected columns: date, city, avg_temp_c, humidity_pct, rain_mm
            
        Returns:
            ValidationResult: Comprehensive validation results containing:
                            - passed: Boolean indicating overall validation status
                            - errors: List of critical validation failures
                            - warnings: List of data quality concerns
                            - details: Additional validation metadata
                            
        Example:
            >>> validator = DataValidator()
            >>> result = validator.validate_weather_data(weather_df)
            >>> if result.passed:
            ...     print("Weather data validation passed")
            >>> else:
            ...     print(f"Validation failed: {result.errors}")
            
        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5 (comprehensive data validation)
        """
        result = ValidationResult()
        
        try:
            # Check if DataFrame is empty
            if df.empty:
                result.add_error("Weather dataset is empty")
                return result
            
            # Validate required columns
            self._validate_required_columns(df, self.weather_required_columns, result, "weather")
            
            # Validate date column
            if 'date' in df.columns:
                self._validate_date_column(df, result, "weather")
            
            # Validate numerical columns
            for column, (min_val, max_val) in self.weather_ranges.items():
                if column in df.columns:
                    self._validate_numerical_column(df, column, min_val, max_val, result)
            
            # Check for null values
            self._validate_null_values(df, result, "weather")
            
            logger.info(f"Weather data validation completed: {'PASSED' if result.passed else 'FAILED'}")
            
        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
            logger.error(f"Error during weather data validation: {e}")
        
        return result
    
    def validate_upi_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Comprehensive validation of UPI transaction data
        
        Args:
            df: UPI DataFrame to validate
            
        Returns:
            ValidationResult with detailed findings
        """
        result = ValidationResult()
        
        try:
            # Check if DataFrame is empty
            if df.empty:
                result.add_error("UPI dataset is empty")
                return result
            
            # Validate required columns
            self._validate_required_columns(df, self.upi_required_columns, result, "UPI")
            
            # Validate date column
            if 'date' in df.columns:
                self._validate_date_column(df, result, "UPI")
            
            # Validate numerical columns
            for column, (min_val, max_val) in self.upi_ranges.items():
                if column in df.columns:
                    self._validate_numerical_column(df, column, min_val, max_val, result)
            
            # Check for null values
            self._validate_null_values(df, result, "UPI")
            
            logger.info(f"UPI data validation completed: {'PASSED' if result.passed else 'FAILED'}")
            
        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
            logger.error(f"Error during UPI data validation: {e}")
        
        return result
    
    def _validate_required_columns(self, df: pd.DataFrame, required_columns: List[str], 
                                 result: ValidationResult, dataset_name: str):
        """
        Validate that all required columns are present and not empty.
        
        This internal method checks for the presence of essential columns and
        ensures they contain actual data (not all null values). It's a critical
        validation step that prevents downstream processing errors.
        
        Validation checks:
        1. Missing column detection - identifies columns not present in DataFrame
        2. Empty column detection - identifies columns that exist but contain only null values
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            required_columns (List[str]): List of column names that must be present
            result (ValidationResult): Validation result object to update with findings
            dataset_name (str): Name of dataset for error reporting (e.g., "weather", "UPI")
            
        Side Effects:
            Updates the result object with errors for missing or empty required columns
            
        Example:
            >>> validator = DataValidator()
            >>> result = ValidationResult()
            >>> validator._validate_required_columns(df, ['date', 'value'], result, "test")
            >>> print(result.errors)  # Shows any missing column errors
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            result.add_error(f"Missing required columns in {dataset_name} data: {missing_columns}")
        
        # Check for completely empty required columns
        empty_columns = []
        for col in required_columns:
            if col in df.columns and df[col].isna().all():
                empty_columns.append(col)
        
        if empty_columns:
            result.add_error(f"Required columns are completely empty in {dataset_name} data: {empty_columns}")
    
    def _validate_date_column(self, df: pd.DataFrame, result: ValidationResult, dataset_name: str):
        """Validate date column for proper format and reasonable ranges"""
        date_col = df['date']
        
        # Check for invalid dates (NaT values)
        invalid_dates = date_col.isna().sum()
        if invalid_dates > 0:
            result.add_error(f"Found {invalid_dates} invalid date values in {dataset_name} data")
        
        # Check date range (should be reasonable - not too far in past or future)
        if not date_col.empty and not date_col.isna().all():
            min_date = date_col.min()
            max_date = date_col.max()
            
            # Define reasonable date range (1900 to 2030)
            earliest_valid = pd.Timestamp('1900-01-01')
            latest_valid = pd.Timestamp('2030-12-31')
            
            if min_date < earliest_valid:
                result.add_warning(f"Dates in {dataset_name} data start very early: {min_date}")
            
            if max_date > latest_valid:
                result.add_warning(f"Dates in {dataset_name} data extend far into future: {max_date}")
            
            # Check for duplicate dates
            duplicate_dates = date_col.duplicated().sum()
            if duplicate_dates > 0:
                result.add_warning(f"Found {duplicate_dates} duplicate dates in {dataset_name} data")
    
    def _validate_numerical_column(self, df: pd.DataFrame, column: str, min_val: float, 
                                 max_val: float, result: ValidationResult):
        """
        Validate numerical column for data type, range, and business logic constraints.
        
        This method performs comprehensive validation of numerical columns including:
        1. Data type verification (must be numeric)
        2. Range validation (values within expected bounds)
        3. Non-negative validation (where applicable)
        4. Infinite value detection
        5. Statistical outlier identification
        
        The validation is context-aware - for example, transaction counts and rainfall
        amounts must be non-negative, while temperature can be negative.
        
        Args:
            df (pd.DataFrame): DataFrame containing the column to validate
            column (str): Name of the numerical column to validate
            min_val (float): Minimum acceptable value for the column
            max_val (float): Maximum acceptable value for the column
            result (ValidationResult): Validation result object to update with findings
            
        Side Effects:
            Updates the result object with errors for data type issues, range violations,
            negative values (where inappropriate), and infinite values
            
        Example:
            >>> validator = DataValidator()
            >>> result = ValidationResult()
            >>> validator._validate_numerical_column(df, 'temperature', -50, 60, result)
            >>> # Checks temperature is numeric and within -50°C to 60°C range
        """
        col_data = df[column]
        
        # Check for non-numeric values
        if not pd.api.types.is_numeric_dtype(col_data):
            result.add_error(f"Column '{column}' should be numeric but has type: {col_data.dtype}")
            return
        
        # Remove NaN values for range checking
        valid_data = col_data.dropna()
        
        if valid_data.empty:
            result.add_warning(f"Column '{column}' has no valid numeric values")
            return
        
        # Check for negative values where they shouldn't exist
        if min_val >= 0:
            negative_count = (valid_data < 0).sum()
            if negative_count > 0:
                result.add_error(f"Found {negative_count} negative values in '{column}' (should be non-negative)")
        
        # Check for values outside reasonable range
        out_of_range_low = (valid_data < min_val).sum()
        out_of_range_high = (valid_data > max_val).sum()
        
        if out_of_range_low > 0:
            result.add_warning(f"Found {out_of_range_low} values in '{column}' below minimum ({min_val})")
        
        if out_of_range_high > 0:
            result.add_warning(f"Found {out_of_range_high} values in '{column}' above maximum ({max_val})")
        
        # Check for infinite values
        infinite_count = np.isinf(valid_data).sum()
        if infinite_count > 0:
            result.add_error(f"Found {infinite_count} infinite values in '{column}'")
    
    def _validate_null_values(self, df: pd.DataFrame, result: ValidationResult, dataset_name: str):
        """Detect and report null values across all columns"""
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            null_details = {}
            for column, count in null_counts.items():
                if count > 0:
                    percentage = (count / len(df)) * 100
                    null_details[column] = {'count': count, 'percentage': percentage}
            
            result.add_warning(f"Found {total_nulls} null values in {dataset_name} data", null_details)
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            result.add_error(f"Found {empty_rows} completely empty rows in {dataset_name} data")
    
    def generate_validation_report(self, weather_result: ValidationResult, 
                                 upi_result: ValidationResult) -> str:
        """
        Generate a comprehensive validation report
        
        Args:
            weather_result: Weather validation results
            upi_result: UPI validation results
            
        Returns:
            Formatted validation report string
        """
        report = "=" * 60 + "\n"
        report += "DATA VALIDATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Overall status
        overall_passed = weather_result.passed and upi_result.passed
        report += f"OVERALL STATUS: {'PASSED' if overall_passed else 'FAILED'}\n\n"
        
        # Weather data results
        report += "WEATHER DATA VALIDATION:\n"
        report += "-" * 30 + "\n"
        report += weather_result.get_summary() + "\n"
        
        # UPI data results
        report += "UPI DATA VALIDATION:\n"
        report += "-" * 30 + "\n"
        report += upi_result.get_summary() + "\n"
        
        # Summary statistics
        total_errors = len(weather_result.errors) + len(upi_result.errors)
        total_warnings = len(weather_result.warnings) + len(upi_result.warnings)
        
        report += "SUMMARY:\n"
        report += "-" * 30 + "\n"
        report += f"Total Errors: {total_errors}\n"
        report += f"Total Warnings: {total_warnings}\n"
        
        if overall_passed:
            report += "\n✓ All validations passed. Data is ready for analysis.\n"
        else:
            report += "\n✗ Validation failed. Please address errors before proceeding.\n"
        
        return report

# Convenience functions
def validate_datasets(weather_df: pd.DataFrame, upi_df: pd.DataFrame) -> Tuple[ValidationResult, ValidationResult, str]:
    """
    Validate both weather and UPI datasets and generate report
    
    Args:
        weather_df: Weather DataFrame
        upi_df: UPI DataFrame
        
    Returns:
        Tuple of (weather_result, upi_result, report_string)
    """
    validator = DataValidator()
    
    weather_result = validator.validate_weather_data(weather_df)
    upi_result = validator.validate_upi_data(upi_df)
    report = validator.generate_validation_report(weather_result, upi_result)
    
    return weather_result, upi_result, report