"""
Data Transformation Module

This module handles standardization of column names, date format normalization,
and dataset merging functionality using date as the primary key.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import re
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from upi_simulator import apply_weather_influence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    """Handles data transformation and merging operations"""
    
    def __init__(self):
        # Define standard column mappings
        self.weather_column_mapping = {
            'date': 'date',
            'date_col': 'date',
            'city': 'city',
            'location': 'city',
            'avg_temp_c': 'avg_temp_c',
            'avg_temp_c': 'avg_temp_c',
            'temperature': 'avg_temp_c',
            'temp_c': 'avg_temp_c',
            'temperature_c': 'avg_temp_c',
            'humidity_pct': 'humidity_pct',
            'humidity': 'humidity_pct',
            'relative_humidity': 'humidity_pct',
            'rain_mm': 'rain_mm',
            'rainfall': 'rain_mm',
            'precipitation': 'rain_mm',
            'precipitation_mm': 'rain_mm',
            'condition': 'condition',
            'weather_condition': 'condition'
        }
        
        self.upi_column_mapping = {
            'date': 'date',
            'total_upi_txn_count': 'total_upi_txn_count',
            'txn_count': 'total_upi_txn_count',
            'transaction_count': 'total_upi_txn_count',
            'upi_transactions': 'total_upi_txn_count',
            'avg_txn_value_inr': 'avg_txn_value_inr',
            'avg_value': 'avg_txn_value_inr',
            'average_transaction_value': 'avg_txn_value_inr',
            'txn_value': 'avg_txn_value_inr',
            'notes': 'notes'
        }
    
    def standardize_columns(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Standardize column names to consistent format for data integration.
        
        This method normalizes column names across different data sources to ensure
        consistent naming conventions. It handles variations in column naming such as
        'temperature' vs 'avg_temp_c' and maps them to standard names.
        
        Args:
            df (pd.DataFrame): Input DataFrame with potentially inconsistent column names
            dataset_type (str): Type of dataset - either 'weather' or 'upi'
                               Determines which column mapping rules to apply
            
        Returns:
            pd.DataFrame: DataFrame with standardized column names that follow
                         consistent naming conventions
            
        Raises:
            ValueError: If dataset_type is not 'weather' or 'upi', or if required
                       columns are missing after standardization
                       
        Example:
            >>> transformer = DataTransformer()
            >>> weather_df = pd.DataFrame({'temperature': [25, 26], 'rainfall': [0, 5]})
            >>> standardized = transformer.standardize_columns(weather_df, 'weather')
            >>> print(standardized.columns.tolist())
            ['avg_temp_c', 'rain_mm']
            
        Requirements: 2.1 (column name standardization)
        """
        if dataset_type not in ['weather', 'upi']:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'weather' or 'upi'")
        
        logger.info(f"Standardizing {dataset_type} column names")
        
        # Get appropriate column mapping
        if dataset_type == 'weather':
            column_mapping = self.weather_column_mapping
            required_standard_columns = ['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm']
        else:  # upi
            column_mapping = self.upi_column_mapping
            required_standard_columns = ['date', 'total_upi_txn_count', 'avg_txn_value_inr']
        
        # Create a copy to avoid modifying original
        standardized_df = df.copy()
        
        # Normalize column names (lowercase, remove spaces/special chars)
        normalized_columns = {}
        for col in standardized_df.columns:
            normalized_col = self._normalize_column_name(col)
            normalized_columns[col] = normalized_col
        
        # Rename columns to normalized versions
        standardized_df = standardized_df.rename(columns=normalized_columns)
        
        # Map normalized columns to standard names
        final_mapping = {}
        for old_col, normalized_col in normalized_columns.items():
            if normalized_col in column_mapping:
                final_mapping[normalized_col] = column_mapping[normalized_col]
            else:
                # Keep original column name if no mapping found
                final_mapping[normalized_col] = normalized_col
        
        # Apply final mapping
        standardized_df = standardized_df.rename(columns=final_mapping)
        
        # Verify required columns are present after standardization
        missing_columns = [col for col in required_standard_columns 
                          if col not in standardized_df.columns]
        
        if missing_columns:
            available_cols = list(standardized_df.columns)
            raise ValueError(f"Missing required columns after standardization: {missing_columns}. "
                           f"Available columns: {available_cols}")
        
        logger.info(f"Successfully standardized {dataset_type} columns: {list(standardized_df.columns)}")
        return standardized_df
    
    def _normalize_column_name(self, column_name: str) -> str:
        """
        Normalize column name to lowercase with underscores for consistency.
        
        This internal method converts column names to a standard format by:
        1. Converting to lowercase
        2. Replacing spaces and special characters with underscores
        3. Removing consecutive underscores
        4. Trimming leading/trailing underscores
        
        Args:
            column_name (str): Original column name that may contain mixed case,
                              spaces, or special characters
            
        Returns:
            str: Normalized column name in lowercase with underscores
            
        Example:
            >>> transformer = DataTransformer()
            >>> normalized = transformer._normalize_column_name("Avg Temp (°C)")
            >>> print(normalized)
            'avg_temp_c'
        """
        # Convert to lowercase
        normalized = column_name.lower()
        
        # Replace spaces and special characters with underscores
        normalized = re.sub(r'[^a-z0-9_]', '_', normalized)
        
        # Remove multiple consecutive underscores
        normalized = re.sub(r'_+', '_', normalized)
        
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        
        return normalized
    
    def normalize_dates(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Normalize date formats to ensure consistency across datasets.
        
        This method handles various date formats and converts them to a standardized
        datetime format. It attempts multiple parsing strategies to handle ambiguous
        date formats and provides intelligent fallback mechanisms.
        
        The method tries formats in this order:
        1. Unambiguous formats (YYYY-MM-DD, YYYY/MM/DD, etc.)
        2. Ambiguous slash formats (MM/DD/YYYY vs DD/MM/YYYY) - picks best match
        3. Pandas automatic parsing as final fallback
        
        Args:
            df (pd.DataFrame): Input DataFrame containing date column to normalize
            date_column (str, optional): Name of the date column. Defaults to 'date'
            
        Returns:
            pd.DataFrame: DataFrame with normalized dates in standard datetime format
                         (date only, no time component)
            
        Raises:
            ValueError: If date column is missing from DataFrame or if critical
                       date conversion failures occur
                       
        Example:
            >>> df = pd.DataFrame({'date': ['2024-01-15', '01/16/2024', '2024/01/17']})
            >>> transformer = DataTransformer()
            >>> normalized = transformer.normalize_dates(df)
            >>> print(normalized['date'].dtype)
            datetime64[ns]
            
        Requirements: 2.3 (date format standardization)
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        
        logger.info(f"Normalizing dates in column '{date_column}'")
        
        # Create a copy to avoid modifying original
        normalized_df = df.copy()
        
        # Convert to datetime with error handling
        try:
            # Try multiple date formats, being smart about ambiguous cases
            # Start with unambiguous formats that clearly indicate year/month/day order
            date_formats = [
                '%Y-%m-%d',          # ISO format: 2024-01-15
                '%Y-%m-%d %H:%M:%S', # ISO with time: 2024-01-15 10:30:00
                '%Y/%m/%d',          # Year first slash: 2024/01/15
                '%d-%m-%Y'           # Day first dash: 15-01-2024
            ]
            
            # Handle ambiguous slash formats separately (MM/DD/YYYY vs DD/MM/YYYY)
            # These require special handling since 01/02/2024 could be Jan 2 or Feb 1
            ambiguous_formats = ['%m/%d/%Y', '%d/%m/%Y']
            
            # Initialize all dates as NaT (Not a Time) - pandas null for datetime
            normalized_df[date_column] = pd.NaT
            
            # First pass: try unambiguous formats
            # Process each format and update only previously unparsed dates
            for date_format in date_formats:
                try:
                    # Only attempt parsing on dates that haven't been successfully parsed yet
                    mask = normalized_df[date_column].isna()
                    if mask.any():
                        # Parse dates with current format, coerce errors to NaT
                        parsed_dates = pd.to_datetime(
                            df.loc[mask, date_column], 
                            format=date_format, 
                            errors='coerce'
                        )
                        # Identify which dates were successfully parsed (not NaT)
                        valid_parsed = ~parsed_dates.isna()
                        if valid_parsed.any():
                            # Update only the successfully parsed dates
                            normalized_df.loc[mask & valid_parsed, date_column] = parsed_dates[valid_parsed]
                except:
                    # If format parsing fails entirely, continue to next format
                    continue
            
            # Second pass: handle ambiguous slash formats intelligently
            # For formats like MM/DD/YYYY vs DD/MM/YYYY, try both and pick the best match
            if normalized_df[date_column].isna().any():
                mask = normalized_df[date_column].isna()
                if mask.any():
                    best_format = None
                    best_count = 0
                    best_parsed = None
                    
                    # Try each ambiguous format and count successful parses
                    for amb_format in ambiguous_formats:
                        try:
                            parsed_dates = pd.to_datetime(
                                df.loc[mask, date_column], 
                                format=amb_format, 
                                errors='coerce'
                            )
                            # Count how many dates were successfully parsed
                            valid_count = (~parsed_dates.isna()).sum()
                            
                            # Keep track of the format that parses the most dates
                            # This heuristic assumes the correct format will parse more dates
                            if valid_count > best_count:
                                best_count = valid_count
                                best_format = amb_format
                                best_parsed = parsed_dates
                        except:
                            continue
                    
                    # Apply the best-performing ambiguous format
                    if best_parsed is not None and best_count > 0:
                        valid_parsed = ~best_parsed.isna()
                        normalized_df.loc[mask & valid_parsed, date_column] = best_parsed[valid_parsed]
            
            # Final fallback to pandas automatic parsing
            if normalized_df[date_column].isna().any():
                mask = normalized_df[date_column].isna()
                if mask.any():
                    fallback_dates = pd.to_datetime(df.loc[mask, date_column], errors='coerce')
                    valid_fallback = ~fallback_dates.isna()
                    if valid_fallback.any():
                        normalized_df.loc[mask & valid_fallback, date_column] = fallback_dates[valid_fallback]
            
            final_na_count = normalized_df[date_column].isna().sum()
            if final_na_count > 0:
                logger.warning(f"Could not parse {final_na_count} date values")
            
            # Ensure dates are in standard format (date only, no time)
            normalized_df[date_column] = normalized_df[date_column].dt.date
            normalized_df[date_column] = pd.to_datetime(normalized_df[date_column])
            
            logger.info(f"Successfully normalized {len(normalized_df)} dates")
            return normalized_df
            
        except Exception as e:
            logger.error(f"Error normalizing dates: {e}")
            raise ValueError(f"Failed to normalize dates in column '{date_column}': {e}")
    
    def merge_datasets(self, weather_df: pd.DataFrame, upi_df: pd.DataFrame, 
                      date_column: str = 'date') -> pd.DataFrame:
        """
        Merge weather and UPI datasets using date as the primary key.
        
        This method performs an inner join on the date column to combine weather
        and UPI transaction data. It ensures both datasets have normalized dates
        before merging and provides comprehensive logging of the merge process.
        
        The merge process:
        1. Validates input DataFrames are not empty
        2. Ensures date columns exist in both datasets
        3. Normalizes dates in both datasets
        4. Removes records with invalid dates (NaT values)
        5. Performs inner join to keep only matching dates
        6. Sorts result by date and resets index
        
        Args:
            weather_df (pd.DataFrame): Weather DataFrame with standardized columns
                                      Must contain the specified date column
            upi_df (pd.DataFrame): UPI DataFrame with standardized columns
                                  Must contain the specified date column
            date_column (str, optional): Name of the date column to merge on.
                                        Defaults to 'date'
            
        Returns:
            pd.DataFrame: Merged DataFrame containing all relevant columns from both
                         input datasets, sorted by date with only matching dates
            
        Raises:
            ValueError: If either DataFrame is empty, date columns are missing,
                       or merge operation fails
                       
        Example:
            >>> weather_df = pd.DataFrame({
            ...     'date': ['2024-01-01', '2024-01-02'],
            ...     'avg_temp_c': [25, 26]
            ... })
            >>> upi_df = pd.DataFrame({
            ...     'date': ['2024-01-01', '2024-01-02'],
            ...     'total_upi_txn_count': [1000, 1100]
            ... })
            >>> transformer = DataTransformer()
            >>> merged = transformer.merge_datasets(weather_df, upi_df)
            >>> print(len(merged))
            2
            
        Requirements: 2.2, 2.4 (date-based merging with data preservation)
        """
        logger.info("Merging weather and UPI datasets")
        
        # Validate inputs
        if weather_df.empty:
            raise ValueError("Weather DataFrame is empty")
        if upi_df.empty:
            raise ValueError("UPI DataFrame is empty")
        
        if date_column not in weather_df.columns:
            raise ValueError(f"Date column '{date_column}' not found in weather DataFrame")
        if date_column not in upi_df.columns:
            raise ValueError(f"Date column '{date_column}' not found in UPI DataFrame")
        
        # Create copies to avoid modifying originals
        weather_clean = weather_df.copy()
        upi_clean = upi_df.copy()
        
        # Ensure both datasets have normalized dates
        weather_clean = self.normalize_dates(weather_clean, date_column)
        upi_clean = self.normalize_dates(upi_clean, date_column)
        
        # Remove any rows with NaT dates before merging
        initial_weather_count = len(weather_clean)
        initial_upi_count = len(upi_clean)
        
        weather_clean = weather_clean.dropna(subset=[date_column])
        upi_clean = upi_clean.dropna(subset=[date_column])
        
        weather_dropped = initial_weather_count - len(weather_clean)
        upi_dropped = initial_upi_count - len(upi_clean)
        
        if weather_dropped > 0:
            logger.warning(f"Dropped {weather_dropped} weather records with invalid dates")
        if upi_dropped > 0:
            logger.warning(f"Dropped {upi_dropped} UPI records with invalid dates")
        
        # Check for overlapping date ranges
        weather_date_range = (weather_clean[date_column].min(), weather_clean[date_column].max())
        upi_date_range = (upi_clean[date_column].min(), upi_clean[date_column].max())
        
        logger.info(f"Weather date range: {weather_date_range[0]} to {weather_date_range[1]}")
        logger.info(f"UPI date range: {upi_date_range[0]} to {upi_date_range[1]}")
        
        # Perform inner join to keep only matching dates
        merged_df = pd.merge(
            weather_clean, 
            upi_clean, 
            on=date_column, 
            how='inner',
            suffixes=('_weather', '_upi')
        )
        
        if merged_df.empty:
            logger.warning("No overlapping dates found between weather and UPI datasets")
        
        # Sort by date
        merged_df = merged_df.sort_values(date_column).reset_index(drop=True)
        
        logger.info(f"Successfully merged datasets: {len(merged_df)} records with matching dates")
        logger.info(f"Merged DataFrame columns: {list(merged_df.columns)}")
        
        return merged_df
    
    def transform_and_merge(self, weather_df: pd.DataFrame, upi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete transformation pipeline: standardize columns, normalize dates, and merge with weather influence
        
        Args:
            weather_df: Raw weather DataFrame
            upi_df: Raw UPI DataFrame
            
        Returns:
            Fully transformed and merged DataFrame with weather influence applied
        """
        logger.info("Starting complete transformation pipeline")
        
        try:
            # Step 1: Standardize column names
            weather_standardized = self.standardize_columns(weather_df, 'weather')
            upi_standardized = self.standardize_columns(upi_df, 'upi')
            
            # Step 2: Apply weather influence to UPI data and merge
            merged_df = apply_weather_influence(upi_standardized, weather_standardized)
            
            logger.info("Transformation pipeline completed successfully")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error in transformation pipeline: {e}")
            raise

# Convenience functions
def standardize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to standardize weather column names
    
    Args:
        df: Weather DataFrame
        
    Returns:
        DataFrame with standardized columns
    """
    transformer = DataTransformer()
    return transformer.standardize_columns(df, 'weather')

def standardize_upi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to standardize UPI column names
    
    Args:
        df: UPI DataFrame
        
    Returns:
        DataFrame with standardized columns
    """
    transformer = DataTransformer()
    return transformer.standardize_columns(df, 'upi')

def merge_weather_upi_data(weather_df: pd.DataFrame, upi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to merge weather and UPI data
    
    Args:
        weather_df: Weather DataFrame
        upi_df: UPI DataFrame
        
    Returns:
        Merged DataFrame
    """
    transformer = DataTransformer()
    return transformer.transform_and_merge(weather_df, upi_df)