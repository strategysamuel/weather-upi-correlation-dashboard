"""
Data Loading Module

This module handles loading UPI transaction data from CSV files with comprehensive
error handling and encoding detection.
"""

import pandas as pd
import chardet
import logging
from typing import Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UPIDataLoader:
    """Handles loading and initial validation of UPI transaction CSV files"""
    
    def __init__(self):
        self.required_columns = ['date', 'total_upi_txn_count', 'avg_txn_value_inr']
        self.optional_columns = ['notes']
    
    def load_upi_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load UPI transaction data from CSV file with error handling
        
        Args:
            csv_path: Path to the UPI CSV file
            
        Returns:
            DataFrame with UPI transaction data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV has invalid structure
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"UPI CSV file not found: {csv_path}")
        
        try:
            logger.info(f"Loading UPI data from {csv_path}")
            
            # Detect encoding
            encoding = self._detect_encoding(csv_path)
            logger.info(f"Detected encoding: {encoding}")
            
            # Load CSV with detected encoding
            df = self._load_csv_with_fallback(csv_path, encoding)
            
            # Validate structure
            self._validate_csv_structure(df, csv_path)
            
            # Standardize data types
            df = self._standardize_data_types(df)
            
            logger.info(f"Successfully loaded {len(df)} UPI transaction records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading UPI data from {csv_path}: {e}")
            raise
    
    def _detect_encoding(self, csv_path: str) -> str:
        """
        Detect file encoding using chardet
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Detected encoding string
        """
        try:
            with open(csv_path, 'rb') as file:
                raw_data = file.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                logger.info(f"Encoding detection confidence: {confidence:.2f}")
                
                # Fallback to utf-8 if confidence is too low
                if confidence < 0.7:
                    logger.warning("Low encoding confidence, using utf-8 as fallback")
                    encoding = 'utf-8'
                
                return encoding
                
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def _load_csv_with_fallback(self, csv_path: str, primary_encoding: str) -> pd.DataFrame:
        """
        Load CSV with encoding fallback mechanisms
        
        Args:
            csv_path: Path to CSV file
            primary_encoding: Primary encoding to try
            
        Returns:
            Loaded DataFrame
        """
        encodings_to_try = [primary_encoding, 'utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                logger.info(f"Attempting to load CSV with encoding: {encoding}")
                df = pd.read_csv(csv_path, encoding=encoding)
                logger.info(f"Successfully loaded CSV with encoding: {encoding}")
                return df
                
            except UnicodeDecodeError as e:
                logger.warning(f"Failed to load with encoding {encoding}: {e}")
                continue
            except pd.errors.EmptyDataError:
                raise ValueError(f"CSV file is empty: {csv_path}")
            except pd.errors.ParserError as e:
                raise ValueError(f"CSV parsing error: {e}")
        
        raise ValueError(f"Could not load CSV with any supported encoding: {csv_path}")
    
    def _validate_csv_structure(self, df: pd.DataFrame, csv_path: str) -> None:
        """
        Validate that CSV has required columns and basic structure
        
        Args:
            df: Loaded DataFrame
            csv_path: Path to CSV file for error reporting
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError(f"CSV file is empty: {csv_path}")
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")
        
        # Check for completely empty columns
        empty_required_cols = [col for col in self.required_columns 
                              if df[col].isna().all()]
        if empty_required_cols:
            raise ValueError(f"Required columns are completely empty in {csv_path}: {empty_required_cols}")
        
        logger.info(f"CSV structure validation passed for {csv_path}")
    
    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize data types for UPI transaction data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            DataFrame with standardized data types
        """
        try:
            # Convert date column
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Convert numeric columns
            df['total_upi_txn_count'] = pd.to_numeric(df['total_upi_txn_count'], errors='coerce')
            df['avg_txn_value_inr'] = pd.to_numeric(df['avg_txn_value_inr'], errors='coerce')
            
            # Keep notes as string if present
            if 'notes' in df.columns:
                df['notes'] = df['notes'].astype(str)
            
            # Check for conversion failures
            date_failures = df['date'].isna().sum()
            txn_count_failures = df['total_upi_txn_count'].isna().sum()
            txn_value_failures = df['avg_txn_value_inr'].isna().sum()
            
            if date_failures > 0:
                logger.warning(f"Failed to parse {date_failures} date values")
            if txn_count_failures > 0:
                logger.warning(f"Failed to parse {txn_count_failures} transaction count values")
            if txn_value_failures > 0:
                logger.warning(f"Failed to parse {txn_value_failures} transaction value amounts")
            
            return df
            
        except Exception as e:
            logger.error(f"Error standardizing data types: {e}")
            raise

def load_weather_fallback(csv_path: str = "weather_mumbai_2024_11_synthetic.csv") -> pd.DataFrame:
    """
    Load weather data from local CSV file (fallback function)
    
    Args:
        csv_path: Path to weather CSV file
        
    Returns:
        Weather DataFrame
    """
    try:
        logger.info(f"Loading weather fallback data from {csv_path}")
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Loaded {len(df)} weather records from fallback CSV")
        return df
        
    except Exception as e:
        logger.error(f"Error loading weather fallback data: {e}")
        raise

# Convenience functions
def load_upi_csv(csv_path: str) -> pd.DataFrame:
    """
    Convenience function to load UPI data
    
    Args:
        csv_path: Path to UPI CSV file
        
    Returns:
        UPI DataFrame
    """
    loader = UPIDataLoader()
    return loader.load_upi_data(csv_path)