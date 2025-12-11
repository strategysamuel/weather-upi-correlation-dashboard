#!/usr/bin/env python3
"""
Weather-UPI Dashboard Main Pipeline Script

This script orchestrates the complete data processing pipeline from raw data
to analytics output and dashboard visualization.
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback
import os

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import pipeline modules
from weather_api import get_weather_data
from data_loader import load_upi_csv
from data_validator import validate_datasets
from data_transformer import DataTransformer
from analytics_engine import analyze_weather_upi_correlations
import config

class WeatherUPIPipeline:
    """Main pipeline orchestrator for Weather-UPI correlation analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.transformer = DataTransformer()
        self.results = {}
        
        # Ensure output directory exists
        config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Configure logging for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.OUTPUT_DIR / 'pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def fetch_weather_data(self):
        """
        Step 1: Fetch weather data via MCP API with fallback to CSV
        
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Fetching Weather Data")
        self.logger.info("=" * 60)
        
        try:
            # Fetch weather data for November 2024 to match UPI data
            # Use interactive=True to prompt user for approval if API fails
            weather_df = get_weather_data(
                start_date="2024-11-01", 
                end_date="2024-11-30",
                use_csv_fallback=False,
                interactive=True
            )
            
            if weather_df.empty:
                raise ValueError("No weather data retrieved from API or fallback CSV")
            
            self.results['weather_data'] = weather_df
            self.logger.info(f"Successfully fetched {len(weather_df)} weather records")
            self.logger.info(f"Weather data columns: {list(weather_df.columns)}")
            
            return weather_df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch weather data: {e}")
            raise
    
    def load_upi_data(self):
        """
        Step 2: Load UPI transaction data from CSV
        
        Requirements: 1.1, 1.2
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: Loading UPI Transaction Data")
        self.logger.info("=" * 60)
        
        try:
            # Load UPI data from CSV
            upi_df = load_upi_csv(str(config.UPI_DATA_FILE))
            
            if upi_df.empty:
                raise ValueError("No UPI data loaded from CSV file")
            
            self.results['upi_data'] = upi_df
            self.logger.info(f"Successfully loaded {len(upi_df)} UPI transaction records")
            self.logger.info(f"UPI data columns: {list(upi_df.columns)}")
            
            return upi_df
            
        except Exception as e:
            self.logger.error(f"Failed to load UPI data: {e}")
            raise
    
    def validate_data(self, weather_df, upi_df):
        """
        Step 3: Validate both datasets for quality and completeness
        
        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: Validating Data Quality")
        self.logger.info("=" * 60)
        
        try:
            # Perform comprehensive validation
            weather_result, upi_result, validation_report = validate_datasets(weather_df, upi_df)
            
            # Store validation results
            self.results['weather_validation'] = weather_result
            self.results['upi_validation'] = upi_result
            self.results['validation_report'] = validation_report
            
            # Log validation summary
            self.logger.info("Validation completed:")
            self.logger.info(f"Weather validation: {'PASSED' if weather_result.passed else 'FAILED'}")
            self.logger.info(f"UPI validation: {'PASSED' if upi_result.passed else 'FAILED'}")
            
            # Save validation report to file
            with open(config.VALIDATION_REPORT_FILE, 'w', encoding='utf-8') as f:
                f.write(validation_report)
            self.logger.info(f"Validation report saved to {config.VALIDATION_REPORT_FILE}")
            
            # Check if validation passed
            if not (weather_result.passed and upi_result.passed):
                self.logger.warning("Data validation failed - proceeding with caution")
                self.logger.warning("Check validation report for details")
            
            return weather_result, upi_result, validation_report
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            raise
    
    def transform_and_merge_data(self, weather_df, upi_df):
        """
        Step 4: Transform and merge datasets
        
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: Transforming and Merging Data")
        self.logger.info("=" * 60)
        
        try:
            # Transform and merge datasets
            merged_df = self.transformer.transform_and_merge(weather_df, upi_df)
            
            if merged_df.empty:
                raise ValueError("No overlapping data found between weather and UPI datasets")
            
            self.results['merged_data'] = merged_df
            self.logger.info(f"Successfully merged datasets: {len(merged_df)} records")
            self.logger.info(f"Merged data columns: {list(merged_df.columns)}")
            
            # Save merged dataset
            merged_df.to_csv(config.MERGED_DATA_FILE, index=False)
            self.logger.info(f"Merged dataset saved to {config.MERGED_DATA_FILE}")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Data transformation and merging failed: {e}")
            raise
    
    def perform_analytics(self, merged_df):
        """
        Step 5: Perform correlation analysis and anomaly detection
        
        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: Performing Analytics")
        self.logger.info("=" * 60)
        
        try:
            # Perform comprehensive analytics
            analytics_results = analyze_weather_upi_correlations(merged_df)
            
            self.results['analytics'] = analytics_results
            
            # Log key findings
            correlations = analytics_results.get('correlations', {})
            anomaly_summary = analytics_results.get('anomaly_summary', {})
            
            self.logger.info("Correlation Analysis Results:")
            for pair, corr in correlations.items():
                if not pd.isna(corr):
                    self.logger.info(f"  {pair}: {corr:.4f}")
            
            self.logger.info("Anomaly Detection Results:")
            if 'transaction_outliers_count' in anomaly_summary:
                self.logger.info(f"  Transaction outliers: {anomaly_summary['transaction_outliers_count']}")
            if 'weather_outliers_count' in anomaly_summary:
                self.logger.info(f"  Weather outliers: {anomaly_summary['weather_outliers_count']}")
            
            # Save enhanced dataset with anomaly flags
            enhanced_df = analytics_results.get('enhanced_dataframe')
            if enhanced_df is not None and not enhanced_df.empty:
                enhanced_df.to_csv(config.ANALYTICS_FILE, index=False)
                self.logger.info(f"Analytics results saved to {config.ANALYTICS_FILE}")
            
            return analytics_results
            
        except Exception as e:
            self.logger.error(f"Analytics processing failed: {e}")
            raise
    
    def generate_summary_report(self):
        """
        Step 6: Generate final pipeline summary report
        
        Requirements: 1.5, 2.5, 4.5
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 6: Generating Summary Report")
        self.logger.info("=" * 60)
        
        try:
            summary_lines = []
            summary_lines.append("WEATHER-UPI CORRELATION PIPELINE SUMMARY")
            summary_lines.append("=" * 50)
            summary_lines.append(f"Pipeline executed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append("")
            
            # Data loading summary
            weather_count = len(self.results.get('weather_data', []))
            upi_count = len(self.results.get('upi_data', []))
            merged_count = len(self.results.get('merged_data', []))
            
            summary_lines.append("DATA LOADING:")
            summary_lines.append(f"  Weather records: {weather_count}")
            summary_lines.append(f"  UPI records: {upi_count}")
            summary_lines.append(f"  Merged records: {merged_count}")
            summary_lines.append("")
            
            # Validation summary
            weather_val = self.results.get('weather_validation')
            upi_val = self.results.get('upi_validation')
            
            summary_lines.append("VALIDATION RESULTS:")
            if weather_val:
                summary_lines.append(f"  Weather validation: {'PASSED' if weather_val.passed else 'FAILED'}")
                summary_lines.append(f"  Weather errors: {len(weather_val.errors)}")
                summary_lines.append(f"  Weather warnings: {len(weather_val.warnings)}")
            
            if upi_val:
                summary_lines.append(f"  UPI validation: {'PASSED' if upi_val.passed else 'FAILED'}")
                summary_lines.append(f"  UPI errors: {len(upi_val.errors)}")
                summary_lines.append(f"  UPI warnings: {len(upi_val.warnings)}")
            summary_lines.append("")
            
            # Analytics summary
            analytics = self.results.get('analytics', {})
            correlations = analytics.get('correlations', {})
            anomaly_summary = analytics.get('anomaly_summary', {})
            
            summary_lines.append("ANALYTICS RESULTS:")
            summary_lines.append(f"  Correlations computed: {len(correlations)}")
            
            if 'transaction_outliers_count' in anomaly_summary:
                summary_lines.append(f"  Transaction outliers: {anomaly_summary['transaction_outliers_count']}")
            if 'weather_outliers_count' in anomaly_summary:
                summary_lines.append(f"  Weather outliers: {anomaly_summary['weather_outliers_count']}")
            summary_lines.append("")
            
            # Output files
            summary_lines.append("OUTPUT FILES GENERATED:")
            summary_lines.append(f"  Merged dataset: {config.MERGED_DATA_FILE}")
            summary_lines.append(f"  Analytics results: {config.ANALYTICS_FILE}")
            summary_lines.append(f"  Validation report: {config.VALIDATION_REPORT_FILE}")
            summary_lines.append(f"  Pipeline log: {config.OUTPUT_DIR / 'pipeline.log'}")
            
            summary_report = "\n".join(summary_lines)
            
            # Save summary report
            summary_file = config.OUTPUT_DIR / "pipeline_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            
            # Log summary
            self.logger.info("Pipeline Summary:")
            self.logger.info(summary_report)
            self.logger.info(f"Summary report saved to {summary_file}")
            
            return summary_report
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            raise
    
    def run_pipeline(self):
        """
        Execute the complete pipeline with comprehensive error handling
        
        Requirements: 1.5, 2.5, 4.5
        """
        self.setup_logging()
        self.logger.info("Starting Weather-UPI Dashboard Pipeline")
        self.logger.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Fetch weather data
            weather_df = self.fetch_weather_data()
            
            # Step 2: Load UPI data
            upi_df = self.load_upi_data()
            
            # Step 3: Validate data
            self.validate_data(weather_df, upi_df)
            
            # Step 4: Transform and merge
            merged_df = self.transform_and_merge_data(weather_df, upi_df)
            
            # Step 5: Perform analytics
            self.perform_analytics(merged_df)
            
            # Step 6: Generate summary
            self.generate_summary_report()
            
            self.logger.info("=" * 60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            self.logger.info("Ready for dashboard visualization!")
            
            return True
            
        except Exception as e:
            self.logger.error("=" * 60)
            self.logger.error("PIPELINE EXECUTION FAILED")
            self.logger.error("=" * 60)
            self.logger.error(f"Error: {e}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            
            # Save error report
            error_file = config.OUTPUT_DIR / "pipeline_error.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Pipeline execution failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {e}\n\n")
                f.write("Full traceback:\n")
                f.write(traceback.format_exc())
            
            self.logger.error(f"Error details saved to {error_file}")
            return False

def main():
    """Main pipeline execution function."""
    pipeline = WeatherUPIPipeline()
    success = pipeline.run_pipeline()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()