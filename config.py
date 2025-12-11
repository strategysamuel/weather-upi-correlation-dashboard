"""
Configuration settings for the Weather-UPI Dashboard
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT

# Data file paths
UPI_DATA_FILE = DATA_DIR / "upi_transactions_india_2024_11_synthetic.csv"
WEATHER_DATA_FILE = DATA_DIR / "weather_mumbai_2024_11_synthetic.csv"

# Output file paths
MERGED_DATA_FILE = OUTPUT_DIR / "merged_weather_upi_data.csv"
ANALYTICS_FILE = OUTPUT_DIR / "analytics.csv"
VALIDATION_REPORT_FILE = OUTPUT_DIR / "validation_report.txt"

# Mumbai coordinates for weather API
MUMBAI_LAT = 19.07
MUMBAI_LON = 72.88

# Open-Meteo API configuration
WEATHER_API_BASE_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_API_PARAMS = {
    "latitude": MUMBAI_LAT,
    "longitude": MUMBAI_LON,
    "daily": ["temperature_2m_mean", "precipitation_sum", "relative_humidity_2m_mean"],
    "timezone": "Asia/Kolkata"
}

# Analysis parameters
OUTLIER_THRESHOLD = 2.0  # Standard deviations for outlier detection
MIN_CORRELATION_SIGNIFICANCE = 0.05  # p-value threshold

# Dashboard configuration
DASHBOARD_TITLE = "Weather-UPI Correlation Dashboard"
DASHBOARD_PORT = 8501