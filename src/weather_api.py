"""
MCP Weather API Module

This module handles fetching weather data from Open-Meteo API via Model Context Protocol
and provides fallback to local CSV data when API calls fail.
"""

import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mumbai coordinates
MUMBAI_LAT = 19.07
MUMBAI_LON = 72.88

class WeatherAPIClient:
    """MCP client for fetching weather data from Open-Meteo API"""
    
    def __init__(self, fallback_csv_path: str = "weather_mumbai_2024_11_synthetic.csv"):
        self.archive_url = "https://archive-api.open-meteo.com/v1/archive"
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"
        self.fallback_csv_path = fallback_csv_path
        
    def fetch_weather_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch weather data from Open-Meteo API with fallback to local CSV
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with weather data
        """
        try:
            # Set default date range if not provided
            if not start_date or not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            logger.info(f"Fetching weather data from API for dates {start_date} to {end_date}")
            
            # Try archive endpoint first for historical data
            api_data = self._call_archive_api_with_retry(start_date, end_date)
            if api_data is not None:
                logger.info("Using live weather API: archive")
                return self._parse_weather_response(api_data)
            
            # Try forecast endpoint as secondary attempt
            logger.info("Archive API failed, trying forecast endpoint")
            api_data = self._call_forecast_api_with_retry(start_date, end_date)
            if api_data is not None:
                logger.info("Using live weather API: forecast")
                return self._parse_weather_response(api_data)
            
            # Fall back to CSV
            logger.warning(f"Weather API failed, falling back to local CSV: {self.fallback_csv_path}")
            return self._fallback_to_csv()
                
        except Exception as e:
            logger.error(f"Error in fetch_weather_data: {e}")
            logger.warning(f"Weather API failed (exception), falling back to local CSV: {self.fallback_csv_path}")
            return self._fallback_to_csv()
    
    def _call_archive_api_with_retry(self, start_date: str, end_date: str) -> Optional[Dict[Any, Any]]:
        """
        Make HTTP request to Open-Meteo Archive API with retry logic
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            JSON response data or None if failed
        """
        params = {
            "latitude": MUMBAI_LAT,
            "longitude": MUMBAI_LON,
            "daily": "temperature_2m_max,temperature_2m_min,rain_sum",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Asia/Kolkata"
        }
        
        # Log the full request URL and params
        logger.info(f"Archive API request URL: {self.archive_url}")
        logger.info(f"Archive API request params: {params}")
        
        return self._make_request_with_retry(self.archive_url, params, "archive")
    
    def _call_forecast_api_with_retry(self, start_date: str, end_date: str) -> Optional[Dict[Any, Any]]:
        """
        Make HTTP request to Open-Meteo Forecast API with retry logic
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            JSON response data or None if failed
        """
        params = {
            "latitude": MUMBAI_LAT,
            "longitude": MUMBAI_LON,
            "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Asia/Kolkata"
        }
        
        # Log the full request URL and params
        logger.info(f"Forecast API request URL: {self.forecast_url}")
        logger.info(f"Forecast API request params: {params}")
        
        return self._make_request_with_retry(self.forecast_url, params, "forecast")
    
    def _make_request_with_retry(self, url: str, params: Dict, api_type: str) -> Optional[Dict[Any, Any]]:
        """
        Make HTTP request with retry logic and exponential backoff
        
        Args:
            url: API endpoint URL
            params: Request parameters
            api_type: Type of API (archive/forecast) for logging
            
        Returns:
            JSON response data or None if failed
        """
        max_retries = 3
        backoff_delays = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                
                # Log response details
                logger.debug(f"{api_type} API response status: {response.status_code}")
                logger.debug(f"{api_type} API response URL: {response.url}")
                logger.debug(f"{api_type} API response text (first 2000 chars): {str(response.text)[:2000]}")
                
                if response.status_code == 200:
                    logger.info(f"Successfully fetched data from {api_type} API")
                    return response.json()
                elif response.status_code == 400:
                    # Bad request - don't retry
                    logger.error(f"{api_type} API bad request (status 400): {response.text}")
                    return None
                elif response.status_code >= 500:
                    # Server error - retry
                    logger.warning(f"{api_type} API server error (status {response.status_code}), attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(backoff_delays[attempt])
                        continue
                else:
                    # Other client errors - don't retry
                    logger.error(f"{api_type} API request failed with status {response.status_code}: {response.text}")
                    return None
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"{api_type} API connection error, attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff_delays[attempt])
                    continue
            except requests.exceptions.Timeout as e:
                logger.warning(f"{api_type} API timeout, attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff_delays[attempt])
                    continue
            except requests.exceptions.RequestException as e:
                logger.error(f"{api_type} API request exception: {e}")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"{api_type} API JSON decode error: {e}")
                return None
        
        logger.error(f"{api_type} API failed after {max_retries} attempts")
        return None
    
    def _parse_weather_response(self, api_data: Dict[Any, Any]) -> pd.DataFrame:
        """
        Parse Open-Meteo API response into standardized DataFrame
        
        Args:
            api_data: JSON response from Open-Meteo API
            
        Returns:
            Standardized weather DataFrame
        """
        try:
            daily_data = api_data.get("daily", {})
            if not daily_data:
                logger.error("API response lacks 'daily' data section")
                raise ValueError("API response lacks 'daily' data section")
            
            dates = daily_data.get("time", [])
            if not dates:
                logger.error("API response lacks 'time' array in daily data")
                raise ValueError("API response lacks 'time' array in daily data")
            
            # Handle different API response formats
            # Archive API uses temperature_2m_max/min, forecast uses temperature_2m_mean
            temp_max = daily_data.get("temperature_2m_max", [])
            temp_min = daily_data.get("temperature_2m_min", [])
            temp_mean = daily_data.get("temperature_2m_mean", [])
            
            # Calculate average temperature
            if temp_max and temp_min and len(temp_max) == len(temp_min):
                # Archive API: compute average from max and min
                temperatures = [(tmax + tmin) / 2 for tmax, tmin in zip(temp_max, temp_min)]
                logger.debug("Using temperature_2m_max and temperature_2m_min to compute average")
            elif temp_mean:
                # Forecast API: use mean directly
                temperatures = temp_mean
                logger.debug("Using temperature_2m_mean directly")
            else:
                logger.error("No valid temperature data found in API response")
                raise ValueError("No valid temperature data found in API response")
            
            # Handle precipitation
            precipitation = daily_data.get("rain_sum", []) or daily_data.get("precipitation_sum", [])
            if not precipitation:
                logger.warning("No precipitation data found, using zeros")
                precipitation = [0.0] * len(dates)
            
            # Handle humidity (may not be available in archive API)
            humidity = daily_data.get("relative_humidity_2m_mean", [])
            if not humidity:
                logger.warning("No humidity data found, using default 60%")
                humidity = [60.0] * len(dates)
            
            # Validate data consistency
            if not (len(dates) == len(temperatures) == len(precipitation) == len(humidity)):
                logger.error(f"Data length mismatch: dates={len(dates)}, temp={len(temperatures)}, precip={len(precipitation)}, humidity={len(humidity)}")
                raise ValueError("Data arrays have inconsistent lengths")
            
            # Create DataFrame with standardized column names
            df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'city': 'Mumbai',
                'avg_temp_c': temperatures,
                'humidity_pct': humidity,
                'rain_mm': precipitation,
                'condition': ['API_Data'] * len(dates)  # Placeholder condition
            })
            
            logger.info(f"Parsed {len(df)} weather records from API response")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
            raise
    
    def _fallback_to_csv(self) -> pd.DataFrame:
        """
        Load weather data from local CSV file as fallback
        
        Returns:
            Weather DataFrame from local CSV
        """
        try:
            logger.info(f"Loading fallback weather data from {self.fallback_csv_path}")
            df = pd.read_csv(self.fallback_csv_path)
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"Loaded {len(df)} weather records from CSV fallback")
            return df
            
        except Exception as e:
            logger.error(f"Error loading fallback CSV: {e}")
            # Return empty DataFrame with correct schema if CSV also fails
            return pd.DataFrame(columns=['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm', 'condition'])

def get_weather_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Convenience function to fetch weather data
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Weather DataFrame
    """
    client = WeatherAPIClient()
    return client.fetch_weather_data(start_date, end_date)