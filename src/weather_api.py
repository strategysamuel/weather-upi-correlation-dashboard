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
import sys
from pathlib import Path
import streamlit as st

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Custom exception for pipeline errors"""
    pass

class FallbackPending(Exception):
    """Exception indicating fallback is pending user approval"""
    pass

# Mumbai coordinates
MUMBAI_LAT = 19.07
MUMBAI_LON = 72.88

class WeatherAPIClient:
    """MCP client for fetching weather data from Open-Meteo API"""
    
    def __init__(self, fallback_csv_path: str = None):
        self.archive_url = "https://archive-api.open-meteo.com/v1/archive"
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"
        self.fallback_csv_path = fallback_csv_path or str(config.WEATHER_DATA_FILE)
        self.max_retries = config.LIVE_FETCH_RETRY_COUNT
        self.retry_delay = config.LIVE_FETCH_RETRY_DELAY_SEC
        
    def fetch_weather_data(self, start_date: str = None, end_date: str = None, 
                          use_csv_fallback: bool = None, interactive: bool = None) -> pd.DataFrame:
        """
        Fetch weather data with live-first approach and controlled fallback
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_csv_fallback: Override for ALLOW_CSV_FALLBACK config
            interactive: Override for INTERACTIVE_FALLBACK_PROMPT config
            
        Returns:
            DataFrame with weather data and source information
            
        Raises:
            PipelineError: If API fails and fallback is not allowed
            FallbackPending: If API fails and interactive approval is needed
        """
        # Set default date range if not provided
        if not start_date or not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Use config defaults if not overridden
        if use_csv_fallback is None:
            use_csv_fallback = config.ALLOW_CSV_FALLBACK
        if interactive is None:
            interactive = config.INTERACTIVE_FALLBACK_PROMPT
        
        # If live weather is disabled, go straight to CSV
        if not config.USE_LIVE_WEATHER:
            logger.info("Live weather disabled in config, using CSV fallback")
            return self._load_csv_fallback()
        
        # Attempt live API fetch with retries
        logger.info("Attempting Live Weather API...")
        logger.info(f"Fetching weather data from API for dates {start_date} to {end_date}")
        
        for attempt in range(config.LIVE_FETCH_RETRY_COUNT):
            try:
                # Try archive endpoint first
                api_data = self._call_archive_api_with_retry(start_date, end_date)
                if api_data is not None:
                    logger.info("Live API succeeded")
                    df = self._parse_weather_response(api_data)
                    # Mark as API data
                    df['condition'] = 'API_Data'
                    df['source'] = 'api'
                    return df
                
                # Try forecast endpoint as backup
                logger.info("Archive API failed, trying forecast endpoint")
                api_data = self._call_forecast_api_with_retry(start_date, end_date)
                if api_data is not None:
                    logger.info("Live API succeeded")
                    df = self._parse_weather_response(api_data)
                    # Mark as API data
                    df['condition'] = 'API_Data'
                    df['source'] = 'api'
                    return df
                
                # Both endpoints failed, retry if attempts remaining
                if attempt < config.LIVE_FETCH_RETRY_COUNT - 1:
                    logger.warning(f"Live API attempt {attempt + 1} failed, retrying in {config.LIVE_FETCH_RETRY_DELAY_SEC}s...")
                    time.sleep(config.LIVE_FETCH_RETRY_DELAY_SEC)
                else:
                    logger.error(f"Live API failed after {config.LIVE_FETCH_RETRY_COUNT} attempts")
                    break
                    
            except Exception as e:
                logger.error(f"Live API attempt {attempt + 1} failed with exception: {e}")
                if attempt < config.LIVE_FETCH_RETRY_COUNT - 1:
                    logger.warning(f"Retrying in {config.LIVE_FETCH_RETRY_DELAY_SEC}s...")
                    time.sleep(config.LIVE_FETCH_RETRY_DELAY_SEC)
                else:
                    logger.error(f"Live API failed after {config.LIVE_FETCH_RETRY_COUNT} attempts")
                    break
        
        # All API attempts failed, handle fallback
        return self._handle_api_failure(use_csv_fallback, interactive)
    
    def _handle_api_failure(self, use_csv_fallback: bool, interactive: bool) -> pd.DataFrame:
        """
        Handle API failure with user approval logic
        
        Args:
            use_csv_fallback: Whether to allow automatic CSV fallback
            interactive: Whether to prompt user for approval
            
        Returns:
            DataFrame with weather data from CSV
            
        Raises:
            PipelineError: If fallback is not approved
            FallbackPending: If interactive approval is needed
        """
        if not use_csv_fallback:
            # Strict mode - no fallback allowed
            logger.error("Live API failed and CSV fallback is disabled (strict mode)")
            raise PipelineError("Live API failed and CSV fallback is disabled")
        
        if interactive:
            # Interactive mode - raise FallbackPending for UI to handle
            logger.warning("Live API failed, interactive fallback approval required")
            raise FallbackPending("Live API failed, user approval required for CSV fallback")
        
        else:
            # Silent fallback allowed
            logger.warning("Live API failed, using automatic CSV fallback")
            return self._load_csv_fallback()
    

    
    def _handle_api_failure(self, use_csv_fallback: bool, interactive: bool) -> pd.DataFrame:
        """
        Handle API failure with user approval logic
        
        Args:
            use_csv_fallback: Whether to allow automatic CSV fallback
            interactive: Whether to prompt user for approval
            
        Returns:
            DataFrame with weather data from CSV
            
        Raises:
            PipelineError: If fallback is not approved
        """
        if not use_csv_fallback:
            # Strict mode - no fallback allowed
            logger.error("Live API failed and CSV fallback is disabled (strict mode)")
            raise PipelineError("Live API failed and CSV fallback is disabled")
        
        if interactive:
            # Interactive mode - raise FallbackPending for UI to handle
            logger.warning("Live API failed, interactive fallback approval required")
            raise FallbackPending("Live API failed, user approval required for CSV fallback")
        
        else:
            # Silent fallback allowed
            logger.warning("Live API failed, using automatic CSV fallback")
            return self._load_csv_fallback()
    
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
        max_retries = self.max_retries
        backoff_delays = [self.retry_delay * (2**i) for i in range(max_retries)]  # Exponential backoff
        
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
            
            # Add condition column if not present
            if 'condition' not in df.columns:
                df['condition'] = df['rain_mm'].apply(lambda x: 'Rain' if x > 0 else 'Clear')
            
            # Mark as CSV fallback
            df['condition'] = 'CSV_Fallback'
            df['source'] = 'csv'
            
            logger.info(f"Loaded {len(df)} weather records from CSV fallback")
            return df
            
        except Exception as e:
            logger.error(f"Error loading fallback CSV: {e}")
            # Return empty DataFrame with correct schema if CSV also fails
            return pd.DataFrame(columns=['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm', 'condition', 'source'])
    
    def _load_csv_fallback(self) -> pd.DataFrame:
        """
        Load weather data from CSV fallback file
        
        Returns:
            DataFrame with weather data marked as CSV fallback
        """
        try:
            logger.info(f"Loading CSV fallback from {config.WEATHER_DATA_FILE}")
            df = pd.read_csv(config.WEATHER_DATA_FILE)
            
            # Ensure required columns exist
            required_columns = ['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV fallback")
            
            # Add condition column if not present
            if 'condition' not in df.columns:
                df['condition'] = df['rain_mm'].apply(lambda x: 'Rain' if x > 0 else 'Clear')
            
            # Mark as CSV fallback
            df['condition'] = 'CSV_Fallback'
            df['source'] = 'csv'
            
            logger.info(f"Successfully loaded {len(df)} records from CSV fallback")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV fallback: {e}")
            raise PipelineError(f"Both live API and CSV fallback failed: {e}")
    
    def request_csv_fallback(self) -> pd.DataFrame:
        """
        Streamlit-compatible function to request CSV fallback after user approval
        
        Returns:
            DataFrame with weather data from CSV
        """
        logger.info("User approved CSV fallback")
        return self._load_csv_fallback()

def get_weather_data(start_date: str = None, end_date: str = None, 
                    use_csv_fallback: bool = None, interactive: bool = None) -> pd.DataFrame:
    """
    Convenience function to fetch weather data with live-first approach
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        use_csv_fallback: Override for ALLOW_CSV_FALLBACK config
        interactive: Override for INTERACTIVE_FALLBACK_PROMPT config
        
    Returns:
        Weather DataFrame with source information
        
    Raises:
        PipelineError: If API fails and fallback is not approved
        FallbackPending: If API fails and interactive approval is needed
    """
    client = WeatherAPIClient()
    return client.fetch_weather_data(start_date, end_date, use_csv_fallback, interactive)
def request_csv_fallback(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Request CSV fallback after user approval (Streamlit-compatible)
    
    Args:
        start_date: Start date in YYYY-MM-DD format (unused for CSV)
        end_date: End date in YYYY-MM-DD format (unused for CSV)
        
    Returns:
        Weather DataFrame from CSV fallback
    """
    client = WeatherAPIClient()
    return client.request_csv_fallback()

# Cached weather data fetching function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_fetch_weather(lat: float, lon: float, start_date: str, end_date: str) -> Optional[Dict[Any, Any]]:
    """
    Cached weather data fetching with 1-hour TTL
    
    Args:
        lat: Latitude
        lon: Longitude  
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        JSON response data or None if failed
    """
    logger.info(f"Cache miss - fetching weather data for {start_date} to {end_date}")
    client = WeatherAPIClient()
    return client._call_archive_api_with_retry(start_date, end_date)

def fetch_and_store_weather(lat: float, lon: float, start_date: str, end_date: str, 
                           try_live: bool = True, auto_fallback: bool = False) -> bool:
    """
    Fetch weather data and store in session state
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        try_live: Whether to attempt live API first
        auto_fallback: Whether to automatically fallback to CSV
        
    Returns:
        True if data was successfully stored, False if API failed and auto_fallback=False
    """
    logger.info(f"Fetch starts: lat={lat}, lon={lon}, start={start_date}, end={end_date}, live={try_live}")
    
    if not try_live:
        # Skip live API, go straight to CSV
        logger.info("Live API disabled, loading CSV fallback")
        try:
            client = WeatherAPIClient()
            df = client._load_csv_fallback()
            
            # Store in session state
            st.session_state["weather_df"] = df
            st.session_state["weather_source"] = "csv"
            
            logger.info(f"CSV fallback loaded: {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"CSV fallback failed: {e}")
            return False
    
    # Try live API first
    try:
        # Use cached fetch
        api_data = cached_fetch_weather(lat, lon, start_date, end_date)
        
        if api_data is not None:
            # Parse API response
            client = WeatherAPIClient()
            df = client._parse_weather_response(api_data)
            df['source'] = 'api'
            df['condition'] = 'API_Data'
            
            # Store in session state
            st.session_state["weather_df"] = df
            st.session_state["weather_source"] = "api"
            
            logger.info(f"Live API success: {len(df)} records loaded")
            return True
        else:
            logger.warning("Live API failed")
            
    except Exception as e:
        logger.error(f"Live API exception: {e}")
    
    # API failed - handle fallback
    if auto_fallback:
        logger.info("Auto fallback enabled, loading CSV")
        try:
            client = WeatherAPIClient()
            df = client._load_csv_fallback()
            
            # Store in session state
            st.session_state["weather_df"] = df
            st.session_state["weather_source"] = "csv"
            
            logger.info(f"Auto fallback success: {len(df)} records loaded")
            return True
        except Exception as e:
            logger.error(f"CSV fallback failed: {e}")
            return False
    else:
        # Interactive fallback required
        logger.info("API failed, interactive fallback required")
        return False
# Cached weather data fetching function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_fetch_weather(lat: float, lon: float, start_date: str, end_date: str) -> Optional[Dict[Any, Any]]:
    """
    Cached weather data fetching with 1-hour TTL
    
    Args:
        lat: Latitude
        lon: Longitude  
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        JSON response data or None if failed
    """
    logger.info(f"Cache miss - fetching weather data for {start_date} to {end_date}")
    client = WeatherAPIClient()
    return client._call_archive_api_with_retry(start_date, end_date)

def fetch_and_store_weather(lat: float, lon: float, start_date: str, end_date: str, 
                           try_live: bool = True, auto_fallback: bool = False) -> bool:
    """
    Fetch weather data and store in session state
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        try_live: Whether to attempt live API first
        auto_fallback: Whether to automatically fallback to CSV
        
    Returns:
        True if data was successfully stored, False if API failed and auto_fallback=False
    """
    logger.info(f"Fetch starts: lat={lat}, lon={lon}, start={start_date}, end={end_date}, live={try_live}")
    
    if not try_live:
        # Skip live API, go straight to CSV
        logger.info("Live API disabled, loading CSV fallback")
        try:
            client = WeatherAPIClient()
            df = client._load_csv_fallback()
            
            # Store in session state
            st.session_state["weather_df"] = df
            st.session_state["weather_source"] = "csv"
            
            logger.info(f"CSV fallback loaded: {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"CSV fallback failed: {e}")
            return False
    
    # Try live API first
    try:
        # Use cached fetch
        api_data = cached_fetch_weather(lat, lon, start_date, end_date)
        
        if api_data is not None:
            # Parse API response
            client = WeatherAPIClient()
            df = client._parse_weather_response(api_data)
            df['source'] = 'api'
            df['condition'] = 'API_Data'
            
            # Store in session state
            st.session_state["weather_df"] = df
            st.session_state["weather_source"] = "api"
            
            logger.info(f"Live API success: {len(df)} records loaded")
            return True
        else:
            logger.warning("Live API failed")
            
    except Exception as e:
        logger.error(f"Live API exception: {e}")
    
    # API failed - handle fallback
    if auto_fallback:
        logger.info("Auto fallback enabled, loading CSV")
        try:
            client = WeatherAPIClient()
            df = client._load_csv_fallback()
            
            # Store in session state
            st.session_state["weather_df"] = df
            st.session_state["weather_source"] = "csv"
            
            logger.info(f"Auto fallback success: {len(df)} records loaded")
            return True
        except Exception as e:
            logger.error(f"CSV fallback failed: {e}")
            return False
    else:
        # Interactive fallback required
        logger.info("API failed, interactive fallback required")
        return False