"""
Streamlit Dashboard Module

This module provides an interactive web dashboard for visualizing weather-UPI
correlation analysis results using Streamlit with live-first weather data fetch
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from pathlib import Path
import sys
import os
import logging
from typing import Optional, Dict, Any

# Add project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import config
from weather_api import get_weather_data, request_csv_fallback, PipelineError, FallbackPending, WeatherAPIClient
from data_loader import load_upi_csv
from data_transformer import DataTransformer
from analytics_engine import analyze_weather_upi_correlations
import statsmodels.api as sm
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mumbai coordinates
MUMBAI_LAT = 19.07
MUMBAI_LON = 72.88

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

class WeatherUPIDashboard:
    """Main dashboard class for Weather-UPI correlation analysis"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.setup_page_config()
        self.data = None
        self.analytics_data = None
        
        # Initialize session state (only if in Streamlit context)
        self.init_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        try:
            st.set_page_config(
                page_title=config.DASHBOARD_TITLE,
                page_icon="🌦️",
                layout="wide",
                initial_sidebar_state="expanded"
            )
        except Exception as e:
            # This is expected when not running in Streamlit context (e.g., during testing)
            logger.debug(f"Page config not set (not in Streamlit context): {e}")
    
    def init_session_state(self):
        """Initialize Streamlit session state"""
        try:
            if "weather_df" not in st.session_state:
                st.session_state["weather_df"] = None
            if "weather_source" not in st.session_state:
                st.session_state["weather_source"] = None
            if "last_fetch_params" not in st.session_state:
                st.session_state["last_fetch_params"] = None
        except Exception as e:
            # Not in Streamlit context (e.g., during testing)
            logger.debug(f"Session state not initialized (not in Streamlit context): {e}")
    
    def get_default_date_range(self):
        """Get default date range (last 30 days)"""
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        return start_date, end_date
    
    def validate_date_range(self, start_date, end_date):
        """Validate date range and return error message if invalid"""
        today = date.today()
        
        # Check for future dates
        if start_date > today or end_date > today:
            return "❌ Future dates are not allowed. Please select dates up to today."
        
        # Check if end < start
        if end_date < start_date:
            return "❌ End date must be after start date."
        
        # Check if range > 90 days
        if (end_date - start_date).days > 90:
            return "❌ Date range cannot exceed 90 days. Please select a shorter range."
        
        return None
    
    def fetch_and_store_weather(self, lat: float, lon: float, start_date: str, end_date: str, 
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
                df = request_csv_fallback(start_date, end_date)
                
                # Store in session state
                try:
                    st.session_state["weather_df"] = df
                    st.session_state["weather_source"] = "csv"
                except:
                    # Not in Streamlit context
                    pass
                
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
                try:
                    st.session_state["weather_df"] = df
                    st.session_state["weather_source"] = "api"
                except:
                    # Not in Streamlit context
                    pass
                
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
                df = request_csv_fallback(start_date, end_date)
                
                # Store in session state
                try:
                    st.session_state["weather_df"] = df
                    st.session_state["weather_source"] = "csv"
                except:
                    # Not in Streamlit context
                    pass
                
                logger.info(f"Auto fallback success: {len(df)} records loaded")
                return True
            except Exception as e:
                logger.error(f"CSV fallback failed: {e}")
                return False
        else:
            # Interactive fallback required
            logger.info("API failed, interactive fallback required")
            return False
    
    def run(self):
        """Main dashboard execution"""
        st.title("🌦️ Weather-UPI Correlation Dashboard")
        st.markdown("**Analyze correlations between weather patterns and UPI transaction data in Mumbai**")
        
        st.info("👆 Please select a date range and click 'Fetch data' to begin analysis.")


def main():
    """Main function to run the dashboard"""
    dashboard = WeatherUPIDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()