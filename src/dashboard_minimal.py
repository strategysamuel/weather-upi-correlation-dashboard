"""
Minimal Dashboard Test
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
import sys
import logging
from typing import Optional, Dict, Any

# Add project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

print("Starting minimal dashboard...")

try:
    import config
    print("✓ config imported")
    
    from weather_api import WeatherAPIClient
    print("✓ WeatherAPIClient imported")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("✓ logging configured")
    
    # Mumbai coordinates
    MUMBAI_LAT = 19.07
    MUMBAI_LON = 72.88
    print("✓ constants defined")
    
    @st.cache_data(ttl=3600)
    def cached_fetch_weather(lat: float, lon: float, start_date: str, end_date: str) -> Optional[Dict[Any, Any]]:
        """Cached weather data fetching"""
        logger.info(f"Cache miss - fetching weather data for {start_date} to {end_date}")
        client = WeatherAPIClient()
        return client._call_archive_api_with_retry(start_date, end_date)
    
    print("✓ cached function defined")
    
    class WeatherUPIDashboard:
        """Main dashboard class for Weather-UPI correlation analysis"""
        
        def __init__(self):
            """Initialize the dashboard"""
            print("Dashboard __init__ called")
            self.setup_page_config()
            self.data = None
        
        def setup_page_config(self):
            """Configure Streamlit page settings"""
            try:
                st.set_page_config(
                    page_title="Weather-UPI Dashboard",
                    page_icon="🌦️",
                    layout="wide",
                    initial_sidebar_state="expanded"
                )
                print("✓ page config set")
            except Exception as e:
                print(f"Page config error (expected in non-streamlit context): {e}")
        
        def get_default_date_range(self):
            """Get default date range (last 30 days)"""
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            return start_date, end_date
    
    print("✓ WeatherUPIDashboard class defined")
    
    # Test instantiation
    dashboard = WeatherUPIDashboard()
    print("✓ Dashboard instantiated successfully")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("Minimal dashboard test complete")