#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

# Test if the issue is with streamlit decorators
import streamlit as st

print("Testing class definition...")

try:
    class WeatherUPIDashboard:
        """Test class"""
        
        def __init__(self):
            print("Dashboard initialized")
        
        @st.cache_data(ttl=3600)
        def cached_fetch_weather(self, lat: float, lon: float, start_date: str, end_date: str):
            """Test cached method"""
            return None
    
    print("✓ Class defined successfully")
    dashboard = WeatherUPIDashboard()
    print("✓ Class instantiated successfully")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()