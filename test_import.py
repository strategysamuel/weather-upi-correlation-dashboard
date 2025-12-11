#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

try:
    print("Testing imports...")
    
    import streamlit as st
    print("✓ streamlit imported")
    
    import pandas as pd
    print("✓ pandas imported")
    
    import config
    print("✓ config imported")
    
    import weather_api
    print("✓ weather_api imported")
    
    import data_loader
    print("✓ data_loader imported")
    
    import data_transformer
    print("✓ data_transformer imported")
    
    import analytics_engine
    print("✓ analytics_engine imported")
    
    import statsmodels.api as sm
    print("✓ statsmodels imported")
    
    print("\nNow testing dashboard import...")
    import dashboard
    print("✓ dashboard module imported")
    
    print("Dashboard attributes:", [x for x in dir(dashboard) if not x.startswith('_')])
    
    if hasattr(dashboard, 'WeatherUPIDashboard'):
        print("✓ WeatherUPIDashboard class found")
    else:
        print("✗ WeatherUPIDashboard class NOT found")
        
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()