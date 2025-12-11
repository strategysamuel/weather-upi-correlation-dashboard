#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

print("Testing dashboard imports step by step...")

try:
    import streamlit as st
    print("✓ streamlit")
    
    import pandas as pd
    print("✓ pandas")
    
    import numpy as np
    print("✓ numpy")
    
    import plotly.express as px
    print("✓ plotly.express")
    
    import plotly.graph_objects as go
    print("✓ plotly.graph_objects")
    
    from plotly.subplots import make_subplots
    print("✓ plotly.subplots")
    
    import seaborn as sns
    print("✓ seaborn")
    
    import matplotlib.pyplot as plt
    print("✓ matplotlib.pyplot")
    
    from datetime import datetime, date, timedelta
    print("✓ datetime")
    
    from pathlib import Path
    print("✓ pathlib")
    
    import sys
    print("✓ sys")
    
    import os
    print("✓ os")
    
    import logging
    print("✓ logging")
    
    from typing import Optional, Dict, Any
    print("✓ typing")
    
    # Now test project imports
    sys.path.insert(0, 'src')
    
    import config
    print("✓ config")
    
    from weather_api import get_weather_data, request_csv_fallback, PipelineError, FallbackPending, WeatherAPIClient
    print("✓ weather_api imports")
    
    from data_loader import load_upi_csv
    print("✓ data_loader")
    
    from data_transformer import DataTransformer
    print("✓ data_transformer")
    
    from analytics_engine import analyze_weather_upi_correlations
    print("✓ analytics_engine")
    
    import statsmodels.api as sm
    print("✓ statsmodels")
    
    import time
    print("✓ time")
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()