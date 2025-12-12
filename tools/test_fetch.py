# tools/test_fetch.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from weather_api import fetch_open_meteo

df = fetch_open_meteo()
print("rows:", None if df is None else len(df))
if df is not None:
    print(df.head().to_string(index=False))