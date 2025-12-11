#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

print("Starting dashboard import debug...")

try:
    # Import the dashboard module and execute it step by step
    import importlib.util
    spec = importlib.util.spec_from_file_location("dashboard", "src/dashboard.py")
    dashboard_module = importlib.util.module_from_spec(spec)
    
    print("Module spec created, executing...")
    spec.loader.exec_module(dashboard_module)
    print("Module executed successfully")
    
    print("Available attributes:", [x for x in dir(dashboard_module) if not x.startswith('_')])
    
    if hasattr(dashboard_module, 'WeatherUPIDashboard'):
        print("✓ WeatherUPIDashboard found!")
    else:
        print("✗ WeatherUPIDashboard not found")
        
except Exception as e:
    print(f"✗ Error during module execution: {e}")
    import traceback
    traceback.print_exc()