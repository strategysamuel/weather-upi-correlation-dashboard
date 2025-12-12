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
from data_loader import load_upi_csv, load_weather_csv, load_upi_data
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
    
    def render_sidebar_controls(self):
        """Render sidebar controls for data fetching"""
        st.sidebar.header("🌦️ Weather Data Controls")
        
        # Default date range (last 30 days)
        default_start, default_end = self.get_default_date_range()
        
        # Date range selector
        st.sidebar.subheader("📅 Date Range")
        start_date = st.sidebar.date_input(
            "Start Date",
            value=default_start,
            max_value=date.today(),
            key="start_date"
        )
        
        end_date = st.sidebar.date_input(
            "End Date", 
            value=default_end,
            max_value=date.today(),
            key="end_date"
        )
        
        # Validate date range
        date_error = self.validate_date_range(start_date, end_date)
        if date_error:
            st.sidebar.error(date_error)
            return None, None, None, None, None, None
        
        # API options
        st.sidebar.subheader("🔧 Fetch Options")
        try_live = st.sidebar.checkbox(
            "Try live API first",
            value=True,
            help="Attempt to fetch live weather data from Open-Meteo API"
        )
        
        auto_fallback = st.sidebar.checkbox(
            "Auto fallback to CSV (no prompt)",
            value=False,
            help="Automatically use CSV fallback if live API fails (no user prompt)"
        )
        
        # Refresh interval
        refresh_interval = st.sidebar.selectbox(
            "Auto-refresh interval",
            options=[None, "5 min", "15 min", "60 min"],
            index=0,
            help="Automatically refresh data at selected interval"
        )
        
        # Fetch button
        fetch_clicked = st.sidebar.button(
            "🔄 Fetch data for selected range",
            type="primary",
            help="Fetch weather data for the selected date range"
        )
        
        return start_date, end_date, try_live, auto_fallback, refresh_interval, fetch_clicked
    
    def render_sidebar_controls(self):
        """Render sidebar controls for date selection and options"""
        st.sidebar.header("🎛️ Controls")
        
        # Default date range (last 30 days)
        default_start, default_end = self.get_default_date_range()
        
        # Date range selector
        st.sidebar.subheader("📅 Date Range")
        start_date = st.sidebar.date_input(
            "Start Date",
            value=default_start,
            max_value=date.today(),
            key="start_date"
        )
        
        end_date = st.sidebar.date_input(
            "End Date", 
            value=default_end,
            max_value=date.today(),
            key="end_date"
        )
        
        # Validate date range
        date_error = self.validate_date_range(start_date, end_date)
        if date_error:
            st.sidebar.error(date_error)
            return None, None, None, None, None, None
        
        # API options
        st.sidebar.subheader("🔧 Fetch Options")
        try_live = st.sidebar.checkbox(
            "Try live API first",
            value=True,
            help="Attempt to fetch live weather data from Open-Meteo API"
        )
        
        auto_fallback = st.sidebar.checkbox(
            "Auto fallback to CSV (no prompt)",
            value=False,
            help="Automatically use CSV fallback if live API fails (no user prompt)"
        )
        
        # Refresh interval
        refresh_interval = st.sidebar.selectbox(
            "Auto-refresh interval",
            options=[None, "5 min", "15 min", "60 min"],
            index=0,
            help="Automatically refresh data at selected interval"
        )
        
        # Fetch button
        fetch_clicked = st.sidebar.button(
            "🔄 Fetch data for selected range",
            type="primary",
            help="Fetch weather data for the selected date range"
        )
        
        return start_date, end_date, try_live, auto_fallback, refresh_interval, fetch_clicked
    
    def render_data_source_badge(self):
        """Render data source badge"""
        if st.session_state["weather_source"] == "api":
            st.success("📡 **Data source: LIVE (API)** - Fresh weather data from Open-Meteo")
        elif st.session_state["weather_source"] == "csv":
            st.warning("📁 **Data source: CSV (fallback)** - Using local weather data")
        else:
            st.info("❓ **Data source: UNKNOWN** - No weather data loaded")
    
    def cached_fetch_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> Optional[Dict[Any, Any]]:
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
        return cached_fetch_weather(lat, lon, start_date, end_date)
    
    def handle_interactive_fallback(self, start_date: str, end_date: str):
        """Handle interactive fallback when API fails"""
        st.error("🚨 **Live weather API failed**")
        st.warning("The live weather API is currently unavailable. Would you like to use fallback CSV data instead?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📁 Load CSV fallback data", type="primary"):
                try:
                    df = request_csv_fallback(start_date, end_date)
                    
                    # Store in session state
                    st.session_state["weather_df"] = df
                    st.session_state["weather_source"] = "csv"
                    
                    st.success(f"✅ CSV fallback loaded: {len(df)} records")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ CSV fallback failed: {e}")
        
        with col2:
            if st.button("❌ Cancel"):
                st.info("Operation cancelled. Try adjusting the date range or check your internet connection.")
    
    def generate_automated_insights(self, data):
        """Generate automated insights from correlation analysis"""
        if data is None or data.empty:
            return ["No data available for analysis."]
        
        if len(data) < 3:
            return ["Insufficient data for meaningful analysis. Need at least 3 records."]
        
        insights = []
        
        try:
            # Calculate correlations
            numeric_cols = ['avg_temp_c', 'humidity_pct', 'rain_mm', 'total_upi_txn_count', 'avg_txn_value_inr']
            available_cols = [col for col in numeric_cols if col in data.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = data[available_cols].corr()
                
                # Find strongest correlations
                for i, col1 in enumerate(available_cols):
                    for j, col2 in enumerate(available_cols[i+1:], i+1):
                        corr_val = corr_matrix.loc[col1, col2]
                        if not pd.isna(corr_val):
                            strength = "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.3 else "weak"
                            direction = "positive" if corr_val > 0 else "negative"
                            insights.append(f"{strength.title()} {direction} correlation ({corr_val:.3f}) between {col1} and {col2}")
            
            # Check for outliers
            if 'weather_outlier' in data.columns:
                weather_outliers = data['weather_outlier'].sum()
                if weather_outliers > 0:
                    insights.append(f"Found {weather_outliers} weather outliers in the dataset")
            
            if 'txn_volume_outlier' in data.columns:
                txn_outliers = data['txn_volume_outlier'].sum()
                if txn_outliers > 0:
                    insights.append(f"Found {txn_outliers} transaction volume outliers in the dataset")
            
            # Weekly patterns
            if 'date' in data.columns and len(data) >= 7:
                data_copy = data.copy()
                data_copy['weekday'] = pd.to_datetime(data_copy['date']).dt.day_name()
                if 'total_upi_txn_count' in data_copy.columns:
                    weekly_avg = data_copy.groupby('weekday')['total_upi_txn_count'].mean()
                    if len(weekly_avg) > 1:
                        peak_day = weekly_avg.idxmax()
                        low_day = weekly_avg.idxmin()
                        insights.append(f"Peak transaction day: {peak_day}, Lowest: {low_day}")
            
        except Exception as e:
            insights.append(f"Error generating insights: {str(e)}")
        
        return insights if insights else ["No significant patterns detected in the data."]
    
    def get_data_source_status(self):
        """Get data source status from merged data file"""
        try:
            if config.MERGED_DATA_FILE.exists():
                df = pd.read_csv(config.MERGED_DATA_FILE)
                if 'condition' in df.columns:
                    if df['condition'].str.contains('API_Data').any():
                        return "LIVE (API)"
                    else:
                        return "CSV (fallback)"
                elif 'source' in df.columns:
                    if df['source'].str.contains('api').any():
                        return "LIVE (API)"
                    else:
                        return "CSV (fallback)"
                else:
                    return "CSV (fallback)"  # Default to CSV if no source info
            else:
                return "Unknown"
        except Exception:
            return "Unknown"
    
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
            api_data = self.cached_fetch_weather(lat, lon, start_date, end_date)
            
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
                df = request_csv_fallback(start_date, end_date)
                
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
    
    def get_default_date_range(self):
        """Get default date range (last 30 days)"""
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        return start_date, end_date
    
    def validate_date_range(self, start_date: date, end_date: date):
        """Validate date range"""
        today = date.today()
        
        # Check for future dates
        if start_date > today or end_date > today:
            st.error("❌ Future dates are not allowed. Please select dates up to today.")
            return False
        
        # Check if end date is before start date
        if end_date < start_date:
            st.error("❌ End date must be after start date.")
            return False
        
        # Check if range is too large (max 90 days)
        if (end_date - start_date).days > 90:
            st.error("❌ Date range cannot exceed 90 days. Please select a shorter range.")
            return False
        
        return True
    
    def render_sidebar_controls(self):
        """Render sidebar controls for date selection and options"""
        from datetime import date, timedelta
        # defaults: last 30 days
        today = date.today() - timedelta(days=1)
        default_start = today - timedelta(days=29)
        
        st.sidebar.header("Controls")
        start_date = st.sidebar.date_input("Start Date", value=default_start, max_value=today)
        end_date = st.sidebar.date_input("End Date", value=today, min_value=start_date, max_value=today)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Data Source Options")
        try_live = st.sidebar.checkbox("Try live API first", value=True)
        auto_fallback = st.sidebar.checkbox("Auto fallback to CSV (no prompt)", value=False)
        upi_simulator = st.sidebar.checkbox("UPI Live Simulator mode", value=True)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Auto Refresh")
        refresh_interval = st.sidebar.selectbox("Refresh Interval", ["None", "5 min", "15 min", "60 min"])
        
        fetch_btn = st.sidebar.button("🚀 Fetch data for selected range")
        
        return start_date, end_date, try_live, auto_fallback, upi_simulator, refresh_interval, fetch_btn
    
    def render_data_source_badge(self):
        """Render data source badges for both weather and UPI data"""
        weather_source = st.session_state.get("weather_source", "unknown")
        
        # Weather source badge
        if weather_source == "api":
            st.success("📡 **Weather Source:** LIVE API")
        elif weather_source == "csv":
            st.warning("📁 **Weather Source:** CSV (fallback)")
        else:
            st.info("❓ No data loaded yet. Click 'Fetch data for selected range' in the sidebar.")
        
        # UPI source badge if data is loaded
        weather_df = st.session_state.get("weather_df")
        if weather_df is not None and len(weather_df) > 0 and 'source' in weather_df.columns:
            upi_source = weather_df['source'].iloc[0] if len(weather_df) > 0 else 'unknown'
            if upi_source == 'simulated':
                st.info("🎲 **UPI Source:** SIMULATED")
            elif upi_source == 'csv':
                st.warning("📁 **UPI Source:** CSV")
        else:
            st.info("❓ **Data source: UNKNOWN**")
    
    def handle_interactive_fallback(self, start_date: str, end_date: str):
        """Handle interactive fallback when API fails"""
        st.error("🚨 **Live weather API failed**")
        st.warning("The live weather API is currently unavailable. Would you like to use fallback CSV data instead?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📁 Load CSV fallback data", type="primary"):
                try:
                    df = request_csv_fallback(start_date, end_date)
                    
                    # Store in session state
                    st.session_state["weather_df"] = df
                    st.session_state["weather_source"] = "csv"
                    
                    st.success(f"✅ CSV fallback loaded: {len(df)} records")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ CSV fallback failed: {e}")
        
        with col2:
            if st.button("❌ Cancel"):
                st.info("Operation cancelled. Try adjusting the date range or check your internet connection.")
    
    def load_and_merge_data(self):
        """Load UPI data and merge with weather data from session state"""
        if st.session_state["weather_df"] is None:
            return None, None
        
        try:
            # Get date range from weather data
            weather_df = st.session_state["weather_df"]
            start_date = weather_df['date'].min().strftime('%Y-%m-%d')
            end_date = weather_df['date'].max().strftime('%Y-%m-%d')
            
            # Load UPI data using simulator for the same date range
            upi_df = load_upi_data(start_date, end_date)
            
            # Transform and merge
            transformer = DataTransformer()
            merged_df = transformer.transform_and_merge(weather_df, upi_df)
            
            # Perform analytics
            analytics_results = analyze_weather_upi_correlations(merged_df)
            enhanced_df = analytics_results.get('enhanced_dataframe')
            
            # Save merged data with source column
            merged_df.to_csv(config.MERGED_DATA_FILE, index=False)
            if enhanced_df is not None:
                enhanced_df.to_csv(config.ANALYTICS_FILE, index=False)
            
            return merged_df, analytics_results
            
        except Exception as e:
            st.error(f"❌ Error processing data: {e}")
            return None, None
    
    def render_data_preview(self, merged_df):
        """Render data preview section"""
        if merged_df is None or merged_df.empty:
            st.info("No data available. Please fetch weather data first.")
            return
        
        st.subheader("📊 Data Preview")
        
        # Show data source labels
        weather_source = st.session_state.get("weather_source", "unknown")
        if 'source' in merged_df.columns:
            upi_source = merged_df['source'].iloc[0] if len(merged_df) > 0 else "unknown"
        else:
            upi_source = "unknown"
        
        st.markdown(f"**Weather Source:** {weather_source.upper()}")
        st.markdown(f"**UPI Source:** {upi_source.upper()}")
        
        # Show data summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(merged_df))
        with col2:
            st.metric("Date Range", f"{len(merged_df)} days")
        with col3:
            st.metric("Weather Source", weather_source.upper())
        with col4:
            st.metric("UPI Source", upi_source.upper())
        
        # Show data with pagination options
        st.subheader("📋 Data Table")
        
        # Add display options
        col1, col2 = st.columns([1, 3])
        with col1:
            show_all = st.checkbox("Show all records", value=False)
            if not show_all:
                num_rows = st.selectbox("Rows to display", [10, 20, 50], index=0)
        
        # Display data based on selection
        if show_all:
            st.dataframe(merged_df, use_container_width=True)
            st.info(f"Showing all {len(merged_df)} records")
        else:
            st.dataframe(merged_df.head(num_rows), use_container_width=True)
            st.info(f"Showing first {min(num_rows, len(merged_df))} of {len(merged_df)} total records")
    
    def generate_automated_insights(self, data):
        """Generate automated insights from correlation analysis"""
        if data is None or data.empty:
            return ["No data available for analysis."]
        
        if len(data) < 3:
            return ["Insufficient data for meaningful analysis. Need at least 3 records."]
        
        insights = []
        
        try:
            # Calculate correlations
            numeric_cols = ['avg_temp_c', 'humidity_pct', 'rain_mm', 'total_upi_txn_count', 'avg_txn_value_inr']
            available_cols = [col for col in numeric_cols if col in data.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = data[available_cols].corr()
                
                # Find strongest correlations
                for i, col1 in enumerate(available_cols):
                    for j, col2 in enumerate(available_cols[i+1:], i+1):
                        corr_val = corr_matrix.loc[col1, col2]
                        if not pd.isna(corr_val):
                            strength = "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.3 else "weak"
                            direction = "positive" if corr_val > 0 else "negative"
                            insights.append(f"{strength.title()} {direction} correlation ({corr_val:.3f}) between {col1} and {col2}")
            
            # Check for outliers
            if 'weather_outlier' in data.columns:
                weather_outliers = data['weather_outlier'].sum()
                if weather_outliers > 0:
                    insights.append(f"Found {weather_outliers} weather outliers in the dataset")
            
            if 'txn_volume_outlier' in data.columns:
                txn_outliers = data['txn_volume_outlier'].sum()
                if txn_outliers > 0:
                    insights.append(f"Found {txn_outliers} transaction volume outliers in the dataset")
            
            # Weekly patterns
            if 'date' in data.columns and len(data) >= 7:
                data_copy = data.copy()
                data_copy['weekday'] = pd.to_datetime(data_copy['date']).dt.day_name()
                if 'total_upi_txn_count' in data_copy.columns:
                    weekly_avg = data_copy.groupby('weekday')['total_upi_txn_count'].mean()
                    if len(weekly_avg) > 1:
                        peak_day = weekly_avg.idxmax()
                        low_day = weekly_avg.idxmin()
                        insights.append(f"Peak transaction day: {peak_day}, Lowest: {low_day}")
            
        except Exception as e:
            insights.append(f"Error generating insights: {str(e)}")
        
        return insights if insights else ["No significant patterns detected in the data."]
    
    def get_data_source_status(self):
        """Get data source status from merged data file"""
        try:
            if config.MERGED_DATA_FILE.exists():
                df = pd.read_csv(config.MERGED_DATA_FILE)
                if 'condition' in df.columns:
                    if df['condition'].str.contains('API_Data').any():
                        return "live"
                    else:
                        return "csv"
                elif 'source' in df.columns:
                    if df['source'].str.contains('api').any():
                        return "live"
                    else:
                        return "csv"
                else:
                    return "csv"  # Default to CSV if no source info
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def render_analytics_dashboard(self, merged_df, analytics_results):
        """Render the main analytics dashboard"""
        if merged_df is None or analytics_results is None:
            return
        
        st.subheader("📈 Weather-UPI Correlation Analysis")
        
        # Generate and display insights
        insights = self.generate_automated_insights(merged_df)
        if insights:
            st.subheader("🔍 Key Insights")
            for insight in insights:
                st.info(f"💡 {insight}")
        
        # Correlation heatmap
        correlations = analytics_results.get('correlations', {})
        if correlations:
            st.subheader("🔥 Correlation Heatmap")
            
            # Create correlation matrix for visualization
            corr_data = []
            for pair, corr in correlations.items():
                if not pd.isna(corr):
                    parts = pair.split('_vs_')
                    if len(parts) == 2:
                        corr_data.append({
                            'Variable 1': parts[0].replace('_', ' ').title(),
                            'Variable 2': parts[1].replace('_', ' ').title(),
                            'Correlation': corr
                        })
            
            if corr_data:
                corr_df = pd.DataFrame(corr_data)
                
                # Create heatmap
                fig = px.imshow(
                    corr_df.pivot(index='Variable 1', columns='Variable 2', values='Correlation'),
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title='Weather-UPI Correlation Matrix'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Time series plots
        st.subheader("📊 Time Series Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weather time series
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged_df['date'],
                y=merged_df['avg_temp_c'],
                mode='lines+markers',
                name='Temperature (°C)',
                line=dict(color='red')
            ))
            fig.add_trace(go.Scatter(
                x=merged_df['date'],
                y=merged_df['rain_mm'],
                mode='lines+markers',
                name='Rainfall (mm)',
                yaxis='y2',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title='Weather Patterns Over Time',
                xaxis_title='Date',
                yaxis=dict(title='Temperature (°C)', side='left'),
                yaxis2=dict(title='Rainfall (mm)', side='right', overlaying='y'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # UPI time series
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged_df['date'],
                y=merged_df['total_upi_txn_count'],
                mode='lines+markers',
                name='Transaction Count',
                line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=merged_df['date'],
                y=merged_df['avg_txn_value_inr'],
                mode='lines+markers',
                name='Avg Transaction Value (₹)',
                yaxis='y2',
                line=dict(color='orange')
            ))
            fig.update_layout(
                title='UPI Transaction Patterns Over Time',
                xaxis_title='Date',
                yaxis=dict(title='Transaction Count', side='left'),
                yaxis2=dict(title='Avg Value (₹)', side='right', overlaying='y'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plots for correlations
        st.subheader("🎯 Correlation Scatter Plots")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                merged_df,
                x='avg_temp_c',
                y='total_upi_txn_count',
                title='Temperature vs UPI Transaction Count',
                labels={'avg_temp_c': 'Temperature (°C)', 'total_upi_txn_count': 'Transaction Count'},
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                merged_df,
                x='rain_mm',
                y='avg_txn_value_inr',
                title='Rainfall vs Average Transaction Value',
                labels={'rain_mm': 'Rainfall (mm)', 'avg_txn_value_inr': 'Avg Transaction Value (₹)'},
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def do_fetch(self, start_date, end_date, try_live, auto_fallback):
        """Handle data fetching logic"""
        # Import here to avoid circular imports
        from weather_api import fetch_open_meteo
        from data_loader import load_weather_csv
        
        # Attempt live (if selected) else CSV
        if try_live:
            df = fetch_open_meteo(start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                st.session_state.weather_df = df
                st.session_state.weather_source = "api"
                st.success(f"Weather data loaded from LIVE API ({len(df)} rows)")
                return
            else:
                st.info("Live API returned no rows for selected range.")
                if auto_fallback:
                    csv_df = load_weather_csv("weather_mumbai_2024_11_synthetic.csv")
                    if csv_df is not None:
                        csv_df["source"] = "csv"
                        st.session_state.weather_df = csv_df
                        st.session_state.weather_source = "csv"
                        st.success(f"Weather data loaded from fallback CSV ({len(csv_df)} rows)")
                        return
                    else:
                        st.error("Fallback CSV not available.")
                        return
                # If not auto_fallback, show a fallback button so user can choose
                if st.sidebar.button("Load fallback CSV now"):
                    csv_df = load_weather_csv("weather_mumbai_2024_11_synthetic.csv")
                    if csv_df is not None:
                        csv_df["source"] = "csv"
                        st.session_state.weather_df = csv_df
                        st.session_state.weather_source = "csv"
                        st.success(f"Weather data loaded from fallback CSV ({len(csv_df)} rows)")
                        return
                st.info("Enable auto-fallback or click 'Load fallback CSV now' in the sidebar.")
                return
        else:
            csv_df = load_weather_csv("weather_mumbai_2024_11_synthetic.csv")
            if csv_df is not None:
                csv_df["source"] = "csv"
                st.session_state.weather_df = csv_df
                st.session_state.weather_source = "csv"
                st.success(f"Weather data loaded from fallback CSV ({len(csv_df)} rows)")
                return
            else:
                st.error("Fallback CSV not available.")

    def run(self):
        """Main dashboard execution"""
        st.title("🌦️ Weather-UPI Correlation Dashboard")
        
        # Initialize session state
        if "weather_df" not in st.session_state:
            st.session_state.weather_df = None
            st.session_state.weather_source = None
        
        # Render sidebar controls
        start_date, end_date, try_live, auto_fallback, refresh_interval, fetch_btn = self.render_sidebar_controls()
        
        if fetch_btn:
            self.do_fetch(start_date, end_date, try_live, auto_fallback)
        
        # show badge
        if st.session_state.weather_source == "api":
            st.success("Data source: LIVE (API)")
        elif st.session_state.weather_source == "csv":
            st.warning("Data source: CSV (fallback)")
        else:
            st.info("No data loaded yet. Click 'Fetch data for selected range' in the sidebar.")
        
        # Check if we have weather data
        weather_df = st.session_state.get("weather_df")
        if weather_df is None:
            st.info("👆 Please select a date range and click 'Fetch data for selected range' to begin analysis.")
            return
        
        # Load and merge data if weather data is available
        merged_df, analytics_results = self.load_and_merge_data()
        
        # Render data preview
        self.render_data_preview(merged_df)
        
        # Render analytics dashboard
        if merged_df is not None and analytics_results is not None:
            self.render_analytics_dashboard(merged_df, analytics_results)


def main():
    """Main function to run the dashboard"""
    dashboard = WeatherUPIDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()