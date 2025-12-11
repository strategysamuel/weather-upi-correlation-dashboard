"""
Streamlit Dashboard Module

This module provides an interactive web dashboard for visualizing weather-UPI
correlation analysis results using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date
from pathlib import Path
import sys
import os

# Add project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import config
from weather_api import get_weather_data, PipelineError
from data_loader import load_upi_csv
from data_transformer import DataTransformer
from analytics_engine import analyze_weather_upi_correlations

class WeatherUPIDashboard:
    """Main dashboard class for Weather-UPI correlation analysis"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.setup_page_config()
        self.data = None
        self.analytics_data = None
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=config.DASHBOARD_TITLE,
            page_icon="🌦️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .correlation-positive {
            color: #28a745;
            font-weight: bold;
        }
        .correlation-negative {
            color: #dc3545;
            font-weight: bold;
        }
        .correlation-neutral {
            color: #6c757d;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @st.cache_data
    def load_data(_self):
        """
        Load and cache analytics data for dashboard performance optimization.
        
        This method implements Streamlit's caching mechanism to improve dashboard
        performance by avoiding repeated file I/O operations. It loads both the
        main analytics dataset and the merged dataset as backup.
        
        Data loading process:
        1. Attempts to load analytics.csv (enhanced with outlier flags and z-scores)
        2. Validates required columns for dashboard functionality
        3. Loads merged_weather_upi_data.csv as backup if available
        4. Converts date columns to datetime format
        5. Provides user feedback on loading status
        
        Caching behavior:
        - Results are cached until data files change
        - Cache is automatically invalidated when source files are modified
        - Improves dashboard responsiveness for repeated access
        
        Returns:
            tuple: (analytics_df, merged_df) where:
                  - analytics_df: Main dataset with outlier analysis (pd.DataFrame or None)
                  - merged_df: Backup merged dataset (pd.DataFrame or None)
                  Returns (None, None) if loading fails
                  
        Side Effects:
            - Displays success/error messages in Streamlit interface
            - Logs data loading status and any issues encountered
            
        Example:
            >>> dashboard = WeatherUPIDashboard()
            >>> analytics_df, merged_df = dashboard.load_data()
            >>> if analytics_df is not None:
            ...     print(f"Loaded {len(analytics_df)} records")
            
        Requirements: 5.1, 5.2 (data loading and caching for dashboard performance)
        """
        try:
            # Load analytics data (enhanced with outlier flags and z-scores)
            analytics_file = config.ANALYTICS_FILE
            if analytics_file.exists():
                analytics_df = pd.read_csv(analytics_file)
                analytics_df['date'] = pd.to_datetime(analytics_df['date'])
                
                # Validate required columns for dashboard functionality
                required_cols = ['date', 'total_upi_txn_count', 'avg_txn_value_inr', 
                               'avg_temp_c', 'humidity_pct', 'rain_mm']
                missing_cols = [col for col in required_cols if col not in analytics_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns in analytics data: {missing_cols}")
                    return None, None
                
                # Load merged data as backup
                merged_file = config.MERGED_DATA_FILE
                merged_df = None
                if merged_file.exists():
                    merged_df = pd.read_csv(merged_file)
                    merged_df['date'] = pd.to_datetime(merged_df['date'])
                
                # Log successful data loading
                st.success(f"✅ Data loaded successfully: {len(analytics_df)} records from {analytics_df['date'].min().strftime('%Y-%m-%d')} to {analytics_df['date'].max().strftime('%Y-%m-%d')}")
                
                return analytics_df, merged_df
            else:
                st.error(f"❌ Analytics file not found: {analytics_file}")
                st.info("💡 Run `python main.py` to generate the analytics data.")
                return None, None
                
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.info("💡 Please check that the data files exist and are properly formatted.")
            return None, None
    
    def create_sidebar_navigation(self):
        """
        Create comprehensive sidebar navigation and interactive filtering controls.
        
        This method builds the dashboard's sidebar interface providing:
        1. Page navigation menu for different analysis views
        2. Date range filtering with intelligent defaults
        3. Weather condition filtering
        4. Outlier-specific filtering options
        5. Real-time data quality indicators
        6. Contextual help and information
        
        Navigation pages:
        - Overview: Key metrics and insights summary
        - Trends: Time series analysis and patterns
        - Correlations: Statistical relationship analysis
        - Anomalies: Outlier detection and explanation
        - Data Summary: Dataset statistics and preview
        
        Filtering capabilities:
        - Date range selection with validation
        - Weather condition filtering (if available)
        - Outlier-only view toggle
        - Real-time filter result preview
        
        Returns:
            tuple: (selected_page, filtered_data) where:
                  - selected_page: String indicating user's page selection
                  - filtered_data: DataFrame filtered according to user selections
                  
        Side Effects:
            - Creates Streamlit sidebar widgets
            - Displays data quality metrics
            - Shows filtering status and warnings
            
        Example:
            >>> dashboard = WeatherUPIDashboard()
            >>> page, data = dashboard.create_sidebar_navigation()
            >>> print(f"Selected page: {page}, Data records: {len(data)}")
            
        Requirements: 5.5, 6.5 (interactive filtering and date range selection)
        """
        st.sidebar.markdown("## 🌦️ Weather-UPI Dashboard")
        st.sidebar.markdown("*Analyzing Mumbai weather patterns and UPI transaction correlations*")
        st.sidebar.markdown("---")
        
        # Navigation menu
        page = st.sidebar.selectbox(
            "📍 Navigate to:",
            ["📊 Overview", "📈 Trends", "🔗 Correlations", "⚠️ Anomalies", "📋 Data Summary"],
            help="Select a page to explore different aspects of the weather-UPI analysis"
        )
        
        st.sidebar.markdown("---")
        
        # Date range filter
        if self.analytics_data is not None and not self.analytics_data.empty:
            min_date = self.analytics_data['date'].min().date()
            max_date = self.analytics_data['date'].max().date()
            
            st.sidebar.markdown("### 📅 Date Range Filter")
            date_range = st.sidebar.date_input(
                "Select date range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Filter data by selecting a specific date range"
            )
            
            # Apply date filter
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_data = self.analytics_data[
                    (self.analytics_data['date'].dt.date >= start_date) &
                    (self.analytics_data['date'].dt.date <= end_date)
                ]
            else:
                filtered_data = self.analytics_data
            
            st.sidebar.markdown("---")
            
            # Additional filters
            st.sidebar.markdown("### 🔍 Additional Filters")
            
            # Weather condition filter
            if 'condition' in filtered_data.columns:
                conditions = ['All'] + sorted(filtered_data['condition'].unique().tolist())
                selected_condition = st.sidebar.selectbox(
                    "Weather Condition:",
                    conditions,
                    help="Filter by weather condition"
                )
                
                if selected_condition != 'All':
                    filtered_data = filtered_data[filtered_data['condition'] == selected_condition]
            
            # Outlier filter
            show_outliers_only = st.sidebar.checkbox(
                "Show outliers only",
                help="Display only records flagged as outliers"
            )
            
            if show_outliers_only:
                filtered_data = filtered_data[
                    (filtered_data['txn_volume_outlier'] == True) | 
                    (filtered_data['weather_outlier'] == True)
                ]
            
            st.sidebar.markdown("---")
            
            # Live data fetch controls
            st.sidebar.markdown("### 🔄 Live Data Fetch")
            auto_fallback = st.sidebar.checkbox(
                "Auto fallback to CSV if live fails",
                value=False,
                help="Automatically use CSV data if live API fails"
            )
            
            if st.sidebar.button("Fetch Live Weather Data", help="Fetch fresh data from weather API"):
                self.handle_live_fetch(auto_fallback)
            
            st.sidebar.markdown("---")
            
            # Data quality indicators
            st.sidebar.markdown("### 📊 Data Quality")
            total_records = len(filtered_data)
            
            if total_records > 0:
                outlier_records = filtered_data['txn_volume_outlier'].sum() + filtered_data['weather_outlier'].sum()
                quality_score = ((total_records - outlier_records) / total_records) * 100
                
                st.sidebar.metric("Total Records", total_records)
                st.sidebar.metric("Outlier Records", outlier_records)
                st.sidebar.metric("Data Quality Score", f"{quality_score:.1f}%")
                
                # Data range info
                if not filtered_data.empty:
                    st.sidebar.info(f"📅 **Date Range:** {filtered_data['date'].min().strftime('%Y-%m-%d')} to {filtered_data['date'].max().strftime('%Y-%m-%d')}")
            else:
                st.sidebar.warning("No data matches the current filters")
        else:
            filtered_data = pd.DataFrame()
            st.sidebar.error("❌ No data available for filtering")
        
        return page, filtered_data
    
    def get_data_source_status(self):
        """Determine if data comes from live API or CSV fallback"""
        try:
            merged_file = config.MERGED_DATA_FILE
            if merged_file.exists():
                df = pd.read_csv(merged_file)
                if 'condition' in df.columns:
                    if any(df['condition'] == 'API_Data'):
                        return "LIVE (API)"
                    else:
                        return "CSV (fallback)"
                else:
                    return "CSV (fallback)"
            else:
                return "Unknown"
        except Exception:
            return "Unknown"
    
    def handle_live_fetch(self, auto_fallback: bool):
        """Handle live data fetch with fallback logic"""
        with st.spinner("Fetching live weather data..."):
            try:
                # Fetch live weather data
                weather_df = get_weather_data(
                    start_date="2024-11-01",
                    end_date="2024-11-30",
                    use_csv_fallback=False,
                    interactive=False
                )
                
                # Load UPI data
                upi_df = load_upi_csv(str(config.UPI_DATA_FILE))
                
                # Transform and merge
                transformer = DataTransformer()
                merged_df = transformer.transform_and_merge(weather_df, upi_df)
                
                # Perform analytics
                analytics_results = analyze_weather_upi_correlations(merged_df)
                enhanced_df = analytics_results.get('enhanced_dataframe')
                
                # Save results
                merged_df.to_csv(config.MERGED_DATA_FILE, index=False)
                if enhanced_df is not None:
                    enhanced_df.to_csv(config.ANALYTICS_FILE, index=False)
                
                # Clear cache and refresh
                self.load_data.clear()
                st.success(f"✅ Live data loaded — {len(merged_df)} records")
                st.rerun()
                
            except PipelineError as e:
                if auto_fallback:
                    self.handle_csv_fallback()
                    st.warning("⚠️ Live failed — CSV fallback loaded")
                else:
                    st.error(f"❌ Live API failed: {e}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load CSV fallback"):
                            self.handle_csv_fallback()
                            st.rerun()
                    with col2:
                        if st.button("Cancel"):
                            st.info("Operation cancelled")
            except Exception as e:
                st.error(f"❌ Error during live fetch: {e}")
    
    def handle_csv_fallback(self):
        """Handle CSV fallback loading"""
        try:
            # Load weather data from CSV
            weather_df = get_weather_data(
                start_date="2024-11-01",
                end_date="2024-11-30",
                use_csv_fallback=True,
                interactive=False
            )
            
            # Load UPI data
            upi_df = load_upi_csv(str(config.UPI_DATA_FILE))
            
            # Transform and merge
            transformer = DataTransformer()
            merged_df = transformer.transform_and_merge(weather_df, upi_df)
            
            # Perform analytics
            analytics_results = analyze_weather_upi_correlations(merged_df)
            enhanced_df = analytics_results.get('enhanced_dataframe')
            
            # Save results
            merged_df.to_csv(config.MERGED_DATA_FILE, index=False)
            if enhanced_df is not None:
                enhanced_df.to_csv(config.ANALYTICS_FILE, index=False)
            
            # Clear cache
            self.load_data.clear()
            
        except Exception as e:
            st.error(f"❌ Error loading CSV fallback: {e}")
    
    def display_header(self):
        """Display main dashboard header with data source badge"""
        st.markdown('<h1 class="main-header">🌦️ Weather-UPI Correlation Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown("**Analyzing correlations between Mumbai weather patterns and UPI transaction data**")
        
        # Data source badge
        data_source = self.get_data_source_status()
        if data_source == "LIVE (API)":
            st.success(f"📡 Data source: {data_source}")
        elif data_source == "CSV (fallback)":
            st.info(f"📁 Data source: {data_source}")
        else:
            st.warning(f"❓ Data source: {data_source}")
        
        st.markdown("---")
    
    def run(self):
        """
        Main dashboard execution function
        
        Implements the main dashboard structure with Streamlit components,
        data loading with caching, and sidebar navigation with filtering controls.
        
        Requirements: 6.5 (filtering and date range selection capabilities)
        """
        # Display header first
        self.display_header()
        
        # Load data with caching for performance
        self.analytics_data, self.data = self.load_data()
        
        # Check if data is available
        if self.analytics_data is None or self.analytics_data.empty:
            # Show error state with helpful information
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.error("❌ No analytics data available")
                st.info("🔧 **Next Steps:**")
                st.markdown("""
                1. Run the data pipeline: `python main.py`
                2. Ensure the output directory contains `analytics.csv`
                3. Refresh this dashboard
                """)
                
                # Show expected file locations
                st.markdown("**Expected Files:**")
                st.code(f"📁 {config.ANALYTICS_FILE}")
                st.code(f"📁 {config.MERGED_DATA_FILE}")
            return
        
        # Create sidebar navigation and get filtered data
        page, filtered_data = self.create_sidebar_navigation()
        
        # Show data loading status in main area
        if filtered_data.empty:
            st.warning("⚠️ No data matches the current filters. Please adjust your selection.")
            return
        
        # Route to appropriate page based on navigation
        try:
            if page == "📊 Overview":
                self.show_overview_page(filtered_data)
            elif page == "📈 Trends":
                self.show_trends_page(filtered_data)
            elif page == "🔗 Correlations":
                self.show_correlations_page(filtered_data)
            elif page == "⚠️ Anomalies":
                self.show_anomalies_page(filtered_data)
            elif page == "📋 Data Summary":
                self.show_data_summary_page(filtered_data)
        except Exception as e:
            st.error(f"❌ Error displaying page: {e}")
            st.info("Please try refreshing the page or selecting a different view.")
    
    def show_overview_page(self, data):
        """
        Display overview page with key metrics and insights
        
        Provides a comprehensive overview of the weather-UPI correlation analysis
        with key performance indicators and initial insights.
        """
        st.header("📊 Dashboard Overview")
        
        if data.empty:
            st.warning("⚠️ No data available for the selected date range.")
            return
        
        # Key metrics in columns
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_temp = data['avg_temp_c'].mean()
            temp_delta = None
            if len(data) > 1:
                temp_delta = data['avg_temp_c'].iloc[-1] - data['avg_temp_c'].iloc[0]
            st.metric("Avg Temperature", f"{avg_temp:.1f}°C", delta=f"{temp_delta:.1f}°C" if temp_delta else None)
        
        with col2:
            avg_rain = data['rain_mm'].mean()
            total_rain = data['rain_mm'].sum()
            st.metric("Avg Daily Rainfall", f"{avg_rain:.2f}mm", help=f"Total rainfall: {total_rain:.1f}mm")
        
        with col3:
            avg_txn = data['total_upi_txn_count'].mean()
            total_txn = data['total_upi_txn_count'].sum()
            st.metric("Avg Daily Transactions", f"{avg_txn:,.0f}", help=f"Total transactions: {total_txn:,}")
        
        with col4:
            avg_value = data['avg_txn_value_inr'].mean()
            st.metric("Avg Transaction Value", f"₹{avg_value:.2f}")
        
        st.markdown("---")
        
        # Data quality overview
        st.subheader("🔍 Data Quality Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_records = len(data)
            st.metric("Total Records", total_records)
        
        with col2:
            weather_outliers = data['weather_outlier'].sum() if 'weather_outlier' in data.columns else 0
            st.metric("Weather Outliers", weather_outliers)
        
        with col3:
            txn_outliers = data['txn_volume_outlier'].sum() if 'txn_volume_outlier' in data.columns else 0
            st.metric("Transaction Outliers", txn_outliers)
        
        st.markdown("---")
        
        # Quick correlation insights
        st.subheader("🔗 Quick Correlation Insights")
        
        # Calculate basic correlations for overview
        temp_txn_corr = data['avg_temp_c'].corr(data['total_upi_txn_count'])
        rain_txn_corr = data['rain_mm'].corr(data['total_upi_txn_count'])
        humidity_txn_corr = data['humidity_pct'].corr(data['total_upi_txn_count'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            corr_class = "correlation-positive" if temp_txn_corr > 0.1 else "correlation-negative" if temp_txn_corr < -0.1 else "correlation-neutral"
            st.markdown(f'<div class="{corr_class}">🌡️ Temperature vs Transactions: {temp_txn_corr:.3f}</div>', unsafe_allow_html=True)
            if abs(temp_txn_corr) > 0.3:
                strength = "Strong" if abs(temp_txn_corr) > 0.7 else "Moderate"
                direction = "positive" if temp_txn_corr > 0 else "negative"
                st.caption(f"{strength} {direction} correlation")
        
        with col2:
            corr_class = "correlation-positive" if rain_txn_corr > 0.1 else "correlation-negative" if rain_txn_corr < -0.1 else "correlation-neutral"
            st.markdown(f'<div class="{corr_class}">🌧️ Rainfall vs Transactions: {rain_txn_corr:.3f}</div>', unsafe_allow_html=True)
            if abs(rain_txn_corr) > 0.3:
                strength = "Strong" if abs(rain_txn_corr) > 0.7 else "Moderate"
                direction = "positive" if rain_txn_corr > 0 else "negative"
                st.caption(f"{strength} {direction} correlation")
        
        with col3:
            corr_class = "correlation-positive" if humidity_txn_corr > 0.1 else "correlation-negative" if humidity_txn_corr < -0.1 else "correlation-neutral"
            st.markdown(f'<div class="{corr_class}">💧 Humidity vs Transactions: {humidity_txn_corr:.3f}</div>', unsafe_allow_html=True)
            if abs(humidity_txn_corr) > 0.3:
                strength = "Strong" if abs(humidity_txn_corr) > 0.7 else "Moderate"
                direction = "positive" if humidity_txn_corr > 0 else "negative"
                st.caption(f"{strength} {direction} correlation")
        
        # Summary insights
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        # Generate comprehensive automated insights
        insights = self.generate_automated_insights(data)
        
        if insights:
            for insight in insights:
                st.info(insight)
        else:
            st.info("📊 **Analysis**: No strong correlations detected in the current data range. Try adjusting filters or exploring different time periods.")
    
    def show_trends_page(self, data):
        """
        Display trends analysis page with line charts for UPI and weather data
        
        Requirements: 5.1, 5.2, 5.3, 5.5 (line charts, weather trends, interactive filtering)
        """
        st.header("📈 Trends Analysis")
        
        if data.empty:
            st.warning("⚠️ No data available for the selected date range.")
            return
        
        # UPI Transaction Trends
        st.subheader("💳 UPI Transaction Volume Over Time")
        
        # Create UPI transaction volume line chart
        fig_upi = px.line(
            data, 
            x='date', 
            y='total_upi_txn_count',
            title='Daily UPI Transaction Volume',
            labels={
                'total_upi_txn_count': 'Transaction Count',
                'date': 'Date'
            },
            line_shape='spline'
        )
        
        # Add outlier markers if available
        if 'txn_volume_outlier' in data.columns:
            outlier_data = data[data['txn_volume_outlier'] == True]
            if not outlier_data.empty:
                fig_upi.add_scatter(
                    x=outlier_data['date'],
                    y=outlier_data['total_upi_txn_count'],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='diamond'),
                    name='Outliers',
                    hovertemplate='<b>Outlier</b><br>Date: %{x}<br>Transactions: %{y:,}<extra></extra>'
                )
        
        fig_upi.update_layout(
            xaxis_title="Date",
            yaxis_title="Transaction Count",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig_upi, width='stretch')
        
        # Average Transaction Value Trend
        st.subheader("💰 Average Transaction Value Over Time")
        
        fig_value = px.line(
            data,
            x='date',
            y='avg_txn_value_inr',
            title='Daily Average Transaction Value',
            labels={
                'avg_txn_value_inr': 'Average Value (₹)',
                'date': 'Date'
            },
            line_shape='spline'
        )
        
        fig_value.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Transaction Value (₹)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_value, width='stretch')
        
        st.markdown("---")
        
        # Weather Trends
        st.subheader("🌦️ Weather Trends Over Time")
        
        # Create subplot for multiple weather variables
        fig_weather = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Temperature (°C)', 'Rainfall (mm)', 'Humidity (%)'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Temperature trend
        fig_weather.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['avg_temp_c'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=4),
                hovertemplate='<b>Temperature</b><br>Date: %{x}<br>Temp: %{y:.1f}°C<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Rainfall trend
        fig_weather.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['rain_mm'],
                mode='lines+markers',
                name='Rainfall',
                line=dict(color='#2ca02c', width=2),
                marker=dict(size=4),
                fill='tonexty' if data['rain_mm'].max() > 0 else None,
                hovertemplate='<b>Rainfall</b><br>Date: %{x}<br>Rain: %{y:.2f}mm<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Humidity trend
        fig_weather.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['humidity_pct'],
                mode='lines+markers',
                name='Humidity',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4),
                hovertemplate='<b>Humidity</b><br>Date: %{x}<br>Humidity: %{y:.1f}%<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add weather outlier markers if available
        if 'weather_outlier' in data.columns:
            weather_outliers = data[data['weather_outlier'] == True]
            if not weather_outliers.empty:
                # Add outlier markers for each weather variable
                for idx, outlier_row in weather_outliers.iterrows():
                    # Temperature outliers
                    fig_weather.add_trace(
                        go.Scatter(
                            x=[outlier_row['date']],
                            y=[outlier_row['avg_temp_c']],
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='diamond'),
                            name='Weather Outlier',
                            showlegend=False,
                            hovertemplate='<b>Temperature Outlier</b><br>Date: %{x}<br>Temp: %{y:.1f}°C<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Rainfall outliers
                    fig_weather.add_trace(
                        go.Scatter(
                            x=[outlier_row['date']],
                            y=[outlier_row['rain_mm']],
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='diamond'),
                            name='Weather Outlier',
                            showlegend=False,
                            hovertemplate='<b>Rainfall Outlier</b><br>Date: %{x}<br>Rain: %{y:.2f}mm<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Humidity outliers
                    fig_weather.add_trace(
                        go.Scatter(
                            x=[outlier_row['date']],
                            y=[outlier_row['humidity_pct']],
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='diamond'),
                            name='Weather Outlier',
                            showlegend=False,
                            hovertemplate='<b>Humidity Outlier</b><br>Date: %{x}<br>Humidity: %{y:.1f}%<extra></extra>'
                        ),
                        row=3, col=1
                    )
        
        fig_weather.update_layout(
            height=800,
            title_text="Weather Variables Over Time",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update y-axis labels
        fig_weather.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig_weather.update_yaxes(title_text="Rainfall (mm)", row=2, col=1)
        fig_weather.update_yaxes(title_text="Humidity (%)", row=3, col=1)
        fig_weather.update_xaxes(title_text="Date", row=3, col=1)
        
        st.plotly_chart(fig_weather, width='stretch')
        
        # Interactive date range selection for detailed view
        st.markdown("---")
        st.subheader("🔍 Detailed View")
        
        # Date range selector for zoomed view
        col1, col2 = st.columns(2)
        with col1:
            zoom_start = st.date_input(
                "Zoom Start Date",
                value=data['date'].min().date(),
                min_value=data['date'].min().date(),
                max_value=data['date'].max().date()
            )
        
        with col2:
            zoom_end = st.date_input(
                "Zoom End Date", 
                value=data['date'].max().date(),
                min_value=data['date'].min().date(),
                max_value=data['date'].max().date()
            )
        
        # Filter data for zoomed view
        if zoom_start <= zoom_end:
            zoom_data = data[
                (data['date'].dt.date >= zoom_start) & 
                (data['date'].dt.date <= zoom_end)
            ]
            
            if not zoom_data.empty:
                # Combined trend chart for detailed view
                fig_combined = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('UPI Transactions vs Temperature', 'Transaction Value vs Rainfall'),
                    vertical_spacing=0.1,
                    shared_xaxes=True,
                    specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
                )
                
                # UPI vs Temperature
                fig_combined.add_trace(
                    go.Scatter(
                        x=zoom_data['date'],
                        y=zoom_data['total_upi_txn_count'],
                        mode='lines+markers',
                        name='UPI Transactions',
                        line=dict(color='blue', width=2),
                        yaxis='y'
                    ),
                    row=1, col=1
                )
                
                fig_combined.add_trace(
                    go.Scatter(
                        x=zoom_data['date'],
                        y=zoom_data['avg_temp_c'],
                        mode='lines+markers',
                        name='Temperature',
                        line=dict(color='red', width=2),
                        yaxis='y2'
                    ),
                    row=1, col=1, secondary_y=True
                )
                
                # Transaction Value vs Rainfall
                fig_combined.add_trace(
                    go.Scatter(
                        x=zoom_data['date'],
                        y=zoom_data['avg_txn_value_inr'],
                        mode='lines+markers',
                        name='Avg Transaction Value',
                        line=dict(color='green', width=2),
                        yaxis='y3'
                    ),
                    row=2, col=1
                )
                
                fig_combined.add_trace(
                    go.Scatter(
                        x=zoom_data['date'],
                        y=zoom_data['rain_mm'],
                        mode='lines+markers',
                        name='Rainfall',
                        line=dict(color='purple', width=2),
                        yaxis='y4'
                    ),
                    row=2, col=1, secondary_y=True
                )
                
                # Update layout
                fig_combined.update_layout(
                    height=600,
                    title_text=f"Detailed Analysis: {zoom_start} to {zoom_end}",
                    hovermode='x unified'
                )
                
                # Update y-axes
                fig_combined.update_yaxes(title_text="Transaction Count", row=1, col=1)
                fig_combined.update_yaxes(title_text="Temperature (°C)", row=1, col=1, secondary_y=True)
                fig_combined.update_yaxes(title_text="Transaction Value (₹)", row=2, col=1)
                fig_combined.update_yaxes(title_text="Rainfall (mm)", row=2, col=1, secondary_y=True)
                
                st.plotly_chart(fig_combined, width='stretch')
            else:
                st.warning("No data available for the selected zoom range.")
        else:
            st.error("Start date must be before or equal to end date.")
    
    def show_correlations_page(self, data):
        """
        Display correlation analysis page with heatmap and detailed correlation metrics
        
        Requirements: 5.3 (correlation heatmap display)
        """
        st.header("🔗 Correlation Analysis")
        
        if data.empty:
            st.warning("⚠️ No data available for the selected date range.")
            return
        
        # Select numeric columns for correlation analysis
        numeric_cols = ['avg_temp_c', 'humidity_pct', 'rain_mm', 'total_upi_txn_count', 'avg_txn_value_inr']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if len(available_cols) < 2:
            st.error("❌ Insufficient numeric data for correlation analysis.")
            return
        
        # Calculate correlation matrix
        correlation_matrix = data[available_cols].corr()
        
        # Create correlation heatmap
        st.subheader("🔥 Correlation Heatmap")
        
        # Create heatmap using plotly
        fig_heatmap = px.imshow(
            correlation_matrix,
            text_auto='.3f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Weather-UPI Correlation Matrix",
            labels=dict(color="Correlation Coefficient")
        )
        
        # Update layout for better readability
        fig_heatmap.update_layout(
            width=800,
            height=600,
            title_x=0.5,
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        # Update axis labels for better readability
        label_mapping = {
            'avg_temp_c': 'Temperature (°C)',
            'humidity_pct': 'Humidity (%)',
            'rain_mm': 'Rainfall (mm)',
            'total_upi_txn_count': 'UPI Transaction Count',
            'avg_txn_value_inr': 'Avg Transaction Value (₹)'
        }
        
        # Update tick labels
        fig_heatmap.update_xaxes(
            ticktext=[label_mapping.get(col, col) for col in correlation_matrix.columns],
            tickvals=list(range(len(correlation_matrix.columns)))
        )
        fig_heatmap.update_yaxes(
            ticktext=[label_mapping.get(col, col) for col in correlation_matrix.index],
            tickvals=list(range(len(correlation_matrix.index)))
        )
        
        st.plotly_chart(fig_heatmap, width='stretch')
        
        st.markdown("---")
        
        # Detailed correlation analysis
        st.subheader("📊 Detailed Correlation Analysis")
        
        # Weather-UPI correlations specifically
        weather_cols = ['avg_temp_c', 'humidity_pct', 'rain_mm']
        upi_cols = ['total_upi_txn_count', 'avg_txn_value_inr']
        
        weather_available = [col for col in weather_cols if col in data.columns]
        upi_available = [col for col in upi_cols if col in data.columns]
        
        if weather_available and upi_available:
            st.subheader("🌦️ Weather vs UPI Correlations")
            
            correlation_results = []
            
            for weather_var in weather_available:
                for upi_var in upi_available:
                    corr_coef = data[weather_var].corr(data[upi_var])
                    
                    # Calculate statistical significance (basic approach)
                    n = len(data.dropna(subset=[weather_var, upi_var]))
                    if n > 2:
                        # Simple t-test approximation for correlation significance
                        t_stat = corr_coef * np.sqrt((n - 2) / (1 - corr_coef**2)) if abs(corr_coef) < 1 else np.inf
                        # Rough p-value approximation (for display purposes)
                        p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + np.sqrt(n - 2))) if abs(t_stat) != np.inf else 0.0
                    else:
                        p_value = 1.0
                    
                    correlation_results.append({
                        'Weather Variable': label_mapping.get(weather_var, weather_var),
                        'UPI Variable': label_mapping.get(upi_var, upi_var),
                        'Correlation': corr_coef,
                        'Strength': 'Strong' if abs(corr_coef) > 0.7 else 'Moderate' if abs(corr_coef) > 0.3 else 'Weak',
                        'Direction': 'Positive' if corr_coef > 0 else 'Negative',
                        'Significance': 'Significant' if p_value < 0.05 else 'Not Significant'
                    })
            
            # Display correlation results in a table
            if correlation_results:
                corr_df = pd.DataFrame(correlation_results)
                
                # Style the dataframe
                def style_correlation(val):
                    if abs(val) > 0.7:
                        return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    elif abs(val) > 0.3:
                        return 'background-color: #fff3cd; color: #856404; font-weight: bold'
                    else:
                        return 'background-color: #f8d7da; color: #721c24'
                
                styled_df = corr_df.style.map(style_correlation, subset=['Correlation'])
                st.dataframe(styled_df, width='stretch')
                
                # Key insights
                st.subheader("💡 Key Correlation Insights")
                
                # Find strongest correlations
                strong_correlations = [r for r in correlation_results if abs(r['Correlation']) > 0.5]
                moderate_correlations = [r for r in correlation_results if 0.3 < abs(r['Correlation']) <= 0.5]
                
                if strong_correlations:
                    st.success("🎯 **Strong Correlations Found:**")
                    for corr in strong_correlations:
                        direction_emoji = "📈" if corr['Direction'] == 'Positive' else "📉"
                        st.write(f"{direction_emoji} **{corr['Weather Variable']}** vs **{corr['UPI Variable']}**: {corr['Correlation']:.3f} ({corr['Strength']} {corr['Direction'].lower()})")
                
                if moderate_correlations:
                    st.info("📊 **Moderate Correlations:**")
                    for corr in moderate_correlations:
                        direction_emoji = "📈" if corr['Direction'] == 'Positive' else "📉"
                        st.write(f"{direction_emoji} **{corr['Weather Variable']}** vs **{corr['UPI Variable']}**: {corr['Correlation']:.3f} ({corr['Strength']} {corr['Direction'].lower()})")
                
                if not strong_correlations and not moderate_correlations:
                    st.warning("⚠️ No strong or moderate correlations detected in the current dataset.")
        
        st.markdown("---")
        
        # Scatter plots for top correlations
        st.subheader("📈 Correlation Scatter Plots")
        
        # Find the strongest correlation for detailed visualization
        if len(available_cols) >= 2:
            # Get all pairwise correlations
            correlations = []
            for i, col1 in enumerate(available_cols):
                for j, col2 in enumerate(available_cols):
                    if i < j:  # Avoid duplicates
                        corr_val = data[col1].corr(data[col2])
                        correlations.append((abs(corr_val), corr_val, col1, col2))
            
            # Sort by absolute correlation value
            correlations.sort(reverse=True)
            
            # Show top 3 correlations as scatter plots
            top_correlations = correlations[:3]
            
            for i, (abs_corr, corr_val, col1, col2) in enumerate(top_correlations):
                if abs_corr > 0.1:  # Only show meaningful correlations
                    col1_label = label_mapping.get(col1, col1)
                    col2_label = label_mapping.get(col2, col2)
                    
                    fig_scatter = px.scatter(
                        data,
                        x=col1,
                        y=col2,
                        title=f"{col1_label} vs {col2_label} (r = {corr_val:.3f})",
                        labels={col1: col1_label, col2: col2_label},
                        trendline="ols",
                        hover_data=['date']
                    )
                    
                    # Add outlier highlighting if available
                    if 'txn_volume_outlier' in data.columns or 'weather_outlier' in data.columns:
                        outlier_mask = data.get('txn_volume_outlier', False) | data.get('weather_outlier', False)
                        if outlier_mask.any():
                            outlier_data = data[outlier_mask]
                            fig_scatter.add_scatter(
                                x=outlier_data[col1],
                                y=outlier_data[col2],
                                mode='markers',
                                marker=dict(color='red', size=10, symbol='diamond'),
                                name='Outliers',
                                hovertemplate=f'<b>Outlier</b><br>{col1_label}: %{{x}}<br>{col2_label}: %{{y}}<extra></extra>'
                            )
                    
                    fig_scatter.update_layout(
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_scatter, width='stretch')
        
        # Correlation interpretation guide
        st.markdown("---")
        st.subheader("📚 Correlation Interpretation Guide")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Strong Correlation**
            - |r| > 0.7
            - Clear linear relationship
            - High predictive value
            """)
        
        with col2:
            st.markdown("""
            **Moderate Correlation**
            - 0.3 < |r| ≤ 0.7
            - Noticeable relationship
            - Some predictive value
            """)
        
        with col3:
            st.markdown("""
            **Weak Correlation**
            - |r| ≤ 0.3
            - Little to no relationship
            - Limited predictive value
            """)
    
    def show_anomalies_page(self, data):
        """
        Display anomaly detection page with outlier analysis and explanations
        
        Requirements: 5.4 (anomaly highlighting and explanation features)
        """
        st.header("⚠️ Anomaly Detection")
        
        if data.empty:
            st.warning("⚠️ No data available for the selected date range.")
            return
        
        # Check if outlier columns exist
        has_txn_outliers = 'txn_volume_outlier' in data.columns
        has_weather_outliers = 'weather_outlier' in data.columns
        has_z_scores = any(col in data.columns for col in ['temp_z_score', 'rain_z_score', 'txn_z_score'])
        
        if not (has_txn_outliers or has_weather_outliers):
            st.warning("⚠️ No outlier detection data available. Please run the analytics pipeline first.")
            return
        
        # Anomaly Summary
        st.subheader("📊 Anomaly Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_records = len(data)
            st.metric("Total Records", total_records)
        
        with col2:
            txn_outliers = data['txn_volume_outlier'].sum() if has_txn_outliers else 0
            st.metric("Transaction Outliers", txn_outliers)
        
        with col3:
            weather_outliers = data['weather_outlier'].sum() if has_weather_outliers else 0
            st.metric("Weather Outliers", weather_outliers)
        
        with col4:
            total_outliers = txn_outliers + weather_outliers
            outlier_percentage = (total_outliers / total_records * 100) if total_records > 0 else 0
            st.metric("Outlier Rate", f"{outlier_percentage:.1f}%")
        
        st.markdown("---")
        
        # Outlier Timeline
        st.subheader("📅 Outlier Timeline")
        
        # Create timeline chart showing outliers
        fig_timeline = go.Figure()
        
        # Add transaction volume line
        fig_timeline.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['total_upi_txn_count'],
                mode='lines',
                name='UPI Transactions',
                line=dict(color='blue', width=2),
                yaxis='y'
            )
        )
        
        # Add transaction outliers
        if has_txn_outliers:
            txn_outlier_data = data[data['txn_volume_outlier'] == True]
            if not txn_outlier_data.empty:
                fig_timeline.add_trace(
                    go.Scatter(
                        x=txn_outlier_data['date'],
                        y=txn_outlier_data['total_upi_txn_count'],
                        mode='markers',
                        name='Transaction Outliers',
                        marker=dict(color='red', size=12, symbol='diamond'),
                        yaxis='y',
                        hovertemplate='<b>Transaction Outlier</b><br>Date: %{x}<br>Transactions: %{y:,}<extra></extra>'
                    )
                )
        
        # Add temperature line on secondary y-axis
        fig_timeline.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['avg_temp_c'],
                mode='lines',
                name='Temperature',
                line=dict(color='orange', width=2),
                yaxis='y2'
            )
        )
        
        # Add weather outliers
        if has_weather_outliers:
            weather_outlier_data = data[data['weather_outlier'] == True]
            if not weather_outlier_data.empty:
                fig_timeline.add_trace(
                    go.Scatter(
                        x=weather_outlier_data['date'],
                        y=weather_outlier_data['avg_temp_c'],
                        mode='markers',
                        name='Weather Outliers',
                        marker=dict(color='purple', size=12, symbol='diamond'),
                        yaxis='y2',
                        hovertemplate='<b>Weather Outlier</b><br>Date: %{x}<br>Temperature: %{y:.1f}°C<extra></extra>'
                    )
                )
        
        # Update layout for dual y-axis
        fig_timeline.update_layout(
            title="Outlier Detection Timeline",
            xaxis_title="Date",
            yaxis=dict(title="UPI Transaction Count", side="left"),
            yaxis2=dict(title="Temperature (°C)", side="right", overlaying="y"),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_timeline, width='stretch')
        
        st.markdown("---")
        
        # Detailed Outlier Analysis
        st.subheader("🔍 Detailed Outlier Analysis")
        
        # Transaction Outliers
        if has_txn_outliers and txn_outliers > 0:
            st.subheader("💳 Transaction Volume Outliers")
            
            txn_outlier_data = data[data['txn_volume_outlier'] == True].copy()
            
            # Add z-score information if available
            if 'txn_z_score' in data.columns:
                txn_outlier_data = txn_outlier_data.copy()
                txn_outlier_data['Z-Score'] = txn_outlier_data['txn_z_score']
            
            # Display outlier details
            display_cols = ['date', 'total_upi_txn_count', 'avg_txn_value_inr']
            if 'txn_z_score' in txn_outlier_data.columns:
                display_cols.append('txn_z_score')
            
            available_display_cols = [col for col in display_cols if col in txn_outlier_data.columns]
            
            if available_display_cols:
                st.dataframe(
                    txn_outlier_data[available_display_cols].sort_values('date'),
                    width='stretch'
                )
                
                # Generate insights for transaction outliers
                insights = self.generate_transaction_outlier_insights(txn_outlier_data, data)
                if insights:
                    st.info("💡 **Transaction Outlier Insights:**")
                    for insight in insights:
                        st.write(f"• {insight}")
        
        # Weather Outliers
        if has_weather_outliers and weather_outliers > 0:
            st.subheader("🌦️ Weather Outliers")
            
            weather_outlier_data = data[data['weather_outlier'] == True].copy()
            
            # Display weather outlier details
            weather_display_cols = ['date', 'avg_temp_c', 'humidity_pct', 'rain_mm']
            if has_z_scores:
                z_score_cols = [col for col in ['temp_z_score', 'rain_z_score'] if col in data.columns]
                weather_display_cols.extend(z_score_cols)
            
            available_weather_cols = [col for col in weather_display_cols if col in weather_outlier_data.columns]
            
            if available_weather_cols:
                st.dataframe(
                    weather_outlier_data[available_weather_cols].sort_values('date'),
                    width='stretch'
                )
                
                # Generate insights for weather outliers
                insights = self.generate_weather_outlier_insights(weather_outlier_data, data)
                if insights:
                    st.info("💡 **Weather Outlier Insights:**")
                    for insight in insights:
                        st.write(f"• {insight}")
        
        st.markdown("---")
        
        # Z-Score Distribution Analysis
        if has_z_scores:
            st.subheader("📈 Z-Score Distribution Analysis")
            
            z_score_cols = [col for col in ['temp_z_score', 'rain_z_score', 'txn_z_score'] if col in data.columns]
            
            if z_score_cols:
                # Create histogram of z-scores
                fig_zscore = make_subplots(
                    rows=1, cols=len(z_score_cols),
                    subplot_titles=[col.replace('_z_score', '').replace('_', ' ').title() for col in z_score_cols]
                )
                
                for i, col in enumerate(z_score_cols):
                    fig_zscore.add_trace(
                        go.Histogram(
                            x=data[col],
                            name=col.replace('_z_score', '').title(),
                            nbinsx=20,
                            opacity=0.7
                        ),
                        row=1, col=i+1
                    )
                    
                    # Add outlier threshold lines
                    fig_zscore.add_vline(x=2, line_dash="dash", line_color="red", row=1, col=i+1)
                    fig_zscore.add_vline(x=-2, line_dash="dash", line_color="red", row=1, col=i+1)
                
                fig_zscore.update_layout(
                    title="Z-Score Distributions (Red lines show ±2σ outlier thresholds)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_zscore, width='stretch')
        
        # Anomaly Impact Analysis
        st.markdown("---")
        st.subheader("📊 Anomaly Impact Analysis")
        
        # Compare normal vs outlier periods
        normal_data = data[
            (data.get('txn_volume_outlier', False) == False) & 
            (data.get('weather_outlier', False) == False)
        ]
        
        outlier_data = data[
            (data.get('txn_volume_outlier', False) == True) | 
            (data.get('weather_outlier', False) == True)
        ]
        
        if not normal_data.empty and not outlier_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Normal Periods**")
                st.metric("Avg Transactions", f"{normal_data['total_upi_txn_count'].mean():,.0f}")
                st.metric("Avg Temperature", f"{normal_data['avg_temp_c'].mean():.1f}°C")
                st.metric("Avg Rainfall", f"{normal_data['rain_mm'].mean():.2f}mm")
            
            with col2:
                st.markdown("**Outlier Periods**")
                st.metric("Avg Transactions", f"{outlier_data['total_upi_txn_count'].mean():,.0f}")
                st.metric("Avg Temperature", f"{outlier_data['avg_temp_c'].mean():.1f}°C")
                st.metric("Avg Rainfall", f"{outlier_data['rain_mm'].mean():.2f}mm")
            
            # Generate comparative insights
            comparative_insights = self.generate_comparative_insights(normal_data, outlier_data)
            if comparative_insights:
                st.info("💡 **Comparative Analysis:**")
                for insight in comparative_insights:
                    st.write(f"• {insight}")
    
    def generate_transaction_outlier_insights(self, outlier_data, full_data):
        """
        Generate automated insights for transaction outliers
        
        Requirements: 5.4 (automated insight generation)
        """
        insights = []
        
        if outlier_data.empty:
            return insights
        
        # Temporal patterns
        outlier_dates = outlier_data['date'].dt.day_name().value_counts()
        if not outlier_dates.empty:
            most_common_day = outlier_dates.index[0]
            insights.append(f"Most transaction outliers occur on {most_common_day}s ({outlier_dates.iloc[0]} occurrences)")
        
        # Magnitude analysis
        avg_normal_txn = full_data[full_data['txn_volume_outlier'] == False]['total_upi_txn_count'].mean()
        avg_outlier_txn = outlier_data['total_upi_txn_count'].mean()
        
        if avg_outlier_txn > avg_normal_txn:
            pct_increase = ((avg_outlier_txn - avg_normal_txn) / avg_normal_txn) * 100
            insights.append(f"Transaction outliers show {pct_increase:.1f}% higher volume than normal periods")
        else:
            pct_decrease = ((avg_normal_txn - avg_outlier_txn) / avg_normal_txn) * 100
            insights.append(f"Transaction outliers show {pct_decrease:.1f}% lower volume than normal periods")
        
        # Weather correlation during outliers
        if 'avg_temp_c' in outlier_data.columns:
            outlier_temp = outlier_data['avg_temp_c'].mean()
            normal_temp = full_data[full_data['txn_volume_outlier'] == False]['avg_temp_c'].mean()
            
            temp_diff = outlier_temp - normal_temp
            if abs(temp_diff) > 2:
                direction = "higher" if temp_diff > 0 else "lower"
                insights.append(f"Transaction outliers coincide with {direction} temperatures (avg {outlier_temp:.1f}°C vs normal {normal_temp:.1f}°C)")
        
        return insights
    
    def generate_weather_outlier_insights(self, outlier_data, full_data):
        """
        Generate automated insights for weather outliers
        
        Requirements: 5.4 (automated insight generation)
        """
        insights = []
        
        if outlier_data.empty:
            return insights
        
        # Temperature extremes
        if 'avg_temp_c' in outlier_data.columns:
            max_temp = outlier_data['avg_temp_c'].max()
            min_temp = outlier_data['avg_temp_c'].min()
            normal_temp_range = full_data[full_data['weather_outlier'] == False]['avg_temp_c']
            
            if max_temp > normal_temp_range.max():
                insights.append(f"Extreme high temperature detected: {max_temp:.1f}°C")
            
            if min_temp < normal_temp_range.min():
                insights.append(f"Extreme low temperature detected: {min_temp:.1f}°C")
        
        # Rainfall extremes
        if 'rain_mm' in outlier_data.columns:
            max_rain = outlier_data['rain_mm'].max()
            if max_rain > 0:
                normal_rain_max = full_data[full_data['weather_outlier'] == False]['rain_mm'].max()
                if max_rain > normal_rain_max * 1.5:
                    insights.append(f"Extreme rainfall event detected: {max_rain:.2f}mm")
        
        # Seasonal patterns
        outlier_months = outlier_data['date'].dt.month_name().value_counts()
        if not outlier_months.empty:
            most_common_month = outlier_months.index[0]
            insights.append(f"Most weather outliers occur in {most_common_month} ({outlier_months.iloc[0]} occurrences)")
        
        return insights
    
    def generate_comparative_insights(self, normal_data, outlier_data):
        """
        Generate insights comparing normal vs outlier periods
        
        Requirements: 5.4 (automated insight generation)
        """
        insights = []
        
        if normal_data.empty or outlier_data.empty:
            return insights
        
        # Transaction volume comparison
        normal_txn_avg = normal_data['total_upi_txn_count'].mean()
        outlier_txn_avg = outlier_data['total_upi_txn_count'].mean()
        
        txn_diff_pct = ((outlier_txn_avg - normal_txn_avg) / normal_txn_avg) * 100
        
        if abs(txn_diff_pct) > 10:
            direction = "higher" if txn_diff_pct > 0 else "lower"
            insights.append(f"Outlier periods show {abs(txn_diff_pct):.1f}% {direction} transaction volumes on average")
        
        # Transaction value comparison
        if 'avg_txn_value_inr' in normal_data.columns and 'avg_txn_value_inr' in outlier_data.columns:
            normal_value_avg = normal_data['avg_txn_value_inr'].mean()
            outlier_value_avg = outlier_data['avg_txn_value_inr'].mean()
            
            value_diff_pct = ((outlier_value_avg - normal_value_avg) / normal_value_avg) * 100
            
            if abs(value_diff_pct) > 5:
                direction = "higher" if value_diff_pct > 0 else "lower"
                insights.append(f"Average transaction values are {abs(value_diff_pct):.1f}% {direction} during outlier periods")
        
        # Weather pattern differences
        if 'avg_temp_c' in normal_data.columns:
            normal_temp_std = normal_data['avg_temp_c'].std()
            outlier_temp_std = outlier_data['avg_temp_c'].std()
            
            if outlier_temp_std > normal_temp_std * 1.5:
                insights.append("Temperature variability is significantly higher during outlier periods")
        
        return insights
    
    def generate_automated_insights(self, data):
        """
        Generate comprehensive automated insights based on correlation analysis.
        
        This method analyzes the weather-UPI correlation data and automatically
        generates human-readable insights about relationships, patterns, and
        anomalies in the dataset. It uses statistical analysis to identify
        meaningful correlations and translate them into actionable insights.
        
        Insight categories generated:
        1. Temperature impact analysis - correlation with transaction patterns
        2. Rainfall effect analysis - weather event impact on payments
        3. Humidity influence analysis - atmospheric condition effects
        4. Transaction value patterns - weather impact on payment amounts
        5. Outlier analysis - data quality and anomaly insights
        6. Temporal patterns - seasonal and time-based trends
        7. Data completeness assessment - quality indicators
        
        Correlation strength interpretation:
        - Strong: |r| > 0.7 - Clear predictive relationship
        - Moderate: 0.3 < |r| ≤ 0.7 - Noticeable relationship
        - Weak: |r| ≤ 0.3 - Limited relationship
        
        Args:
            data (pd.DataFrame): Filtered dataset containing weather and UPI data
                                Must include correlation-relevant columns
            
        Returns:
            List[str]: List of insight strings formatted for display
                      Each insight includes emoji indicators and statistical details
                      Empty list if no significant patterns detected
                      
        Example:
            >>> dashboard = WeatherUPIDashboard()
            >>> insights = dashboard.generate_automated_insights(filtered_data)
            >>> for insight in insights:
            ...     print(insight)
            "🌡️ Temperature Impact: Strong correlation (0.734) - higher temperatures tend to increase transaction volumes"
            
        Requirements: 5.4 (automated insight generation describing key findings)
        """
        insights = []
        
        if data.empty:
            return insights
        
        # Calculate correlations
        temp_txn_corr = data['avg_temp_c'].corr(data['total_upi_txn_count'])
        rain_txn_corr = data['rain_mm'].corr(data['total_upi_txn_count'])
        humidity_txn_corr = data['humidity_pct'].corr(data['total_upi_txn_count'])
        
        # Temperature insights
        if abs(temp_txn_corr) > 0.5:
            direction = "increase" if temp_txn_corr > 0 else "decrease"
            strength = "Strong" if abs(temp_txn_corr) > 0.7 else "Moderate"
            insights.append(f"🌡️ **Temperature Impact**: {strength} correlation detected ({temp_txn_corr:.3f}) - higher temperatures tend to {direction} transaction volumes")
        elif abs(temp_txn_corr) > 0.3:
            direction = "increase" if temp_txn_corr > 0 else "decrease"
            insights.append(f"🌡️ **Temperature Trend**: Moderate correlation ({temp_txn_corr:.3f}) suggests temperature may {direction} transaction activity")
        
        # Rainfall insights  
        if abs(rain_txn_corr) > 0.5:
            direction = "boost" if rain_txn_corr > 0 else "reduce"
            strength = "Strong" if abs(rain_txn_corr) > 0.7 else "Moderate"
            insights.append(f"🌧️ **Weather Effect**: {strength} correlation ({rain_txn_corr:.3f}) - rainfall appears to significantly {direction} digital payment activity")
        elif abs(rain_txn_corr) > 0.3:
            direction = "boost" if rain_txn_corr > 0 else "reduce"
            insights.append(f"🌧️ **Rainfall Pattern**: Moderate correlation ({rain_txn_corr:.3f}) suggests rain may {direction} transaction volumes")
        
        # Humidity insights
        if abs(humidity_txn_corr) > 0.5:
            direction = "increase" if humidity_txn_corr > 0 else "decrease"
            strength = "Strong" if abs(humidity_txn_corr) > 0.7 else "Moderate"
            insights.append(f"💧 **Humidity Impact**: {strength} correlation ({humidity_txn_corr:.3f}) - higher humidity tends to {direction} transaction volumes")
        
        # Transaction value insights
        if 'avg_txn_value_inr' in data.columns:
            temp_value_corr = data['avg_temp_c'].corr(data['avg_txn_value_inr'])
            rain_value_corr = data['rain_mm'].corr(data['avg_txn_value_inr'])
            
            if abs(temp_value_corr) > 0.4:
                direction = "higher" if temp_value_corr > 0 else "lower"
                insights.append(f"💰 **Transaction Value**: Temperature shows correlation ({temp_value_corr:.3f}) with transaction values - warmer weather associated with {direction} value transactions")
            
            if abs(rain_value_corr) > 0.4:
                direction = "higher" if rain_value_corr > 0 else "lower"
                insights.append(f"💰 **Payment Behavior**: Rainfall correlates ({rain_value_corr:.3f}) with transaction values - rainy days see {direction} value payments")
        
        # Outlier insights
        weather_outliers = data['weather_outlier'].sum() if 'weather_outlier' in data.columns else 0
        txn_outliers = data['txn_volume_outlier'].sum() if 'txn_volume_outlier' in data.columns else 0
        total_outliers = weather_outliers + txn_outliers
        total_records = len(data)
        
        if total_outliers > 0:
            outlier_pct = (total_outliers / total_records) * 100
            if outlier_pct > 10:
                insights.append(f"⚠️ **Data Quality Alert**: High outlier rate ({outlier_pct:.1f}%) detected - {weather_outliers} weather and {txn_outliers} transaction anomalies")
            else:
                insights.append(f"✅ **Data Quality**: Good data quality with {outlier_pct:.1f}% outliers ({weather_outliers} weather, {txn_outliers} transaction)")
        
        # Seasonal/temporal insights
        if len(data) > 7:  # Need sufficient data for temporal analysis
            # Weekly patterns
            data_with_weekday = data.copy()
            data_with_weekday['weekday'] = data_with_weekday['date'].dt.day_name()
            weekday_txn = data_with_weekday.groupby('weekday')['total_upi_txn_count'].mean()
            
            if not weekday_txn.empty:
                highest_day = weekday_txn.idxmax()
                lowest_day = weekday_txn.idxmin()
                
                if weekday_txn.max() / weekday_txn.min() > 1.2:  # 20% difference
                    insights.append(f"📅 **Weekly Pattern**: {highest_day}s show highest transaction activity, {lowest_day}s show lowest")
        
        # Extreme weather impact
        if 'avg_temp_c' in data.columns and len(data) > 5:
            # Find extreme weather days (top 20% of temperatures)
            temp_threshold = data['avg_temp_c'].quantile(0.8)
            
            extreme_temp_data = data[data['avg_temp_c'] >= temp_threshold]
            normal_temp_data = data[data['avg_temp_c'] < temp_threshold]
            
            if not extreme_temp_data.empty and not normal_temp_data.empty:
                extreme_avg = extreme_temp_data['total_upi_txn_count'].mean()
                normal_avg = normal_temp_data['total_upi_txn_count'].mean()
                
                # Prevent divide-by-zero
                if normal_avg > 0 and not np.isnan(extreme_avg) and not np.isnan(normal_avg):
                    rel_change = (extreme_avg - normal_avg) / normal_avg
                    
                    # Only generate insight if change is significant (15% threshold)
                    if abs(rel_change) >= 0.15:
                        direction = "higher" if rel_change > 0 else "lower"
                        change_pct = abs(rel_change) * 100
                        insights.append(f"🔥 **Extreme Heat Impact**: Very hot days (>{temp_threshold:.1f}°C) show {change_pct:.1f}% {direction} transaction volumes")
            
            # Heavy rain impact analysis
            if 'rain_mm' in data.columns:
                rain_q95 = data['rain_mm'].quantile(0.95)
                
                if rain_q95 > 0:
                    extreme_rain = data[data['rain_mm'] >= rain_q95]
                    normal_rain = data[data['rain_mm'] < rain_q95]
                    
                    if not extreme_rain.empty and not normal_rain.empty:
                        rain_extreme_avg = extreme_rain['total_upi_txn_count'].mean()
                        rain_normal_avg = normal_rain['total_upi_txn_count'].mean()
                        
                        # Prevent divide-by-zero
                        if rain_normal_avg > 0 and not np.isnan(rain_extreme_avg) and not np.isnan(rain_normal_avg):
                            rain_rel_change = (rain_extreme_avg - rain_normal_avg) / rain_normal_avg
                            
                            # Only generate insight if change is significant (15% threshold)
                            if abs(rain_rel_change) >= 0.15:
                                rain_direction = "higher" if rain_rel_change > 0 else "lower"
                                rain_change_pct = abs(rain_rel_change) * 100
                                insights.append(f"🌊 **Heavy Rain Impact**: Heavy rain days (>{rain_q95:.1f}mm) show {rain_change_pct:.1f}% {rain_direction} transaction activity")
        
        # Data completeness insights
        missing_data_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_data_pct > 5:
            insights.append(f"📊 **Data Completeness**: {missing_data_pct:.1f}% missing data detected - consider data quality improvements")
        elif missing_data_pct == 0:
            insights.append("✅ **Data Completeness**: Perfect data quality - no missing values detected")
        
        return insights
    
    def show_data_summary_page(self, data):
        """Display data summary and statistics"""
        st.header("📋 Data Summary")
        
        if data.empty:
            st.warning("No data available for the selected date range.")
            return
        
        # Data overview
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Records", len(data))
            st.metric("Date Range", f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
        
        with col2:
            st.metric("Weather Outliers", data['weather_outlier'].sum())
            st.metric("Transaction Outliers", data['txn_volume_outlier'].sum())
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10))
        
        # Basic statistics
        st.subheader("Statistical Summary")
        numeric_cols = ['avg_temp_c', 'humidity_pct', 'rain_mm', 'total_upi_txn_count', 'avg_txn_value_inr']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if available_cols:
            st.dataframe(data[available_cols].describe())


def main():
    """Main function to run the Streamlit dashboard"""
    dashboard = WeatherUPIDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()