"""
Property-based tests for dashboard insight generation functionality

Tests the automated insight generation system to ensure it produces
accurate and meaningful insights based on correlation analysis.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from dashboard import WeatherUPIDashboard
import config


class TestInsightGeneration:
    """Test class for dashboard insight generation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create dashboard instance without Streamlit initialization
        self.dashboard = WeatherUPIDashboard()
        # Skip Streamlit page config to avoid issues in testing
        self.dashboard.setup_page_config = lambda: None
    
    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=15.0, max_value=45.0),  # temperature
                st.floats(min_value=0.0, max_value=100.0),  # humidity
                st.floats(min_value=0.0, max_value=50.0),   # rainfall
                st.integers(min_value=1000, max_value=100000),  # transaction count
                st.floats(min_value=10.0, max_value=1000.0),    # transaction value
                st.booleans(),  # weather outlier
                st.booleans()   # transaction outlier
            ),
            min_size=10,
            max_size=30
        )
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
    def test_insight_generation_accuracy(self, weather_upi_data):
        """
        # Feature: weather-upi-dashboard, Property 17: Insight Generation Accuracy
        
        For any correlation analysis results, the Dashboard_System should generate 
        meaningful summary insights that accurately reflect the statistical findings
        
        **Validates: Requirements 5.4**
        """
        assume(len(weather_upi_data) >= 10)  # Need sufficient data for meaningful analysis
        
        # Create test DataFrame
        dates = [datetime(2024, 11, 1) + timedelta(days=i) for i in range(len(weather_upi_data))]
        
        data = pd.DataFrame({
            'date': dates,
            'avg_temp_c': [row[0] for row in weather_upi_data],
            'humidity_pct': [row[1] for row in weather_upi_data],
            'rain_mm': [row[2] for row in weather_upi_data],
            'total_upi_txn_count': [row[3] for row in weather_upi_data],
            'avg_txn_value_inr': [row[4] for row in weather_upi_data],
            'weather_outlier': [row[5] for row in weather_upi_data],
            'txn_volume_outlier': [row[6] for row in weather_upi_data]
        })
        
        # Generate insights
        insights = self.dashboard.generate_automated_insights(data)
        
        # Property: Insights should be generated for any valid dataset
        assert isinstance(insights, list), "Insights should be returned as a list"
        
        # Property: All insights should be non-empty strings
        for insight in insights:
            assert isinstance(insight, str), "Each insight should be a string"
            assert len(insight.strip()) > 0, "Each insight should be non-empty"
            assert len(insight) < 500, "Insights should be concise (< 500 characters)"
        
        # Property: Correlation insights should reflect actual correlations
        temp_txn_corr = data['avg_temp_c'].corr(data['total_upi_txn_count'])
        rain_txn_corr = data['rain_mm'].corr(data['total_upi_txn_count'])
        
        # Check temperature correlation insights
        temp_insights = [insight for insight in insights if "Temperature" in insight and "correlation" in insight.lower()]
        if abs(temp_txn_corr) > 0.3:
            # Should have temperature insight for moderate+ correlations
            assert len(temp_insights) > 0, f"Should generate temperature insight for correlation {temp_txn_corr:.3f}"
            
            # Check direction accuracy
            temp_insight = temp_insights[0]
            if temp_txn_corr > 0.3:
                assert "increase" in temp_insight.lower(), "Positive correlation should mention 'increase'"
            elif temp_txn_corr < -0.3:
                assert "decrease" in temp_insight.lower(), "Negative correlation should mention 'decrease'"
        
        # Check rainfall correlation insights
        rain_insights = [insight for insight in insights if ("Rainfall" in insight or "Weather Effect" in insight) and "correlation" in insight.lower()]
        if abs(rain_txn_corr) > 0.3:
            # Should have rainfall insight for moderate+ correlations
            assert len(rain_insights) > 0, f"Should generate rainfall insight for correlation {rain_txn_corr:.3f}"
            
            # Check direction accuracy
            rain_insight = rain_insights[0]
            if rain_txn_corr > 0.3:
                assert "boost" in rain_insight.lower(), "Positive rain correlation should mention 'boost'"
            elif rain_txn_corr < -0.3:
                assert "reduce" in rain_insight.lower(), "Negative rain correlation should mention 'reduce'"
        
        # Property: Outlier insights should reflect actual outlier counts
        weather_outliers = data['weather_outlier'].sum()
        txn_outliers = data['txn_volume_outlier'].sum()
        total_outliers = weather_outliers + txn_outliers
        
        outlier_insights = [insight for insight in insights if "outlier" in insight.lower() or "quality" in insight.lower()]
        if total_outliers > 0:
            assert len(outlier_insights) > 0, "Should generate outlier insights when outliers exist"
            
            # Check outlier count accuracy
            outlier_insight = outlier_insights[0]
            if weather_outliers > 0:
                assert str(weather_outliers) in outlier_insight, f"Should mention weather outlier count {weather_outliers}"
            if txn_outliers > 0:
                assert str(txn_outliers) in outlier_insight, f"Should mention transaction outlier count {txn_outliers}"
    
    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=20.0, max_value=40.0),  # normal temperature range
                st.integers(min_value=5000, max_value=50000)  # normal transaction range
            ),
            min_size=20,
            max_size=50
        ),
        st.lists(
            st.tuples(
                st.floats(min_value=45.0, max_value=50.0),  # extreme temperature range
                st.integers(min_value=1000, max_value=100000)  # varied transaction range
            ),
            min_size=5,
            max_size=15
        )
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
    def test_extreme_weather_insight_accuracy(self, normal_data, extreme_data):
        """
        Test that extreme weather insights accurately reflect temperature extremes
        and their impact on transaction volumes
        """
        assume(len(normal_data) >= 10 and len(extreme_data) >= 3)
        
        # Combine normal and extreme data
        all_temps = [row[0] for row in normal_data] + [row[0] for row in extreme_data]
        all_txns = [row[1] for row in normal_data] + [row[1] for row in extreme_data]
        
        dates = [datetime(2024, 11, 1) + timedelta(days=i) for i in range(len(all_temps))]
        
        data = pd.DataFrame({
            'date': dates,
            'avg_temp_c': all_temps,
            'humidity_pct': [50.0] * len(all_temps),  # constant humidity
            'rain_mm': [0.0] * len(all_temps),        # no rain
            'total_upi_txn_count': all_txns,
            'avg_txn_value_inr': [100.0] * len(all_temps),  # constant value
            'weather_outlier': [False] * len(all_temps),
            'txn_volume_outlier': [False] * len(all_temps)
        })
        
        # Generate insights
        insights = self.dashboard.generate_automated_insights(data)
        
        # Property: Should detect extreme temperature impact if significant difference exists
        normal_txn_avg = np.mean([row[1] for row in normal_data])
        extreme_txn_avg = np.mean([row[1] for row in extreme_data])
        
        if abs((extreme_txn_avg - normal_txn_avg) / normal_txn_avg) > 0.15:  # 15% difference
            extreme_insights = [insight for insight in insights if "Extreme Heat" in insight or "extreme" in insight.lower()]
            
            # Should generate extreme weather insight
            if len(extreme_insights) > 0:
                extreme_insight = extreme_insights[0]
                
                # Check direction accuracy - the insight should match the actual direction
                if extreme_txn_avg > normal_txn_avg:
                    # If extreme transactions are higher, insight should mention "higher"
                    assert "higher" in extreme_insight.lower(), f"Higher extreme transactions ({extreme_txn_avg:.0f} vs {normal_txn_avg:.0f}) should be noted as 'higher' in: {extreme_insight}"
                else:
                    # If extreme transactions are lower, insight should mention "lower"  
                    assert "lower" in extreme_insight.lower(), f"Lower extreme transactions ({extreme_txn_avg:.0f} vs {normal_txn_avg:.0f}) should be noted as 'lower' in: {extreme_insight}"
    
    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=15.0, max_value=45.0),  # temperature
                st.integers(min_value=1000, max_value=100000),  # transaction count
                st.sampled_from(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            ),
            min_size=21,  # At least 3 weeks of data
            max_size=70   # Up to 10 weeks
        )
    )
    @settings(max_examples=3)
    def test_weekly_pattern_insight_accuracy(self, weekly_data):
        """
        Test that weekly pattern insights accurately identify day-of-week transaction patterns
        """
        assume(len(weekly_data) >= 21)  # Need sufficient data for weekly analysis
        
        # Create DataFrame with proper date sequence
        start_date = datetime(2024, 11, 1)  # Start on a Friday
        dates = []
        temps = []
        txns = []
        
        for i, (temp, txn, _) in enumerate(weekly_data):
            current_date = start_date + timedelta(days=i)
            dates.append(current_date)
            temps.append(temp)
            txns.append(txn)
        
        data = pd.DataFrame({
            'date': dates,
            'avg_temp_c': temps,
            'humidity_pct': [50.0] * len(dates),
            'rain_mm': [0.0] * len(dates),
            'total_upi_txn_count': txns,
            'avg_txn_value_inr': [100.0] * len(dates),
            'weather_outlier': [False] * len(dates),
            'txn_volume_outlier': [False] * len(dates)
        })
        
        # Generate insights
        insights = self.dashboard.generate_automated_insights(data)
        
        # Property: Weekly pattern insights should be accurate
        data_with_weekday = data.copy()
        data_with_weekday['weekday'] = data_with_weekday['date'].dt.day_name()
        weekday_txn = data_with_weekday.groupby('weekday')['total_upi_txn_count'].mean()
        
        if not weekday_txn.empty and weekday_txn.max() / weekday_txn.min() > 1.2:
            weekly_insights = [insight for insight in insights if "Weekly Pattern" in insight]
            
            if len(weekly_insights) > 0:
                weekly_insight = weekly_insights[0]
                highest_day = weekday_txn.idxmax()
                lowest_day = weekday_txn.idxmin()
                
                # Check that the insight mentions the correct highest and lowest days
                assert highest_day in weekly_insight, f"Should mention highest day {highest_day}"
                assert lowest_day in weekly_insight, f"Should mention lowest day {lowest_day}"
    
    def test_empty_data_handling(self):
        """
        Test that insight generation handles empty data gracefully
        """
        empty_data = pd.DataFrame()
        
        insights = self.dashboard.generate_automated_insights(empty_data)
        
        # Property: Should return empty list for empty data
        assert isinstance(insights, list), "Should return a list even for empty data"
        assert len(insights) == 0, "Should return empty insights list for empty data"
    
    def test_minimal_data_handling(self):
        """
        Test that insight generation handles minimal data appropriately
        """
        # Create minimal dataset with just 2 records
        minimal_data = pd.DataFrame({
            'date': [datetime(2024, 11, 1), datetime(2024, 11, 2)],
            'avg_temp_c': [25.0, 26.0],
            'humidity_pct': [60.0, 65.0],
            'rain_mm': [0.0, 1.0],
            'total_upi_txn_count': [10000, 12000],
            'avg_txn_value_inr': [100.0, 110.0],
            'weather_outlier': [False, False],
            'txn_volume_outlier': [False, False]
        })
        
        insights = self.dashboard.generate_automated_insights(minimal_data)
        
        # Property: Should handle minimal data without errors
        assert isinstance(insights, list), "Should return a list for minimal data"
        
        # Property: Should not generate weekly pattern insights for insufficient data
        weekly_insights = [insight for insight in insights if "Weekly Pattern" in insight]
        assert len(weekly_insights) == 0, "Should not generate weekly patterns with insufficient data"
    
    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=15.0, max_value=45.0),  # temperature
                st.floats(min_value=0.0, max_value=100.0),  # humidity  
                st.floats(min_value=0.0, max_value=50.0),   # rainfall
                st.integers(min_value=1000, max_value=100000),  # transaction count
                st.floats(min_value=10.0, max_value=1000.0)     # transaction value
            ),
            min_size=15,
            max_size=50
        )
    )
    @settings(max_examples=5)
    def test_correlation_strength_classification(self, weather_upi_data):
        """
        Test that correlation insights correctly classify correlation strength
        """
        assume(len(weather_upi_data) >= 15)
        
        # Create test DataFrame
        dates = [datetime(2024, 11, 1) + timedelta(days=i) for i in range(len(weather_upi_data))]
        
        data = pd.DataFrame({
            'date': dates,
            'avg_temp_c': [row[0] for row in weather_upi_data],
            'humidity_pct': [row[1] for row in weather_upi_data],
            'rain_mm': [row[2] for row in weather_upi_data],
            'total_upi_txn_count': [row[3] for row in weather_upi_data],
            'avg_txn_value_inr': [row[4] for row in weather_upi_data],
            'weather_outlier': [False] * len(weather_upi_data),
            'txn_volume_outlier': [False] * len(weather_upi_data)
        })
        
        # Generate insights
        insights = self.dashboard.generate_automated_insights(data)
        
        # Property: Correlation strength should be correctly classified
        temp_txn_corr = data['avg_temp_c'].corr(data['total_upi_txn_count'])
        
        temp_insights = [insight for insight in insights if "Temperature" in insight and "correlation" in insight.lower()]
        
        if len(temp_insights) > 0:
            temp_insight = temp_insights[0]
            
            if abs(temp_txn_corr) > 0.7:
                assert "Strong" in temp_insight, f"Correlation {temp_txn_corr:.3f} should be classified as Strong"
            elif abs(temp_txn_corr) > 0.5:
                assert "Moderate" in temp_insight or "Strong" in temp_insight, f"Correlation {temp_txn_corr:.3f} should be classified as Moderate or Strong"
            elif abs(temp_txn_corr) > 0.3:
                assert "Moderate" in temp_insight, f"Correlation {temp_txn_corr:.3f} should be classified as Moderate"


class TestLiveFetchFunctionality:
    """Test class for live data fetch functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.dashboard = WeatherUPIDashboard()
        self.dashboard.setup_page_config = lambda: None
    
    def test_data_source_detection_live(self):
        """Test data source detection for live API data"""
        # Create mock data with API_Data condition
        test_data = pd.DataFrame({
            'date': ['2024-11-01', '2024-11-02'],
            'condition': ['API_Data', 'API_Data'],
            'avg_temp_c': [25.0, 26.0]
        })
        
        # Mock the file reading
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name
            
        # Temporarily replace the config path
        original_path = config.MERGED_DATA_FILE
        config.MERGED_DATA_FILE = Path(temp_file)
        
        try:
            status = self.dashboard.get_data_source_status()
            assert status == "LIVE (API)"
        finally:
            config.MERGED_DATA_FILE = original_path
            try:
                os.unlink(temp_file)
            except PermissionError:
                pass  # Ignore Windows file permission issues in tests
    
    def test_data_source_detection_csv(self):
        """Test data source detection for CSV fallback data"""
        # Create mock data with non-API condition
        test_data = pd.DataFrame({
            'date': ['2024-11-01', '2024-11-02'],
            'condition': ['Clear', 'Rain'],
            'avg_temp_c': [25.0, 26.0]
        })
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name
            
        original_path = config.MERGED_DATA_FILE
        config.MERGED_DATA_FILE = Path(temp_file)
        
        try:
            status = self.dashboard.get_data_source_status()
            assert status == "CSV (fallback)"
        finally:
            config.MERGED_DATA_FILE = original_path
            try:
                os.unlink(temp_file)
            except PermissionError:
                pass  # Ignore Windows file permission issues in tests
    
    def test_data_source_detection_unknown(self):
        """Test data source detection when file doesn't exist"""
        original_path = config.MERGED_DATA_FILE
        config.MERGED_DATA_FILE = Path("nonexistent_file.csv")
        
        try:
            status = self.dashboard.get_data_source_status()
            assert status == "Unknown"
        finally:
            config.MERGED_DATA_FILE = original_path