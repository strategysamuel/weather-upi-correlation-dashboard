"""
Property-based tests for MCP Weather API module
"""

import pytest
import pandas as pd
from hypothesis import given, strategies as st, settings, HealthCheck
import hypothesis
from unittest.mock import patch, Mock
import json
import requests
from datetime import datetime, timedelta

from src.weather_api import WeatherAPIClient, get_weather_data, request_csv_fallback, PipelineError, FallbackPending
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

class TestWeatherAPIProperties:
    """Property-based tests for weather API functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = WeatherAPIClient()
    
    @given(
        temperatures=st.lists(st.floats(min_value=-50, max_value=60, allow_nan=False, allow_infinity=False), min_size=1, max_size=10),
        humidity=st.lists(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=10),
        precipitation=st.lists(st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False), min_size=1, max_size=10)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_column_validation_completeness(self, temperatures, humidity, precipitation):
        """
        # Feature: weather-upi-dashboard, Property 1: Column Validation Completeness
        **Validates: Requirements 1.1**
        
        For any weather API response with valid data, the parsed DataFrame should contain
        all required columns with correct data types.
        """
        # Ensure all lists have the same length
        min_length = min(len(temperatures), len(humidity), len(precipitation))
        temperatures = temperatures[:min_length]
        humidity = humidity[:min_length]
        precipitation = precipitation[:min_length]
        
        # Generate corresponding dates
        base_date = datetime(2024, 1, 1)
        dates = [(base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(min_length)]
        
        # Create mock API response
        mock_response = {
            "daily": {
                "time": dates,
                "temperature_2m_mean": temperatures,
                "relative_humidity_2m_mean": humidity,
                "precipitation_sum": precipitation
            }
        }
        
        # Parse the response
        df = self.client._parse_weather_response(mock_response)
        
        # Verify all required columns are present
        required_columns = ['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm', 'condition']
        for column in required_columns:
            assert column in df.columns, f"Missing required column: {column}"
        
        # Verify data types
        assert pd.api.types.is_datetime64_any_dtype(df['date']), "Date column should be datetime"
        assert df['city'].dtype == 'object', "City column should be string/object"
        assert pd.api.types.is_numeric_dtype(df['avg_temp_c']), "Temperature should be numeric"
        assert pd.api.types.is_numeric_dtype(df['humidity_pct']), "Humidity should be numeric"
        assert pd.api.types.is_numeric_dtype(df['rain_mm']), "Precipitation should be numeric"
        
        # Verify data integrity
        assert len(df) == min_length, "DataFrame should have same length as input data"
        assert not df['date'].isna().any(), "No date values should be NaN"
        assert (df['city'] == 'Mumbai').all(), "All city values should be Mumbai"
    
    @given(
        start_date=st.dates(min_value=datetime(2020, 1, 1).date(), 
                           max_value=datetime(2024, 12, 31).date()),
        days_range=st.integers(min_value=1, max_value=30)
    )
    @settings(max_examples=100)
    def test_property_19_api_response_validation(self, start_date, days_range):
        """
        # Feature: weather-upi-dashboard, Property 19: API Response Validation
        **Validates: Requirements 2.2, 2.3**
        
        For any successful MCP API call to Open-Meteo, the response should contain
        valid JSON with required weather fields (temperature, rainfall, humidity).
        """
        end_date = start_date + timedelta(days=days_range)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Create valid mock response
        dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") 
                for i in range(days_range + 1)]
        
        mock_api_response = {
            "daily": {
                "time": dates,
                "temperature_2m_mean": [25.0 + i for i in range(len(dates))],
                "relative_humidity_2m_mean": [60.0 + i for i in range(len(dates))],
                "precipitation_sum": [0.1 * i for i in range(len(dates))]
            }
        }
        
        # Mock successful API call
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_api_response
            mock_get.return_value = mock_response
            
            # Test API call (using archive API method)
            result = self.client._call_archive_api_with_retry(start_str, end_str)
            
            # Verify response structure
            assert result is not None, "API call should return data"
            assert "daily" in result, "Response should contain 'daily' key"
            
            daily_data = result["daily"]
            assert "time" in daily_data, "Daily data should contain 'time'"
            assert "temperature_2m_mean" in daily_data, "Daily data should contain temperature"
            assert "relative_humidity_2m_mean" in daily_data, "Daily data should contain humidity"
            assert "precipitation_sum" in daily_data, "Daily data should contain precipitation"
            
            # Verify data consistency
            time_data = daily_data["time"]
            temp_data = daily_data["temperature_2m_mean"]
            humidity_data = daily_data["relative_humidity_2m_mean"]
            precip_data = daily_data["precipitation_sum"]
            
            assert len(time_data) == len(temp_data), "Time and temperature arrays should have same length"
            assert len(time_data) == len(humidity_data), "Time and humidity arrays should have same length"
            assert len(time_data) == len(precip_data), "Time and precipitation arrays should have same length"
            
            # Verify data can be parsed successfully
            parsed_df = self.client._parse_weather_response(result)
            assert not parsed_df.empty, "Parsed DataFrame should not be empty"
            assert len(parsed_df) == len(time_data), "Parsed DataFrame should have correct length"
    
    def test_api_fallback_mechanism(self):
        """Test that API fallback works when API calls fail"""
        # Mock failed API call
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API unavailable")
            
            # Should fallback to CSV without raising exception when use_csv_fallback=True
            df = self.client.fetch_weather_data(use_csv_fallback=True)
            
            # Verify fallback data has correct structure
            required_columns = ['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm', 'condition']
            for column in required_columns:
                assert column in df.columns, f"Fallback data missing column: {column}"
    
    def test_invalid_api_response_handling(self):
        """Test handling of invalid API responses"""
        # Test with missing required fields
        invalid_responses = [
            {},  # Empty response
            {"daily": {}},  # Empty daily data
            {"daily": {"time": ["2024-01-01"]}},  # Missing other fields
            {"daily": {
                "time": ["2024-01-01", "2024-01-02"],
                "temperature_2m_mean": [25.0],  # Mismatched lengths
                "relative_humidity_2m_mean": [60.0, 65.0],
                "precipitation_sum": [0.0, 0.1]
            }}
        ]
        
        for invalid_response in invalid_responses:
            with pytest.raises((ValueError, KeyError, IndexError)):
                self.client._parse_weather_response(invalid_response)


class TestMCPIntegration:
    """Integration tests for MCP weather API functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = WeatherAPIClient()
    
    def test_successful_api_calls_with_http_200(self):
        """
        Test successful API calls with HTTP 200 responses
        Requirements: 7.1, 2.2, 2.4
        """
        # Mock successful HTTP 200 response
        mock_response_data = {
            "daily": {
                "time": ["2024-11-01", "2024-11-02", "2024-11-03"],
                "temperature_2m_mean": [25.5, 26.0, 24.8],
                "relative_humidity_2m_mean": [65.0, 70.0, 68.5],
                "precipitation_sum": [0.0, 2.5, 0.1]
            }
        }
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_get.return_value = mock_response
            
            # Test API call (using archive API method)
            result = self.client._call_archive_api_with_retry("2024-11-01", "2024-11-03")
            
            # Verify successful response
            assert result is not None
            assert result == mock_response_data
            
            # Verify API was called with correct parameters
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "https://archive-api.open-meteo.com/v1/archive" in call_args[0][0]
            
            # Verify parameters include Mumbai coordinates
            params = call_args[1]['params']
            assert params['latitude'] == 19.07
            assert params['longitude'] == 72.88
            assert params['start_date'] == "2024-11-01"
            assert params['end_date'] == "2024-11-03"
    
    def test_api_failure_scenarios_and_fallback(self):
        """
        Test API failure scenarios and fallback mechanisms
        Requirements: 7.1, 2.2, 2.4
        """
        failure_scenarios = [
            # HTTP error codes
            (400, "Bad Request"),
            (401, "Unauthorized"),
            (403, "Forbidden"),
            (404, "Not Found"),
            (429, "Too Many Requests"),
            (500, "Internal Server Error"),
            (502, "Bad Gateway"),
            (503, "Service Unavailable")
        ]
        
        for status_code, error_message in failure_scenarios:
            with patch('requests.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.text = error_message
                mock_get.return_value = mock_response
                
                # Should return None for failed API calls
                result = self.client._call_archive_api_with_retry("2024-11-01", "2024-11-03")
                assert result is None, f"API call should return None for status {status_code}"
        
        # Test network exceptions
        network_exceptions = [
            requests.exceptions.RequestException("Network error"),
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.Timeout("Request timeout"),
            requests.exceptions.HTTPError("HTTP error")
        ]
        
        for exception in network_exceptions:
            with patch('requests.get') as mock_get:
                mock_get.side_effect = exception
                
                # Should return None for network exceptions
                result = self.client._call_archive_api_with_retry("2024-11-01", "2024-11-03")
                assert result is None, f"API call should return None for exception {type(exception).__name__}"
        
        # Test complete fallback mechanism
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API unavailable")
            
            # get_weather_data should fallback to CSV
            df = get_weather_data("2024-11-01", "2024-11-30", use_csv_fallback=True)
            
            # Verify fallback data structure
            assert not df.empty, "Fallback should provide data"
            required_columns = ['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm', 'condition']
            for column in required_columns:
                assert column in df.columns, f"Fallback data missing column: {column}"
    
    def test_json_parsing_and_data_normalization(self):
        """
        Test JSON parsing and data normalization
        Requirements: 7.1, 2.2, 2.4
        """
        # Test various valid JSON response formats
        test_cases = [
            # Standard response
            {
                "daily": {
                    "time": ["2024-11-01", "2024-11-02"],
                    "temperature_2m_mean": [25.5, 26.0],
                    "relative_humidity_2m_mean": [65.0, 70.0],
                    "precipitation_sum": [0.0, 2.5]
                }
            },
            # Response with extra fields (should be ignored)
            {
                "daily": {
                    "time": ["2024-11-01"],
                    "temperature_2m_mean": [25.5],
                    "relative_humidity_2m_mean": [65.0],
                    "precipitation_sum": [0.0],
                    "extra_field": ["ignored"]
                },
                "metadata": {"ignored": True}
            },
            # Response with edge case values
            {
                "daily": {
                    "time": ["2024-11-01", "2024-11-02", "2024-11-03"],
                    "temperature_2m_mean": [-5.0, 0.0, 45.0],  # Extreme temperatures
                    "relative_humidity_2m_mean": [0.0, 50.0, 100.0],  # Full humidity range
                    "precipitation_sum": [0.0, 0.1, 100.0]  # Various precipitation levels
                }
            }
        ]
        
        for i, response_data in enumerate(test_cases):
            with patch('requests.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = response_data
                mock_get.return_value = mock_response
                
                # Test parsing
                df = self.client.fetch_weather_data("2024-11-01", "2024-11-03")
                
                # Verify normalization
                assert not df.empty, f"Test case {i}: DataFrame should not be empty"
                
                # Verify date normalization
                assert pd.api.types.is_datetime64_any_dtype(df['date']), f"Test case {i}: Dates should be normalized to datetime"
                
                # Verify city normalization
                assert (df['city'] == 'Mumbai').all(), f"Test case {i}: All cities should be normalized to Mumbai"
                
                # Verify column names are normalized
                expected_columns = ['date', 'city', 'avg_temp_c', 'humidity_pct', 'rain_mm', 'condition']
                for col in expected_columns:
                    assert col in df.columns, f"Test case {i}: Missing normalized column {col}"
                
                # Verify data types
                assert pd.api.types.is_numeric_dtype(df['avg_temp_c']), f"Test case {i}: Temperature should be numeric"
                assert pd.api.types.is_numeric_dtype(df['humidity_pct']), f"Test case {i}: Humidity should be numeric"
                assert pd.api.types.is_numeric_dtype(df['rain_mm']), f"Test case {i}: Precipitation should be numeric"
    
    def test_rate_limiting_and_error_handling(self):
        """
        Test rate limiting and error handling
        Requirements: 7.1, 2.2, 2.4
        """
        # Test rate limiting (HTTP 429)
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.text = "Too Many Requests"
            mock_response.headers = {"Retry-After": "60"}
            mock_get.return_value = mock_response
            
            # Should handle rate limiting gracefully
            result = self.client._call_archive_api_with_retry("2024-11-01", "2024-11-03")
            assert result is None, "Rate limited requests should return None"
            
            # Should fallback to CSV data
            df = get_weather_data("2024-11-01", "2024-11-30", use_csv_fallback=True)
            assert not df.empty, "Should fallback to CSV when rate limited"
        
        # Test malformed JSON responses
        malformed_json_cases = [
            '{"incomplete": json',  # Invalid JSON syntax
            '{"daily": {"time": ["2024-11-01"], "temperature_2m_mean": [}',  # Incomplete JSON
            '',  # Empty response
            'not json at all',  # Non-JSON response
        ]
        
        for malformed_json in malformed_json_cases:
            with patch('requests.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", malformed_json, 0)
                mock_get.return_value = mock_response
                
                # Should handle JSON decode errors gracefully
                result = self.client._call_archive_api_with_retry("2024-11-01", "2024-11-03")
                assert result is None, "Malformed JSON should return None"
        
        # Test timeout handling
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
            
            # Should handle timeouts gracefully
            result = self.client._call_archive_api_with_retry("2024-11-01", "2024-11-03")
            assert result is None, "Timeout should return None"
            
            # Should fallback to CSV
            df = get_weather_data("2024-11-01", "2024-11-30", use_csv_fallback=True)
            assert not df.empty, "Should fallback to CSV on timeout"
    
    def test_api_parameter_validation(self):
        """
        Test API parameter validation and construction
        Requirements: 7.1, 2.2
        """
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "daily": {
                    "time": ["2024-11-01"],
                    "temperature_2m_mean": [25.0],
                    "relative_humidity_2m_mean": [60.0],
                    "precipitation_sum": [0.0]
                }
            }
            mock_get.return_value = mock_response
            
            # Test API call (using archive API method)
            self.client._call_archive_api_with_retry("2024-11-01", "2024-11-01")
            
            # Verify API was called with correct parameters
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            
            # Check URL
            assert call_args[0][0] == "https://archive-api.open-meteo.com/v1/archive"
            
            # Check parameters
            params = call_args[1]['params']
            expected_params = {
                'latitude': 19.07,
                'longitude': 72.88,
                'daily': 'temperature_2m_max,temperature_2m_min,rain_sum',
                'timezone': 'Asia/Kolkata',
                'start_date': '2024-11-01',
                'end_date': '2024-11-01'
            }
            
            for key, value in expected_params.items():
                assert key in params, f"Missing parameter: {key}"
                assert params[key] == value, f"Parameter {key} has incorrect value: {params[key]} != {value}"


class TestLiveFirstBehavior:
    """Test class for live-first weather API behavior"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = WeatherAPIClient()
        # Store original config values
        self.original_use_live = config.USE_LIVE_WEATHER
        self.original_allow_fallback = config.ALLOW_CSV_FALLBACK
        self.original_interactive = config.INTERACTIVE_FALLBACK_PROMPT
    
    def teardown_method(self):
        """Restore original config values"""
        config.USE_LIVE_WEATHER = self.original_use_live
        config.ALLOW_CSV_FALLBACK = self.original_allow_fallback
        config.INTERACTIVE_FALLBACK_PROMPT = self.original_interactive
    
    def test_live_first_success(self):
        """Test successful live API fetch"""
        config.USE_LIVE_WEATHER = True
        
        with patch('requests.get') as mock_get:
            # Mock successful API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'daily': {
                    'time': ['2024-11-01', '2024-11-02'],
                    'temperature_2m_max': [30.0, 31.0],
                    'temperature_2m_min': [20.0, 21.0],
                    'rain_sum': [0.0, 5.0]
                }
            }
            mock_get.return_value = mock_response
            
            df = self.client.fetch_weather_data()
            
            # Verify API data markers
            assert 'source' in df.columns
            assert df['source'].iloc[0] == 'api'
            assert 'condition' in df.columns
            assert df['condition'].iloc[0] == 'API_Data'
    
    def test_live_disabled_uses_csv(self):
        """Test that disabling live weather uses CSV directly"""
        config.USE_LIVE_WEATHER = False
        
        df = self.client.fetch_weather_data()
        
        # Should use CSV without trying API
        assert 'source' in df.columns
        assert df['source'].iloc[0] == 'csv'
        assert df['condition'].iloc[0] == 'CSV_Fallback'
    
    def test_strict_mode_no_fallback(self):
        """Test strict mode raises error when API fails"""
        config.USE_LIVE_WEATHER = True
        config.ALLOW_CSV_FALLBACK = False
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API unavailable")
            
            with pytest.raises(PipelineError, match="CSV fallback is disabled"):
                self.client.fetch_weather_data()
    
    def test_interactive_fallback_pending(self):
        """Test interactive mode raises FallbackPending"""
        config.USE_LIVE_WEATHER = True
        config.ALLOW_CSV_FALLBACK = True
        config.INTERACTIVE_FALLBACK_PROMPT = True
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API unavailable")
            
            with pytest.raises(FallbackPending, match="user approval required"):
                self.client.fetch_weather_data()
    
    def test_silent_fallback(self):
        """Test silent fallback when API fails"""
        config.USE_LIVE_WEATHER = True
        config.ALLOW_CSV_FALLBACK = True
        config.INTERACTIVE_FALLBACK_PROMPT = False
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API unavailable")
            
            df = self.client.fetch_weather_data()
            
            # Should use CSV fallback
            assert 'source' in df.columns
            assert df['source'].iloc[0] == 'csv'
            assert df['condition'].iloc[0] == 'CSV_Fallback'
    
    def test_request_csv_fallback_function(self):
        """Test the request_csv_fallback function"""
        df = request_csv_fallback()
        
        # Verify CSV fallback markers
        assert 'source' in df.columns
        assert df['source'].iloc[0] == 'csv'
        assert df['condition'].iloc[0] == 'CSV_Fallback'
        assert not df.empty