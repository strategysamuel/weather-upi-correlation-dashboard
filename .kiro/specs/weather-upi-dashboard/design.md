# Design Document

## Overview

The Weather-UPI Correlation Dashboard is a comprehensive data analytics system that combines live weather data (via Model Context Protocol) with UPI transaction datasets to discover meaningful correlations and insights. The system demonstrates modern data integration patterns using MCP for external API communication and follows a modular pipeline architecture with distinct components for data fetching, loading, validation, transformation, analysis, and visualization.

The core workflow fetches live weather data from Open-Meteo API via MCP, processes local UPI transaction CSV data, merges them on date, computes statistical correlations, detects anomalies, and presents findings through an interactive Streamlit dashboard. The system includes intelligent fallback to local weather CSV data when API calls fail.

## Architecture

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐
│ Open-Meteo API  │    │   UPI CSV       │
│ (via MCP)       │    │                 │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      │
┌─────────────────┐              │
│ Weather Fallback│              │
│ (Local CSV)     │              │
└─────────┬───────┘              │
          │                      │
          └──────┬─────────────────┘
                 │
         ┌───────▼────────┐
         │ Data Pipeline  │
         │ - Fetch (MCP)  │
         │ - Load         │
         │ - Validate     │
         │ - Transform    │
         │ - Merge        │
         │ - Analyze      │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │ Output Files   │
         │ - Merged CSV   │
         │ - Analytics    │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │ Streamlit      │
         │ Dashboard      │
         └────────────────┘
```

## Components and Interfaces

### MCP Weather API Module (`weather_api.py`)
- **Purpose**: Fetch live weather data from Open-Meteo API via Model Context Protocol
- **Interface**: `fetch_weather_data()`, `parse_weather_response()`, `fallback_to_csv()`
- **Input**: Mumbai coordinates, date range
- **Output**: Normalized weather DataFrame or fallback to local CSV

### Data Loading Module (`data_loader.py`)
- **Purpose**: Load and initial validation of CSV files
- **Interface**: `load_upi_data()`, `load_weather_fallback()`
- **Input**: CSV file paths
- **Output**: Pandas DataFrames with validated structure

### Data Validation Module (`data_validator.py`)
- **Purpose**: Comprehensive data quality checks
- **Interface**: `validate_weather_data()`, `validate_upi_data()`, `generate_validation_report()`
- **Input**: Raw DataFrames
- **Output**: Validation results and cleaned DataFrames

### Data Transformation Module (`data_transformer.py`)
- **Purpose**: Standardize column names and data formats
- **Interface**: `standardize_columns()`, `normalize_dates()`, `merge_datasets()`
- **Input**: Validated DataFrames
- **Output**: Standardized and merged DataFrame

### Analytics Engine (`analytics_engine.py`)
- **Purpose**: Compute correlations and detect anomalies
- **Interface**: `compute_correlations()`, `detect_outliers()`, `generate_insights()`
- **Input**: Merged DataFrame
- **Output**: Analytics results and enhanced DataFrame with anomaly flags

### Dashboard Module (`dashboard.py`)
- **Purpose**: Interactive visualization and insights presentation
- **Interface**: Streamlit web application
- **Input**: Analytics results and processed data
- **Output**: Interactive web dashboard

## Data Models

### Weather Data Schema
```python
{
    'date': 'datetime64[ns]',
    'city': 'string',
    'avg_temp_c': 'float64',
    'humidity_pct': 'float64', 
    'rain_mm': 'float64',
    'condition': 'string'
}
```

### UPI Transaction Schema
```python
{
    'date': 'datetime64[ns]',
    'total_upi_txn_count': 'int64',
    'avg_txn_value_inr': 'float64',
    'notes': 'string'
}
```

### Merged Analytics Schema
```python
{
    'date': 'datetime64[ns]',
    'total_upi_txn_count': 'int64',
    'avg_txn_value_inr': 'float64',
    'avg_temp_c': 'float64',
    'humidity_pct': 'float64',
    'rain_mm': 'float64',
    'condition': 'string',
    'txn_volume_outlier': 'bool',
    'weather_outlier': 'bool',
    'temp_z_score': 'float64',
    'rain_z_score': 'float64',
    'txn_z_score': 'float64'
}
```
## Cor
rectness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Column Validation Completeness
*For any* CSV dataset with missing required columns, the Data_Validator should correctly identify and report all missing columns
**Validates: Requirements 1.1**

### Property 2: Date Range Validation
*For any* dataset containing date values, the Data_Validator should correctly identify dates outside valid ranges and invalid date formats
**Validates: Requirements 1.2**

### Property 3: Non-negative Value Validation
*For any* dataset containing numerical values, the Data_Validator should correctly flag all negative transaction counts and weather measurements
**Validates: Requirements 1.3**

### Property 4: Null Value Detection Accuracy
*For any* dataset with missing values, the Data_Validator should accurately count and locate all null values across all columns
**Validates: Requirements 1.4**

### Property 5: Validation Report Generation
*For any* validation process, the Analytics_Pipeline should generate a complete validation report with clear pass/fail status for all checks
**Validates: Requirements 1.5**

### Property 6: Column Standardization Consistency
*For any* dataset with varying column naming conventions, the Analytics_Pipeline should standardize all column names to a consistent format
**Validates: Requirements 2.1**

### Property 7: Date-based Merge Completeness
*For any* two datasets with overlapping date ranges, the merge operation should preserve all records with matching dates without data loss
**Validates: Requirements 2.2, 2.4**

### Property 8: Date Format Standardization
*For any* datasets with different date formats, the Analytics_Pipeline should convert all dates to identical formats before merging
**Validates: Requirements 2.3**

### Property 9: Merged Dataset Structure
*For any* successful merge operation, the output dataset should contain all relevant columns from both input datasets
**Validates: Requirements 2.5**

### Property 10: Correlation Calculation Accuracy
*For any* numerical dataset, the Correlation_Engine should compute mathematically correct Pearson correlation coefficients between all weather and transaction variables
**Validates: Requirements 3.1, 3.2, 3.3**

### Property 11: Correlation Matrix Completeness
*For any* correlation analysis, the Correlation_Engine should generate a complete matrix containing all possible weather-payment variable relationships
**Validates: Requirements 3.4**

### Property 12: Statistical Significance Testing
*For any* correlation calculation, the Correlation_Engine should provide corresponding p-values for statistical significance assessment
**Validates: Requirements 3.5**

### Property 13: Outlier Detection Accuracy
*For any* numerical dataset, the Anomaly_Detector should correctly identify values exceeding 2 standard deviations from the mean
**Validates: Requirements 4.1, 4.2**

### Property 14: Outlier Flagging Consistency
*For any* anomaly detection process, detected outliers should be consistently marked in the output dataset
**Validates: Requirements 4.3**

### Property 15: Z-score Calculation Correctness
*For any* numerical variable, the Anomaly_Detector should compute mathematically accurate z-scores
**Validates: Requirements 4.4**

### Property 16: Anomaly Summary Generation
*For any* outlier detection process, the Anomaly_Detector should generate a comprehensive summary of all detected anomalies
**Validates: Requirements 4.5**

### Property 17: Insight Generation Accuracy
*For any* correlation analysis results, the Dashboard_System should generate meaningful summary insights that accurately reflect the statistical findings
**Validates: Requirements 5.4**

### Property 18: Correlation Value Range Validation
*For any* correlation calculation, all computed correlation coefficients should fall within the mathematically valid range of -1 to 1
**Validates: Requirements 7.4**

### Property 19: API Response Validation
*For any* successful MCP API call to Open-Meteo, the response should contain valid JSON with required weather fields (temperature, rainfall, humidity)
**Validates: Requirements 2.2, 2.3**

### Property 20: API Fallback Mechanism
*For any* failed MCP API call, the system should automatically fallback to local CSV weather data without data loss
**Validates: Requirements 2.4**

## Error Handling

The system implements comprehensive error handling across all components:

### Data Loading Errors
- **File Not Found**: Graceful handling with clear error messages when CSV files are missing
- **Malformed CSV**: Detection and reporting of CSV parsing errors with line numbers
- **Encoding Issues**: Automatic encoding detection and fallback mechanisms

### Data Validation Errors
- **Schema Mismatches**: Clear reporting when required columns are missing or have incorrect types
- **Data Quality Issues**: Detailed reporting of validation failures with specific error locations
- **Range Violations**: Identification and flagging of values outside expected ranges

### Processing Errors
- **Memory Limitations**: Chunked processing for large datasets to prevent memory overflow
- **Computation Failures**: Robust handling of mathematical operations that may fail (division by zero, etc.)
- **Merge Conflicts**: Detection and resolution of date format mismatches during merging

### Dashboard Errors
- **Data Loading Failures**: Graceful degradation when analytics data is unavailable
- **Visualization Errors**: Fallback displays when chart generation fails
- **User Input Validation**: Proper handling of invalid date ranges or filter selections

## Testing Strategy

The system employs a dual testing approach combining unit tests and property-based tests:

### Unit Testing Approach
- **Specific Examples**: Test known input-output pairs for each transformation function
- **Edge Cases**: Test boundary conditions like empty datasets, single-row data, extreme values
- **Integration Points**: Test component interactions and data flow between modules
- **Error Conditions**: Verify proper error handling for various failure scenarios

### Property-Based Testing Approach
- **Framework**: Uses Hypothesis for Python to generate test cases automatically
- **Test Configuration**: Each property test runs a minimum of 100 iterations with random inputs
- **Universal Properties**: Verify correctness properties hold across all valid inputs
- **Data Generators**: Smart generators that create realistic weather and transaction data within valid ranges
- **MCP Testing**: Mock API responses to test various success and failure scenarios

### Property Test Implementation Requirements
- Each property-based test must include a comment referencing the design document property
- Test format: `# Feature: weather-upi-dashboard, Property X: [property description]`
- All correlation calculations must be validated to produce values between -1 and 1
- Merge operations must preserve data integrity across all test scenarios
- Anomaly detection must correctly identify outliers using statistical methods

### Test Coverage Goals
- **Unit Tests**: Cover specific functionality and integration points
- **Property Tests**: Verify universal correctness properties across input space
- **End-to-End Tests**: Validate complete pipeline execution with sample datasets
- **Performance Tests**: Ensure reasonable execution times for expected data volumes