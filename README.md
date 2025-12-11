# Weather-UPI Correlation Dashboard

A comprehensive data analytics system that combines live weather data from Mumbai (fetched via Model Context Protocol from Open-Meteo API) with UPI (Unified Payments Interface) transaction data from India to discover correlations between weather patterns and digital payment behaviors. This project demonstrates the power of MCP for external data integration and provides insights into how environmental factors influence financial transactions.

## ⚙️ Configuration

### Live-First Behavior Settings

The system can be configured via `config.py` or environment variables:

```python
# Live-first weather data configuration
USE_LIVE_WEATHER = True                    # Enable/disable live API attempts
ALLOW_CSV_FALLBACK = True                  # Allow fallback to CSV data
INTERACTIVE_FALLBACK_PROMPT = True         # Prompt user for fallback approval
LIVE_FETCH_RETRY_COUNT = 3                 # Number of API retry attempts
LIVE_FETCH_RETRY_DELAY_SEC = 3             # Delay between retries (seconds)
```

### Operation Modes

| Mode | Live API | CSV Fallback | User Prompt | Use Case |
|------|----------|--------------|-------------|----------|
| **Default** | ✅ | ✅ | ✅ | Interactive development |
| **Live-only** | ✅ | ❌ | ❌ | Production with reliable API |
| **CSV-only** | ❌ | ✅ | ❌ | Offline development/testing |
| **Silent fallback** | ✅ | ✅ | ❌ | Automated pipelines |

### Environment Variables

```bash
# Force live-only mode
export LIVE_ONLY=true

# Override in Docker
docker run -e LIVE_ONLY=true weather-upi-insights
```

## 🏗️ Architecture Overview

The system follows a modular pipeline architecture with live-first data fetching and intelligent fallback:

```
┌─────────────────┐    ┌─────────────────┐
│ Open-Meteo API  │    │   UPI CSV       │
│ (Live-First)    │    │   Data Files    │
└─────────┬───────┘    └─────────┬───────┘
          │ (primary)            │
          ▼                      │
┌─────────────────┐              │
│ Retry Logic     │              │
│ (3 attempts)    │              │
└─────────┬───────┘              │
          │ (on failure)         │
          ▼                      │
┌─────────────────┐              │
│ User Approval   │              │
│ (Interactive)   │              │
└─────────┬───────┘              │
          │ (approved)           │
          ▼                      │
┌─────────────────┐              │
│ Weather CSV     │              │
│ (Fallback)      │              │
└─────────┬───────┘              │
          │                      │
          └──────┬─────────────────┘
                 │
         ┌───────▼────────┐
         │ Data Pipeline  │
         │ ┌────────────┐ │
         │ │ Validate   │ │
         │ │ Transform  │ │
         │ │ Merge      │ │
         │ │ Analyze    │ │
         │ └────────────┘ │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │ Dashboard      │
         │ - Live Status  │
         │ - Auto-refresh │
         │ - Data Source  │
         │   Badges       │
         └────────────────┘
                 │
         ┌───────▼────────┐
         │ Streamlit      │
         │ Dashboard      │
         │ - Visualizations│
         │ - Correlations │
         │ - Insights     │
         └────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Internet connection for MCP weather API calls
- Required CSV data files (provided)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd weather-upi-dashboard
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify data files are present:**
```bash
ls -la *.csv
# Should show:
# upi_transactions_india_2024_11_synthetic.csv
# weather_mumbai_2024_11_synthetic.csv
```

### Execution

#### Option 1: Complete Pipeline + Dashboard
```bash
# Run the complete data pipeline
python main.py

# Launch the interactive dashboard
streamlit run src/dashboard.py
```

#### Option 2: Dashboard Only (using existing data)
```bash
# If you already have processed data in output/
streamlit run src/dashboard.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Data Sources and Schemas

### Weather Data (via MCP)
**Primary Source:** Open-Meteo API (https://api.open-meteo.com)
- **Location:** Mumbai, India (19.07°N, 72.88°E)
- **Parameters:** Daily temperature, humidity, precipitation
- **Fallback:** Local CSV file when API unavailable

**Schema:**
```python
{
    'date': 'datetime64[ns]',      # Date of observation
    'city': 'string',              # City name (Mumbai)
    'avg_temp_c': 'float64',       # Average temperature in Celsius
    'humidity_pct': 'float64',     # Relative humidity percentage
    'rain_mm': 'float64',          # Precipitation in millimeters
    'condition': 'string'          # Weather condition description
}
```

### UPI Transaction Data
**Source:** Synthetic dataset based on real UPI transaction patterns
- **Coverage:** India-wide UPI transactions for November 2024
- **Granularity:** Daily aggregated transaction metrics

**Schema:**
```python
{
    'date': 'datetime64[ns]',           # Transaction date
    'total_upi_txn_count': 'int64',     # Total number of transactions
    'avg_txn_value_inr': 'float64',     # Average transaction value in INR
    'notes': 'string'                   # Additional notes/metadata
}
```

### Merged Analytics Dataset
**Output:** Combined weather and UPI data with analytics enhancements

**Schema:**
```python
{
    'date': 'datetime64[ns]',           # Common date key
    'total_upi_txn_count': 'int64',     # UPI transaction count
    'avg_txn_value_inr': 'float64',     # Average transaction value
    'avg_temp_c': 'float64',            # Temperature data
    'humidity_pct': 'float64',          # Humidity data
    'rain_mm': 'float64',               # Rainfall data
    'condition': 'string',              # Weather condition
    'txn_volume_outlier': 'bool',       # Transaction anomaly flag
    'weather_outlier': 'bool',          # Weather anomaly flag
    'temp_z_score': 'float64',          # Temperature z-score
    'rain_z_score': 'float64',          # Rainfall z-score
    'txn_z_score': 'float64'            # Transaction z-score
}
```

## 🔧 Pipeline Stages

### Stage 1: Data Fetching (MCP Integration)
**Module:** `src/weather_api.py`
- Fetches live weather data from Open-Meteo API via MCP
- Implements intelligent fallback to local CSV
- Handles API rate limiting and error responses
- Normalizes API response to standard schema

**MCP Implementation Details:**
```python
# Example MCP weather API call
client = WeatherAPIClient()
weather_data = client.fetch_weather_data(
    start_date="2024-11-01", 
    end_date="2024-11-30"
)
```

### Stage 2: Data Loading
**Module:** `src/data_loader.py`
- Loads UPI transaction data from CSV files
- Automatic encoding detection and fallback
- Comprehensive error handling for malformed files
- Data type standardization and validation

### Stage 3: Data Validation
**Module:** `src/data_validator.py`
- Validates required columns presence
- Checks date ranges and formats
- Ensures non-negative numerical values
- Detects and reports missing values
- Generates detailed validation reports

### Stage 4: Data Transformation
**Module:** `src/data_transformer.py`
- Standardizes column names across datasets
- Normalizes date formats for consistent merging
- Performs date-based dataset merging
- Preserves data integrity during transformations

### Stage 5: Analytics Engine
**Module:** `src/analytics_engine.py`
- Computes Pearson correlation coefficients
- Generates correlation matrices
- Performs statistical significance testing
- Detects outliers using z-score analysis (2σ threshold)
- Creates anomaly flags and summaries

### Stage 6: Visualization Dashboard
**Module:** `src/dashboard.py`
- Interactive Streamlit web application
- Time series visualizations for both datasets
- Correlation heatmaps and scatter plots
- Automated insight generation
- Filtering and date range selection

## 📈 Key Features

### Model Context Protocol (MCP) Integration
- **Real-time Data:** Fetches live weather data from Open-Meteo API
- **Resilient Design:** Automatic fallback to local data when API unavailable
- **Error Handling:** Comprehensive handling of network and API failures
- **Data Normalization:** Converts API responses to standardized format

### Statistical Analysis
- **Correlation Analysis:** Pearson correlation between weather and transaction variables
- **Significance Testing:** P-value calculations for statistical validity
- **Anomaly Detection:** Z-score based outlier identification
- **Trend Analysis:** Time series pattern recognition

### Interactive Dashboard
- **Real-time Visualizations:** Dynamic charts and graphs
- **Correlation Heatmaps:** Visual representation of variable relationships
- **Insight Generation:** Automated discovery and explanation of patterns
- **User Controls:** Date filtering and variable selection

### Data Quality Assurance
- **Comprehensive Validation:** Multi-stage data quality checks
- **Error Reporting:** Detailed validation reports with specific issues
- **Data Integrity:** Preservation of data relationships during processing
- **Fallback Mechanisms:** Graceful handling of data availability issues

## 🧪 Testing Framework

The project includes comprehensive testing using both unit tests and property-based testing:

### Property-Based Testing
Uses Hypothesis framework for automated test case generation:
- **Correlation Validation:** Ensures correlation values are within [-1, 1]
- **Data Integrity:** Validates merge operations preserve data
- **Anomaly Detection:** Tests outlier identification accuracy
- **API Response Validation:** Verifies MCP API response handling

### Unit Testing
Traditional test cases for specific functionality:
- **Module Integration:** Tests component interactions
- **Error Handling:** Validates error conditions and recovery
- **Data Transformations:** Ensures correct data processing
- **Dashboard Components:** Tests visualization rendering

## 📁 Project Structure

```
weather-upi-dashboard/
├── 📁 src/                          # Source code modules
│   ├── 🐍 weather_api.py            # MCP weather API integration
│   ├── 🐍 data_loader.py            # CSV data loading utilities
│   ├── 🐍 data_validator.py         # Data quality validation
│   ├── 🐍 data_transformer.py       # Data transformation and merging
│   ├── 🐍 analytics_engine.py       # Statistical analysis engine
│   ├── 🐍 dashboard.py              # Streamlit dashboard application
│   └── 🐍 __init__.py               # Package initialization
├── 📁 tests/                        # Test suite
│   ├── 🧪 test_weather_api.py       # MCP API tests
│   ├── 🧪 test_data_loader.py       # Data loading tests
│   ├── 🧪 test_data_validator.py    # Validation tests
│   ├── 🧪 test_data_transformer.py  # Transformation tests
│   ├── 🧪 test_analytics_engine.py  # Analytics tests
│   ├── 🧪 test_dashboard.py         # Dashboard tests
│   ├── 🧪 test_pipeline.py          # End-to-end pipeline tests
│   └── 🧪 __init__.py               # Test package initialization
├── 📁 output/                       # Generated output files
│   ├── 📊 merged_weather_upi_data.csv    # Merged dataset
│   ├── 📊 analytics.csv             # Analytics results
│   ├── 📄 validation_report.txt     # Data validation report
│   ├── 📄 pipeline_summary.txt      # Pipeline execution summary
│   └── 📄 pipeline.log              # Detailed execution log
├── 📁 .kiro/                        # Kiro specification files
│   └── 📁 specs/weather-upi-dashboard/
│       ├── 📄 requirements.md       # Project requirements
│       ├── 📄 design.md             # System design document
│       └── 📄 tasks.md              # Implementation tasks
├── 🐍 main.py                       # Main pipeline orchestrator
├── ⚙️ config.py                     # Configuration settings
├── 📄 requirements.txt              # Python dependencies
├── 📊 upi_transactions_india_2024_11_synthetic.csv  # UPI data
├── 📊 weather_mumbai_2024_11_synthetic.csv          # Weather fallback data
└── 📖 README.md                     # This documentation
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Override default API settings
export WEATHER_API_TIMEOUT=30
export OUTLIER_THRESHOLD=2.5
export DASHBOARD_PORT=8502
```

### Configuration File (`config.py`)
Key settings that can be modified:
```python
# Mumbai coordinates for weather API
MUMBAI_LAT = 19.07
MUMBAI_LON = 72.88

# Analysis parameters
OUTLIER_THRESHOLD = 2.0  # Standard deviations for outlier detection
MIN_CORRELATION_SIGNIFICANCE = 0.05  # p-value threshold

# Dashboard configuration
DASHBOARD_PORT = 8501
```

## 📊 Output Files

The pipeline generates several output files in the `output/` directory:

1. **`merged_weather_upi_data.csv`** - Combined weather and UPI dataset
2. **`analytics.csv`** - Enhanced dataset with correlation analysis and anomaly flags
3. **`validation_report.txt`** - Detailed data quality validation results
4. **`pipeline_summary.txt`** - Executive summary of pipeline execution
5. **`pipeline.log`** - Detailed execution log with timestamps

## 🚨 Troubleshooting

### Common Issues

**1. MCP API Connection Failures**
```bash
# Check internet connectivity
ping api.open-meteo.com

# Verify fallback data exists
ls -la weather_mumbai_2024_11_synthetic.csv
```

**2. Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**3. CSV File Issues**
```bash
# Check file encoding
file -bi *.csv

# Verify file structure
head -5 upi_transactions_india_2024_11_synthetic.csv
```

**4. Dashboard Not Loading**
```bash
# Check if port is available
netstat -an | grep 8501

# Try alternative port
streamlit run src/dashboard.py --server.port 8502
```

### Error Recovery

The system includes comprehensive error handling:
- **API Failures:** Automatic fallback to local weather data
- **Data Issues:** Detailed validation reports with specific error locations
- **Processing Errors:** Graceful degradation with informative error messages
- **Dashboard Errors:** Fallback visualizations when data is unavailable

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest hypothesis black flake8

# Run tests
python -m pytest tests/ -v

# Run property-based tests
python -m pytest tests/ -k "property" -v

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

### Adding New Features
1. Update requirements in `.kiro/specs/weather-upi-dashboard/requirements.md`
2. Modify design in `.kiro/specs/weather-upi-dashboard/design.md`
3. Add implementation tasks to `.kiro/specs/weather-upi-dashboard/tasks.md`
4. Implement code with comprehensive tests
5. Update documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Open-Meteo API** for providing free weather data
- **Model Context Protocol** for enabling seamless external data integration
- **Streamlit** for the interactive dashboard framework
- **Hypothesis** for property-based testing capabilities

## 📞 Support

For questions, issues, or contributions:
1. Check the troubleshooting section above
2. Review the detailed logs in `output/pipeline.log`
3. Examine validation reports in `output/validation_report.txt`
4. Open an issue with detailed error information and steps to reproduce