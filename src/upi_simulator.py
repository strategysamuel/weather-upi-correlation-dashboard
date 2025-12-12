from datetime import timedelta
import numpy as np
import pandas as pd

def simulate_upi_data(start_date, end_date,
                     baseline_txn_count=5_000_000,
                     baseline_txn_value=800.0,
                     weekend_multiplier=0.92,
                     weekday_multiplier=1.02,
                     rain_impact_per_mm=0.03,
                     temp_value_sensitivity=0.2,
                     outlier_prob=0.08,
                     outlier_scale=1.6,
                     seed=42):
    """
    Generate realistic UPI transaction data for a given date range.
    
    Args:
        start_date: Start date for simulation
        end_date: End date for simulation
        baseline_txn_count: Base number of transactions per day
        baseline_txn_value: Base average transaction value in INR
        weekend_multiplier: Multiplier for weekend transaction counts
        weekday_multiplier: Multiplier for weekday transaction counts
        rain_impact_per_mm: Impact of rain per mm on transaction count
        temp_value_sensitivity: Temperature sensitivity for transaction values
        outlier_prob: Probability of outlier days
        outlier_scale: Scale factor for outliers
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with simulated UPI data
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    # Base trend (slight growth over time)
    base = np.linspace(0.98, 1.02, n)
    
    # Weekend/weekday patterns
    weekday = np.array([weekday_multiplier if d.weekday() < 5 else weekend_multiplier
                       for d in dates])
    
    # Random noise
    noise = rng.normal(loc=1.0, scale=0.03, size=n)
    
    # Generate transaction counts
    txn_counts = baseline_txn_count * base * weekday * noise
    
    # Generate average transaction values
    avg_values = baseline_txn_value * (1.0 + rng.normal(0, 0.03, size=n))
    
    # Add outliers
    outlier_flags = rng.random(n) < outlier_prob
    if outlier_flags.any():
        signs = rng.choice([0.5, 1.6, 0.7, 2.0], size=n)
        txn_counts[outlier_flags] *= signs[outlier_flags] * outlier_scale
        avg_values[outlier_flags] *= (1 + (signs[outlier_flags] - 1) * 0.5)
    
    # Round and convert to appropriate types
    txn_counts = np.round(txn_counts).astype(int)
    avg_values = np.round(avg_values, 2)
    
    df = pd.DataFrame({
        'date': dates,
        'total_upi_txn_count': txn_counts,
        'avg_txn_value_inr': avg_values,
        'notes': [f"Simulated UPI (seed={seed})"] * n,
        'source': ['simulated'] * n
    })
    
    return df

def apply_weather_influence(upi_df, weather_df,
                           rain_impact_per_mm=0.03,
                           temp_value_sensitivity=0.2):
    """
    Apply weather influence to UPI transaction data.
    
    Args:
        upi_df: UPI DataFrame
        weather_df: Weather DataFrame
        rain_impact_per_mm: Impact of rain per mm on transaction count
        temp_value_sensitivity: Temperature sensitivity for transaction values
    
    Returns:
        Merged DataFrame with weather influence applied
    """
    upi = upi_df.copy()
    w = weather_df.copy()
    
    # Normalize dates
    upi['date'] = pd.to_datetime(upi['date']).dt.normalize()
    w['date'] = pd.to_datetime(w['date']).dt.normalize()
    
    # Merge weather and UPI data
    merged = pd.merge(w, upi, on='date', how='left')
    
    # Apply rain impact on transaction count
    rain = merged['rain_mm'].fillna(0.0)
    multiplier = 1.0 - (rain * rain_impact_per_mm)
    multiplier = multiplier.clip(lower=0.5, upper=1.2)
    merged['total_upi_txn_count'] = (merged['total_upi_txn_count'] * multiplier).round().astype('Int64')
    
    # Apply temperature impact on transaction value
    temp = merged['avg_temp_c'].fillna(0.0)
    merged['avg_txn_value_inr'] = (merged['avg_txn_value_inr'] + temp * temp_value_sensitivity).round(2)
    
    # Ensure source column is preserved
    merged['source'] = merged.get('source', 'simulated')
    
    return merged