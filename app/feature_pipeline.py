
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy.fft import fft, fftfreq
import holidays

def create_features_new(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw Hanoi weather data into a feature-rich DataFrame
    based on the new data processing pipeline from the final notebook.
    """
    df = df_raw.copy()

    # --- 1. Initial Cleaning & Type Conversion ---
    if 'datetime' not in df.columns:
        return pd.DataFrame()
        
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    initial_cols_to_drop = [
        'name', 'severerisk', 'preciptype', 'snow', 'snowdepth',
        'address', 'resolvedAddress', 'source', 'latitude', 'longitude', 'stations',
        'description'
    ]
    df.drop(columns=[col for col in initial_cols_to_drop if col in df.columns], inplace=True)

    if 'conditions' in df.columns:
        conditions_dummies = df['conditions'].str.get_dummies(sep=', ')
        df = pd.concat([df, conditions_dummies], axis=1)
        df.drop('conditions', axis=1, inplace=True)

    if 'icon' in df.columns:
        icon_dummies = pd.get_dummies(df['icon'], prefix='icon')
        df = pd.concat([df, icon_dummies], axis=1)
        df.drop('icon', axis=1, inplace=True)

    # --- 2. Feature Engineering ---
    
    # Date & Time Features
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Domain-Specific & Holiday Features
    df['sunrise'] = pd.to_datetime(df['sunrise'])
    df['sunset'] = pd.to_datetime(df['sunset'])
    df['daylight_duration_sec'] = (df['sunset'] - df['sunrise']).dt.total_seconds()
    
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Fall'
    df['season'] = df['month'].apply(get_season)
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)

    vn_holidays = holidays.VN(years=np.arange(df['year'].min(), df['year'].max() + 1))
    df['is_holiday'] = df.index.isin(vn_holidays).astype(int)

    # Time Series Decomposition (STL)
    temp_series = df['temp'].dropna()
    if len(temp_series) >= 365 * 2:
        stl_yearly = STL(temp_series, period=365, seasonal=13, robust=True)
        result_yearly = stl_yearly.fit()
        df['temp_trend_yearly'] = result_yearly.trend
        df['temp_seasonal_yearly'] = result_yearly.seasonal
        df['temp_resid_yearly'] = result_yearly.resid
        stl_cols = ['temp_trend_yearly', 'temp_seasonal_yearly', 'temp_resid_yearly']
        df[stl_cols] = df[stl_cols].bfill().ffill()

    # Lag, Difference, and Rolling Window Features
    new_features = []
    lag_cols = ['temp', 'humidity', 'windspeed', 'cloudcover', 'precip']
    lags = [1, 2, 3, 4, 5, 7, 8, 9, 30, 365]
    diff_periods = [1, 7]
    windows = [3, 7, 14, 30]
    ewm_spans = [3, 7, 14, 30]

    for col in lag_cols:
        for lag in lags:
            new_features.append(df[col].shift(lag).rename(f'{col}_lag_{lag}'))
        for p in diff_periods:
            new_features.append(df[col].diff(periods=p).rename(f'{col}_diff_{p}'))
        shifted = df[col].shift(1)
        for w in windows:
            new_features.append(shifted.rolling(window=w, min_periods=1).mean().rename(f'{col}_roll_mean_{w}'))
            new_features.append(shifted.rolling(window=w, min_periods=1).std().rename(f'{col}_roll_std_{w}'))
            if col in ['temp', 'precip', 'windspeed']:
                new_features.append(shifted.rolling(window=w, min_periods=1).min().rename(f'{col}_roll_min_{w}'))
                new_features.append(shifted.rolling(window=w, min_periods=1).max().rename(f'{col}_roll_max_{w}'))
        for span in ewm_spans:
            new_features.append(shifted.ewm(span=span, adjust=False).mean().rename(f'{col}_ewm_{span}'))

    # Interaction & Advanced Domain Features
    df['interaction_lag1_summer'] = df.get('temp_lag_1', 0) * df.get('season_Summer', 0)
    df['interaction_lag1_weekend'] = df.get('temp_lag_1', 0) * df.get('is_weekend', 0)
    df['interaction_wind_clearsky_effect'] = df.get('windspeed', 0) * (1 - df.get('cloudcover', 0) / 100.0)
    df['effective_radiation'] = df.get('solarradiation', 0) * (1 - df.get('cloudcover', 0) / 100)
    df['humidity_temp_interact'] = df.get('humidity', 0) * df.get('temp', 0)
    df['temp_range'] = df['tempmax'] - df['tempmin']
    df['dew_point_depression'] = df['temp'] - df['dew']

    # Cyclical & Vector Features
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['winddir_sin'] = np.sin(np.deg2rad(df['winddir']))
    df['winddir_cos'] = np.cos(np.deg2rad(df['winddir']))
    df['moonphase_sin'] = np.sin(2 * np.pi * df['moonphase'])
    df['moonphase_cos'] = np.cos(2 * np.pi * df['moonphase'])
    df['wind_vector_ns'] = df['windspeed'] * df['winddir_cos']
    df['wind_vector_ew'] = df['windspeed'] * df['winddir_sin']
    
    # Fourier Features from STL Residual
    if 'temp_resid_yearly' in df.columns:
        residual = df['temp_resid_yearly'].dropna().values
        n = len(residual)
        if n > 0:
            fft_values = fft(residual)
            frequencies = fftfreq(n)
            amplitudes = np.abs(fft_values)
            top_freq_indices = np.argsort(amplitudes)[::-1][1:4]
            top_frequencies = frequencies[top_freq_indices]
            df_len = len(df)
            for i, freq in enumerate(top_frequencies):
                df[f'fft_sin_freq_{i+1}'] = np.sin(2 * np.pi * freq * np.arange(df_len))
                df[f'fft_cos_freq_{i+1}'] = np.cos(2 * np.pi * freq * np.arange(df_len))
    
    df = pd.concat([df] + new_features, axis=1)

    # --- 3. Final Cleanup ---
    final_cols_to_drop = [
        'sunrise', 'sunset', 'winddir', 'moonphase', 'season',
        'feelslike', 'feelslikemax', 'feelslikemin'
    ]
    df.drop(columns=[col for col in final_cols_to_drop if col in df.columns], inplace=True)
    
    df.bfill(inplace=True)
    df.ffill(inplace=True)

    return df