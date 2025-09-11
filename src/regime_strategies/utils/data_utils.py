"""
Data utility functions for the regime strategies system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from loguru import logger


def validate_data(data: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate data integrity and structure.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data to validate
    required_columns : List[str], optional
        List of required columns
    
    Returns:
    --------
    bool
        True if data passes validation
    """
    if data.empty:
        logger.error("Data is empty")
        return False
    
    if required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
    
    # Check for NaN values
    if data.isnull().any().any():
        logger.warning("Data contains NaN values")
        nan_summary = data.isnull().sum()
        nan_summary = nan_summary[nan_summary > 0]
        logger.warning(f"NaN counts by column: {nan_summary.to_dict()}")
    
    # Check date index
    if isinstance(data.index, pd.DatetimeIndex):
        if not data.index.is_monotonic_increasing:
            logger.warning("Date index is not monotonic increasing")
    
    logger.info(f"Data validation completed. Shape: {data.shape}")
    return True


def align_data(*dataframes: pd.DataFrame, method: str = 'inner') -> List[pd.DataFrame]:
    """
    Align multiple dataframes by their index.
    
    Parameters:
    -----------
    *dataframes : pd.DataFrame
        Variable number of dataframes to align
    method : str
        Join method ('inner', 'outer', 'left', 'right')
    
    Returns:
    --------
    List[pd.DataFrame]
        List of aligned dataframes
    """
    if not dataframes:
        return []
    
    if len(dataframes) == 1:
        return list(dataframes)
    
    # Find common index
    common_index = dataframes[0].index
    for df in dataframes[1:]:
        if method == 'inner':
            common_index = common_index.intersection(df.index)
        elif method == 'outer':
            common_index = common_index.union(df.index)
    
    # Align all dataframes
    aligned_dfs = []
    for df in dataframes:
        if method in ['inner', 'outer']:
            aligned_df = df.reindex(common_index)
        else:
            aligned_df = df.copy()
        aligned_dfs.append(aligned_df)
    
    logger.info(f"Aligned {len(dataframes)} dataframes using {method} join. "
                f"Result shape: {aligned_dfs[0].shape}")
    
    return aligned_dfs


def clean_returns(returns: pd.Series, 
                  winsorize_percentile: float = 0.01,
                  fill_method: str = 'forward') -> pd.Series:
    """
    Clean return data by handling outliers and missing values.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series to clean
    winsorize_percentile : float
        Percentile for winsorization (e.g., 0.01 for 1% and 99%)
    fill_method : str
        Method to fill missing values ('forward', 'backward', 'zero')
    
    Returns:
    --------
    pd.Series
        Cleaned return series
    """
    cleaned_returns = returns.copy()
    
    # Handle missing values
    if fill_method == 'forward':
        cleaned_returns = cleaned_returns.ffill()
    elif fill_method == 'backward':
        cleaned_returns = cleaned_returns.bfill()
    elif fill_method == 'zero':
        cleaned_returns = cleaned_returns.fillna(0)
    
    # Winsorize extreme values
    if winsorize_percentile > 0:
        lower_bound = cleaned_returns.quantile(winsorize_percentile)
        upper_bound = cleaned_returns.quantile(1 - winsorize_percentile)
        
        cleaned_returns = cleaned_returns.clip(lower=lower_bound, upper=upper_bound)
        
        n_winsorized = ((returns < lower_bound) | (returns > upper_bound)).sum()
        if n_winsorized > 0:
            logger.info(f"Winsorized {n_winsorized} extreme values")
    
    return cleaned_returns


def calculate_rolling_statistics(data: pd.Series, 
                                window: int,
                                statistics: List[str] = None) -> pd.DataFrame:
    """
    Calculate rolling statistics for a time series.
    
    Parameters:
    -----------
    data : pd.Series
        Input time series
    window : int
        Rolling window size
    statistics : List[str], optional
        List of statistics to calculate ('mean', 'std', 'var', 'skew', 'kurt')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling statistics
    """
    if statistics is None:
        statistics = ['mean', 'std']
    
    results = pd.DataFrame(index=data.index)
    
    for stat in statistics:
        if stat == 'mean':
            results[f'rolling_{stat}_{window}'] = data.rolling(window).mean()
        elif stat == 'std':
            results[f'rolling_{stat}_{window}'] = data.rolling(window).std()
        elif stat == 'var':
            results[f'rolling_{stat}_{window}'] = data.rolling(window).var()
        elif stat == 'skew':
            results[f'rolling_{stat}_{window}'] = data.rolling(window).skew()
        elif stat == 'kurt':
            results[f'rolling_{stat}_{window}'] = data.rolling(window).kurt()
        else:
            logger.warning(f"Unknown statistic: {stat}")
    
    return results


def resample_data(data: pd.DataFrame, 
                  frequency: str,
                  aggregation_method: Union[str, Dict[str, str]] = 'last') -> pd.DataFrame:
    """
    Resample data to different frequency.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data with datetime index
    frequency : str
        Target frequency ('D', 'W', 'M', 'Q', 'Y')
    aggregation_method : str or dict
        Aggregation method(s) for resampling
    
    Returns:
    --------
    pd.DataFrame
        Resampled data
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have datetime index for resampling")
    
    if isinstance(aggregation_method, str):
        resampled = data.resample(frequency).agg(aggregation_method)
    else:
        resampled = data.resample(frequency).agg(aggregation_method)
    
    logger.info(f"Resampled data from {len(data)} to {len(resampled)} periods "
                f"using {frequency} frequency")
    
    return resampled


def enforce_point_in_time(data: pd.DataFrame, 
                          current_date: datetime,
                          lag_days: int = 0) -> pd.DataFrame:
    """
    Enforce point-in-time data constraints to avoid look-ahead bias.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data with datetime index
    current_date : datetime
        Current decision date
    lag_days : int
        Number of days to lag data (for publication delays)
    
    Returns:
    --------
    pd.DataFrame
        Data available as of current_date with specified lag
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have datetime index")
    
    cutoff_date = current_date - timedelta(days=lag_days)
    available_data = data[data.index <= cutoff_date]
    
    logger.debug(f"Applied point-in-time filter. "
                 f"Current date: {current_date}, "
                 f"Cutoff: {cutoff_date}, "
                 f"Available periods: {len(available_data)}")
    
    return available_data


def create_feature_matrix(data: Dict[str, pd.Series], 
                          lags: List[int] = None,
                          differences: bool = True,
                          returns: bool = True) -> pd.DataFrame:
    """
    Create feature matrix from multiple time series.
    
    Parameters:
    -----------
    data : Dict[str, pd.Series]
        Dictionary of time series data
    lags : List[int], optional
        List of lag periods to include
    differences : bool
        Whether to include first differences
    returns : bool
        Whether to include percentage returns
    
    Returns:
    --------
    pd.DataFrame
        Feature matrix
    """
    if lags is None:
        lags = [1, 3, 6, 12]
    
    features = pd.DataFrame()
    
    for name, series in data.items():
        # Original series
        features[name] = series
        
        # Lagged values
        for lag in lags:
            features[f"{name}_lag_{lag}"] = series.shift(lag)
        
        # First differences
        if differences:
            features[f"{name}_diff"] = series.diff()
            for lag in lags:
                features[f"{name}_diff_lag_{lag}"] = series.diff().shift(lag)
        
        # Returns
        if returns:
            features[f"{name}_return"] = series.pct_change()
            for lag in lags:
                features[f"{name}_return_lag_{lag}"] = series.pct_change().shift(lag)
    
    logger.info(f"Created feature matrix with {features.shape[1]} features")
    
    return features