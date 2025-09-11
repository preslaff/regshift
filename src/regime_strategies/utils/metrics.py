"""
Performance metrics and financial calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats
from loguru import logger


def sharpe_ratio(returns: pd.Series, 
                 risk_free_rate: float = 0.02,
                 periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float
        Annualized Sharpe ratio
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    
    # Convert risk-free rate to same frequency
    rf_per_period = risk_free_rate / periods_per_year
    
    excess_returns = returns - rf_per_period
    
    # Annualize
    annualized_return = excess_returns.mean() * periods_per_year
    annualized_std = returns.std() * np.sqrt(periods_per_year)
    
    if annualized_std == 0:
        return 0.0
    
    return annualized_return / annualized_std


def sortino_ratio(returns: pd.Series,
                  risk_free_rate: float = 0.02,
                  periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sortino ratio.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float
        Annualized Sortino ratio
    """
    if returns.empty:
        return 0.0
    
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period
    
    # Downside deviation
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return np.inf
    
    downside_std = negative_returns.std() * np.sqrt(periods_per_year)
    if downside_std == 0:
        return 0.0
    
    annualized_return = excess_returns.mean() * periods_per_year
    
    return annualized_return / downside_std


def maximum_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and its duration.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    
    Returns:
    --------
    Tuple[float, pd.Timestamp, pd.Timestamp]
        Maximum drawdown, start date, end date
    """
    if returns.empty:
        return 0.0, None, None
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    rolling_max = cum_returns.expanding().max()
    
    # Calculate drawdown
    drawdown = (cum_returns - rolling_max) / rolling_max
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    max_dd_end = drawdown.idxmin()
    
    # Find start of maximum drawdown period
    max_dd_start = rolling_max[rolling_max.index <= max_dd_end].idxmax()
    
    return max_dd, max_dd_start, max_dd_end


def value_at_risk(returns: pd.Series, 
                  confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    confidence_level : float
        Confidence level (e.g., 0.05 for 95% VaR)
    
    Returns:
    --------
    float
        Value at Risk
    """
    if returns.empty:
        return 0.0
    
    return returns.quantile(confidence_level)


def conditional_value_at_risk(returns: pd.Series,
                              confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR).
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    confidence_level : float
        Confidence level
    
    Returns:
    --------
    float
        Conditional Value at Risk
    """
    if returns.empty:
        return 0.0
    
    var = value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()


def calmar_ratio(returns: pd.Series, 
                 periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annualized return / maximum drawdown).
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float
        Calmar ratio
    """
    if returns.empty:
        return 0.0
    
    annualized_return = returns.mean() * periods_per_year
    max_dd, _, _ = maximum_drawdown(returns)
    
    if max_dd == 0:
        return np.inf if annualized_return > 0 else 0.0
    
    return annualized_return / abs(max_dd)


def information_ratio(returns: pd.Series,
                      benchmark_returns: pd.Series) -> float:
    """
    Calculate Information ratio (active return / tracking error).
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns
    
    Returns:
    --------
    float
        Information ratio
    """
    if returns.empty or benchmark_returns.empty:
        return 0.0
    
    # Align returns
    aligned_returns = returns.align(benchmark_returns, join='inner')
    active_returns = aligned_returns[0] - aligned_returns[1]
    
    if active_returns.std() == 0:
        return 0.0
    
    return active_returns.mean() / active_returns.std()


def calculate_metrics(returns: pd.Series,
                      benchmark_returns: Optional[pd.Series] = None,
                      risk_free_rate: float = 0.02,
                      periods_per_year: int = 252,
                      confidence_level: float = 0.05) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    benchmark_returns : pd.Series, optional
        Benchmark return series
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year
    confidence_level : float
        Confidence level for VaR calculation
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of performance metrics
    """
    if returns.empty:
        logger.warning("Empty returns series provided")
        return {}
    
    metrics = {}
    
    # Basic statistics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = returns.mean() * periods_per_year
    metrics['annualized_volatility'] = returns.std() * np.sqrt(periods_per_year)
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    
    # Risk-adjusted metrics
    metrics['sharpe_ratio'] = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    metrics['sortino_ratio'] = sortino_ratio(returns, risk_free_rate, periods_per_year)
    metrics['calmar_ratio'] = calmar_ratio(returns, periods_per_year)
    
    # Drawdown metrics
    max_dd, dd_start, dd_end = maximum_drawdown(returns)
    metrics['maximum_drawdown'] = max_dd
    metrics['max_drawdown_duration'] = (dd_end - dd_start).days if dd_start and dd_end else 0
    
    # Risk metrics
    metrics['value_at_risk'] = value_at_risk(returns, confidence_level)
    metrics['conditional_var'] = conditional_value_at_risk(returns, confidence_level)
    
    # Hit rate
    metrics['win_rate'] = (returns > 0).mean()
    
    # Benchmark comparison
    if benchmark_returns is not None:
        metrics['information_ratio'] = information_ratio(returns, benchmark_returns)
        
        # Beta calculation
        aligned_returns = returns.align(benchmark_returns, join='inner')
        if len(aligned_returns[0]) > 1:
            beta, alpha, r_value, p_value, std_err = stats.linregress(
                aligned_returns[1], aligned_returns[0]
            )
            metrics['beta'] = beta
            metrics['alpha'] = alpha * periods_per_year
            metrics['r_squared'] = r_value ** 2
    
    logger.info(f"Calculated {len(metrics)} performance metrics")
    
    return metrics


def rolling_metrics(returns: pd.Series,
                    window: int,
                    metric_func,
                    **kwargs) -> pd.Series:
    """
    Calculate rolling metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    window : int
        Rolling window size
    metric_func : callable
        Metric calculation function
    **kwargs
        Additional arguments for metric function
    
    Returns:
    --------
    pd.Series
        Rolling metric values
    """
    rolling_values = []
    
    for i in range(window - 1, len(returns)):
        window_returns = returns.iloc[i - window + 1:i + 1]
        value = metric_func(window_returns, **kwargs)
        rolling_values.append(value)
    
    # Create series with appropriate index
    rolling_index = returns.index[window - 1:]
    return pd.Series(rolling_values, index=rolling_index)