"""
Economic data provider for macroeconomic indicators.
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import date, datetime
from pathlib import Path
import requests
from loguru import logger

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logger.warning("fredapi not available. Economic data will be limited.")

from ..config import Config


class EconomicDataProvider:
    """
    Provider for economic data from various sources.
    """
    
    def __init__(self, config: Config):
        """
        Initialize economic data provider.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        
        # Initialize FRED client
        self.fred_client = None
        if FRED_AVAILABLE and config.data.fred_api_key:
            try:
                self.fred_client = Fred(api_key=config.data.fred_api_key)
                self.logger.info("FRED client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize FRED client: {e}")
        
        # Data cache
        self._cache = {}
    
    def get_cpi_data(self, 
                     start_date: date,
                     end_date: date,
                     use_cache: bool = True) -> pd.DataFrame:
        """
        Get Consumer Price Index data.
        
        Parameters:
        -----------
        start_date : date
            Start date
        end_date : date
            End date
        use_cache : bool
            Whether to use cached data
        
        Returns:
        --------
        pd.DataFrame
            CPI data
        """
        cache_key = f"cpi_{start_date}_{end_date}"
        
        if use_cache and cache_key in self._cache:
            self.logger.debug("Using cached CPI data")
            return self._cache[cache_key].copy()
        
        self.logger.info(f"Loading CPI data from {start_date} to {end_date}")
        
        cpi_data = None
        
        # Try FRED first
        if self.fred_client:
            try:
                cpi_series = self.fred_client.get_series(
                    self.config.data.cpi_series,
                    start=start_date,
                    end=end_date
                )
                cpi_data = pd.DataFrame({'CPI': cpi_series})
                self.logger.info(f"Loaded CPI data from FRED. Shape: {cpi_data.shape}")
                
            except Exception as e:
                self.logger.error(f"Failed to load CPI from FRED: {e}")
        
        # Fallback to local data or synthetic data
        if cpi_data is None:
            cpi_data = self._load_local_cpi_data(start_date, end_date)
        
        if cpi_data is None:
            cpi_data = self._generate_synthetic_cpi_data(start_date, end_date)
        
        # Forward fill missing values
        cpi_data = cpi_data.ffill()
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = cpi_data.copy()
        
        return cpi_data
    
    def get_cli_data(self,
                     start_date: date,
                     end_date: date,
                     use_cache: bool = True) -> pd.DataFrame:
        """
        Get Composite Leading Indicator data.
        
        Parameters:
        -----------
        start_date : date
            Start date
        end_date : date
            End date
        use_cache : bool
            Whether to use cached data
        
        Returns:
        --------
        pd.DataFrame
            CLI data
        """
        cache_key = f"cli_{start_date}_{end_date}"
        
        if use_cache and cache_key in self._cache:
            self.logger.debug("Using cached CLI data")
            return self._cache[cache_key].copy()
        
        self.logger.info(f"Loading CLI data from {start_date} to {end_date}")
        
        cli_data = None
        
        # Try FRED first (using a proxy series like Leading Economic Index)
        if self.fred_client:
            try:
                # Use US Leading Economic Index as CLI proxy
                cli_series = self.fred_client.get_series(
                    "USSLIND",  # US Leading Economic Index
                    start=start_date,
                    end=end_date
                )
                cli_data = pd.DataFrame({'CLI': cli_series})
                self.logger.info(f"Loaded CLI data from FRED. Shape: {cli_data.shape}")
                
            except Exception as e:
                self.logger.error(f"Failed to load CLI from FRED: {e}")
        
        # Fallback to local data or synthetic data
        if cli_data is None:
            cli_data = self._load_local_cli_data(start_date, end_date)
        
        if cli_data is None:
            cli_data = self._generate_synthetic_cli_data(start_date, end_date)
        
        # Forward fill missing values
        cli_data = cli_data.ffill()
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = cli_data.copy()
        
        return cli_data
    
    def get_interest_rate_data(self,
                              start_date: date,
                              end_date: date,
                              use_cache: bool = True) -> pd.DataFrame:
        """
        Get interest rate data.
        
        Parameters:
        -----------
        start_date : date
            Start date
        end_date : date
            End date
        use_cache : bool
            Whether to use cached data
        
        Returns:
        --------
        pd.DataFrame
            Interest rate data
        """
        cache_key = f"interest_rates_{start_date}_{end_date}"
        
        if use_cache and cache_key in self._cache:
            self.logger.debug("Using cached interest rate data")
            return self._cache[cache_key].copy()
        
        self.logger.info(f"Loading interest rate data from {start_date} to {end_date}")
        
        rate_data = None
        
        if self.fred_client:
            try:
                # Get 10-year Treasury rate
                rate_series = self.fred_client.get_series(
                    "GS10",  # 10-Year Treasury Constant Maturity Rate
                    start=start_date,
                    end=end_date
                )
                rate_data = pd.DataFrame({'Interest_Rate_10Y': rate_series})
                self.logger.info(f"Loaded interest rate data from FRED. Shape: {rate_data.shape}")
                
            except Exception as e:
                self.logger.error(f"Failed to load interest rates from FRED: {e}")
        
        # Fallback to synthetic data
        if rate_data is None:
            rate_data = self._generate_synthetic_interest_rate_data(start_date, end_date)
        
        # Forward fill missing values
        rate_data = rate_data.ffill()
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = rate_data.copy()
        
        return rate_data
    
    def _load_local_cpi_data(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Load CPI data from local files."""
        local_file = self.config.data.raw_data_dir / "cpi_data.csv"
        
        if local_file.exists():
            try:
                data = pd.read_csv(local_file, index_col=0, parse_dates=True)
                data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]
                self.logger.info(f"Loaded local CPI data. Shape: {data.shape}")
                return data
            except Exception as e:
                self.logger.error(f"Failed to load local CPI data: {e}")
        
        return None
    
    def _load_local_cli_data(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Load CLI data from local files."""
        local_file = self.config.data.raw_data_dir / "cli_data.csv"
        
        if local_file.exists():
            try:
                data = pd.read_csv(local_file, index_col=0, parse_dates=True)
                data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]
                self.logger.info(f"Loaded local CLI data. Shape: {data.shape}")
                return data
            except Exception as e:
                self.logger.error(f"Failed to load local CLI data: {e}")
        
        return None
    
    def _generate_synthetic_cpi_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Generate synthetic CPI data for testing purposes."""
        self.logger.warning("Generating synthetic CPI data - for testing only!")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Generate realistic CPI trend with some volatility
        np.random.seed(42)
        base_cpi = 100
        trend = 0.002  # 2% annual inflation
        volatility = 0.005
        
        cpi_values = []
        current_cpi = base_cpi
        
        for i, _ in enumerate(date_range):
            # Add trend and random shock
            monthly_change = trend + np.random.normal(0, volatility)
            current_cpi *= (1 + monthly_change)
            cpi_values.append(current_cpi)
        
        return pd.DataFrame({'CPI': cpi_values}, index=date_range)
    
    def _generate_synthetic_cli_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Generate synthetic CLI data for testing purposes."""
        self.logger.warning("Generating synthetic CLI data - for testing only!")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Generate CLI with cyclical pattern
        np.random.seed(43)
        base_cli = 100
        
        cli_values = []
        for i, _ in enumerate(date_range):
            # Cyclical pattern with noise
            cycle = np.sin(i * 0.1) * 5  # 5-point amplitude cycle
            noise = np.random.normal(0, 2)
            cli_value = base_cli + cycle + noise
            cli_values.append(cli_value)
        
        return pd.DataFrame({'CLI': cli_values}, index=date_range)
    
    def _generate_synthetic_interest_rate_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Generate synthetic interest rate data for testing purposes."""
        self.logger.warning("Generating synthetic interest rate data - for testing only!")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate interest rate with mean reversion
        np.random.seed(44)
        base_rate = 3.0  # 3% base rate
        mean_reversion = 0.001
        volatility = 0.01
        
        rates = []
        current_rate = base_rate
        
        for _ in date_range:
            # Mean reversion + random walk
            change = mean_reversion * (base_rate - current_rate) + np.random.normal(0, volatility)
            current_rate += change
            current_rate = max(0.01, min(current_rate, 10.0))  # Bound between 1bp and 10%
            rates.append(current_rate)
        
        return pd.DataFrame({'Interest_Rate_10Y': rates}, index=date_range)
    
    def clear_cache(self):
        """Clear data cache."""
        self._cache.clear()
        self.logger.info("Economic data cache cleared")