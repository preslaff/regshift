"""
Main data loading and management class.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from pathlib import Path
from loguru import logger

from ..config import Config
from ..utils.data_utils import validate_data, align_data, enforce_point_in_time
from .economic_data import EconomicDataProvider
from .market_data import MarketDataProvider


class DataLoader:
    """
    Main class for loading and managing all data sources.
    """
    
    def __init__(self, config: Config):
        """
        Initialize data loader with configuration.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        
        # Initialize data providers
        self.economic_data = EconomicDataProvider(config)
        self.market_data = MarketDataProvider(config)
        
        # Data cache
        self._cache = {}
        self._cache_timestamps = {}
        
    def load_economic_indicators(self, 
                                start_date: Optional[date] = None,
                                end_date: Optional[date] = None,
                                use_cache: bool = True) -> pd.DataFrame:
        """
        Load economic indicators for regime identification.
        
        Parameters:
        -----------
        start_date : date, optional
            Start date for data
        end_date : date, optional
            End date for data
        use_cache : bool
            Whether to use cached data
        
        Returns:
        --------
        pd.DataFrame
            Economic indicators data
        """
        cache_key = f"economic_indicators_{start_date}_{end_date}"
        
        if use_cache and cache_key in self._cache:
            self.logger.debug("Using cached economic indicators")
            return self._cache[cache_key].copy()
        
        start_date = start_date or self.config.data.start_date
        end_date = end_date or self.config.data.end_date
        
        self.logger.info(f"Loading economic indicators from {start_date} to {end_date}")
        
        # Load CPI data
        cpi_data = self.economic_data.get_cpi_data(start_date, end_date)
        
        # Load CLI data
        cli_data = self.economic_data.get_cli_data(start_date, end_date)
        
        # Align data
        aligned_data = align_data(cpi_data, cli_data, method='inner')
        
        # Combine into single DataFrame
        economic_data = pd.DataFrame({
            'CPI': aligned_data[0]['CPI'] if 'CPI' in aligned_data[0].columns else aligned_data[0].iloc[:, 0],
            'CLI': aligned_data[1]['CLI'] if 'CLI' in aligned_data[1].columns else aligned_data[1].iloc[:, 0]
        })
        
        # Calculate Investment Clock features
        economic_data = self._calculate_investment_clock_features(economic_data)
        
        # Validate data
        validate_data(economic_data, required_columns=['CPI', 'CLI'])
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = economic_data.copy()
            self._cache_timestamps[cache_key] = datetime.now()
        
        self.logger.info(f"Loaded economic indicators. Shape: {economic_data.shape}")
        
        return economic_data
    
    def load_market_data(self,
                        assets: Optional[List[str]] = None,
                        start_date: Optional[date] = None,
                        end_date: Optional[date] = None,
                        use_cache: bool = True) -> pd.DataFrame:
        """
        Load market data for specified assets.
        
        Parameters:
        -----------
        assets : List[str], optional
            List of asset symbols
        start_date : date, optional
            Start date for data
        end_date : date, optional
            End date for data
        use_cache : bool
            Whether to use cached data
        
        Returns:
        --------
        pd.DataFrame
            Market data with returns
        """
        assets = assets or self.config.data.default_assets
        start_date = start_date or self.config.data.start_date
        end_date = end_date or self.config.data.end_date
        
        cache_key = f"market_data_{'-'.join(assets)}_{start_date}_{end_date}"
        
        if use_cache and cache_key in self._cache:
            self.logger.debug("Using cached market data")
            return self._cache[cache_key].copy()
        
        self.logger.info(f"Loading market data for {len(assets)} assets from {start_date} to {end_date}")
        
        # Load price data
        price_data = self.market_data.get_price_data(assets, start_date, end_date)
        
        # Calculate returns
        returns_data = self.market_data.calculate_returns(price_data)
        
        # Load VIX data
        vix_data = self.market_data.get_vix_data(start_date, end_date)
        
        # Combine market data
        market_data = returns_data.copy()
        if not vix_data.empty:
            market_data = market_data.join(vix_data, how='left')
        
        # Validate data
        validate_data(market_data)
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = market_data.copy()
            self._cache_timestamps[cache_key] = datetime.now()
        
        self.logger.info(f"Loaded market data. Shape: {market_data.shape}")
        
        return market_data
    
    def load_forecasting_features(self,
                                 current_date: datetime,
                                 use_cache: bool = True) -> pd.DataFrame:
        """
        Load features for regime forecasting at a specific point in time.
        
        Parameters:
        -----------
        current_date : datetime
            Current decision date
        use_cache : bool
            Whether to use cached data
        
        Returns:
        --------
        pd.DataFrame
            Forecasting features available at current_date
        """
        cache_key = f"forecasting_features_{current_date.date()}"
        
        if use_cache and cache_key in self._cache:
            self.logger.debug("Using cached forecasting features")
            return self._cache[cache_key].copy()
        
        self.logger.info(f"Loading forecasting features for {current_date}")
        
        # Load market data up to current date
        market_data = self.load_market_data(end_date=current_date.date(), use_cache=use_cache)
        
        # Apply point-in-time constraints
        market_data = enforce_point_in_time(market_data, current_date, lag_days=0)
        
        # Load VIX data
        vix_data = self.market_data.get_vix_data(end_date=current_date.date())
        vix_data = enforce_point_in_time(vix_data, current_date, lag_days=0)
        
        # Create feature matrix
        features = pd.DataFrame(index=market_data.index)
        
        # Market volatility features
        if not market_data.empty:
            # Rolling volatilities
            for window in [5, 20, 60]:
                features[f'market_vol_{window}d'] = market_data.std(axis=1).rolling(window).mean()
            
            # Market return features
            market_return = market_data.mean(axis=1)
            features['market_return'] = market_return
            features['market_return_5d'] = market_return.rolling(5).mean()
            features['market_return_20d'] = market_return.rolling(20).mean()
        
        # VIX features
        if not vix_data.empty:
            features = features.join(vix_data, how='left')
            if 'VIX' in features.columns:
                features['VIX_change'] = features['VIX'].pct_change()
                features['VIX_ma_20'] = features['VIX'].rolling(20).mean()
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = features.copy()
            self._cache_timestamps[cache_key] = datetime.now()
        
        self.logger.info(f"Loaded forecasting features. Shape: {features.shape}")
        
        return features
    
    def get_point_in_time_data(self,
                              data_type: str,
                              current_date: datetime,
                              **kwargs) -> pd.DataFrame:
        """
        Get point-in-time data to avoid look-ahead bias.
        
        Parameters:
        -----------
        data_type : str
            Type of data ('economic', 'market', 'features')
        current_date : datetime
            Current decision date
        **kwargs
            Additional arguments for data loading
        
        Returns:
        --------
        pd.DataFrame
            Point-in-time data
        """
        self.logger.debug(f"Loading point-in-time {data_type} data for {current_date}")
        
        if data_type == 'economic':
            data = self.load_economic_indicators(end_date=current_date.date(), **kwargs)
            # Apply lag for economic data publication
            data = enforce_point_in_time(data, current_date, lag_days=30)  # 30-day lag for econ data
            
        elif data_type == 'market':
            data = self.load_market_data(end_date=current_date.date(), **kwargs)
            data = enforce_point_in_time(data, current_date, lag_days=0)
            
        elif data_type == 'features':
            data = self.load_forecasting_features(current_date, **kwargs)
            
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return data
    
    def clear_cache(self):
        """Clear data cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Data cache cleared")
    
    def _calculate_investment_clock_features(self, economic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Investment Clock features from CPI and CLI data.
        
        Parameters:
        -----------
        economic_data : pd.DataFrame
            Economic data with CPI and CLI columns
        
        Returns:
        --------
        pd.DataFrame
            Economic data with Investment Clock features
        """
        data = economic_data.copy()
        
        # CPI transformations
        if 'CPI' in data.columns:
            # Quarterly change
            data['CPI_quarterly'] = data['CPI'].pct_change(periods=3)  # 3 months
            
            # 36-month moving average
            data['CPI_ma36'] = data['CPI'].rolling(window=36).mean()
            
            # Investment Clock CPI signal
            data['CPI_signal'] = data['CPI_quarterly'] - data['CPI_ma36'].pct_change(periods=3)
            
        # CLI transformations
        if 'CLI' in data.columns:
            # 3-month change
            data['CLI_change3m'] = data['CLI'].pct_change(periods=3)
            
            # Investment Clock CLI signal
            data['CLI_signal'] = data['CLI_change3m']
        
        return data