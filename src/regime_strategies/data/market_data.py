"""
Market data provider for financial instruments.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import date, datetime, timedelta
from loguru import logger

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available. Market data will be limited.")

from ..config import Config
from ..utils.data_utils import clean_returns


class MarketDataProvider:
    """
    Provider for market data from various sources.
    """
    
    def __init__(self, config: Config):
        """
        Initialize market data provider.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        
        # Data cache
        self._cache = {}
    
    def get_price_data(self,
                      symbols: List[str],
                      start_date: date,
                      end_date: date,
                      use_cache: bool = True) -> pd.DataFrame:
        """
        Get price data for specified symbols.
        
        Parameters:
        -----------
        symbols : List[str]
            List of ticker symbols
        start_date : date
            Start date
        end_date : date
            End date
        use_cache : bool
            Whether to use cached data
        
        Returns:
        --------
        pd.DataFrame
            Price data with symbols as columns
        """
        cache_key = f"prices_{'-'.join(symbols)}_{start_date}_{end_date}"
        
        if use_cache and cache_key in self._cache:
            self.logger.debug("Using cached price data")
            return self._cache[cache_key].copy()
        
        self.logger.info(f"Loading price data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        price_data = None
        
        # Try Yahoo Finance first
        if YFINANCE_AVAILABLE:
            try:
                price_data = self._fetch_yahoo_prices(symbols, start_date, end_date)
                self.logger.info(f"Loaded price data from Yahoo Finance. Shape: {price_data.shape}")
            except Exception as e:
                self.logger.error(f"Failed to load prices from Yahoo Finance: {e}")
        
        # Fallback to local data or synthetic data
        if price_data is None:
            price_data = self._load_local_price_data(symbols, start_date, end_date)
        
        if price_data is None:
            price_data = self._generate_synthetic_price_data(symbols, start_date, end_date)
        
        # Forward fill missing values
        price_data = price_data.ffill()
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = price_data.copy()
        
        return price_data
    
    def calculate_returns(self,
                         price_data: pd.DataFrame,
                         method: str = 'simple',
                         clean_data: bool = True) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data
        method : str
            Return calculation method ('simple' or 'log')
        clean_data : bool
            Whether to clean return data
        
        Returns:
        --------
        pd.DataFrame
            Return data
        """
        if method == 'simple':
            returns = price_data.pct_change()
        elif method == 'log':
            returns = np.log(price_data / price_data.shift(1))
        else:
            raise ValueError(f"Unknown return calculation method: {method}")
        
        # Drop first row (NaN values)
        returns = returns.dropna()
        
        # Clean returns if requested
        if clean_data:
            for column in returns.columns:
                returns[column] = clean_returns(returns[column])
        
        self.logger.info(f"Calculated {method} returns. Shape: {returns.shape}")
        
        return returns
    
    def get_vix_data(self,
                    start_date: Optional[date] = None,
                    end_date: Optional[date] = None,
                    use_cache: bool = True) -> pd.DataFrame:
        """
        Get VIX volatility index data.
        
        Parameters:
        -----------
        start_date : date, optional
            Start date
        end_date : date, optional
            End date
        use_cache : bool
            Whether to use cached data
        
        Returns:
        --------
        pd.DataFrame
            VIX data
        """
        start_date = start_date or self.config.data.start_date
        end_date = end_date or self.config.data.end_date
        
        cache_key = f"vix_{start_date}_{end_date}"
        
        if use_cache and cache_key in self._cache:
            self.logger.debug("Using cached VIX data")
            return self._cache[cache_key].copy()
        
        self.logger.info(f"Loading VIX data from {start_date} to {end_date}")
        
        vix_data = None
        
        # Try Yahoo Finance
        if YFINANCE_AVAILABLE:
            try:
                vix_ticker = yf.Ticker(self.config.data.vix_symbol)
                vix_history = vix_ticker.history(start=start_date, end=end_date)
                
                if not vix_history.empty:
                    # Ensure timezone-naive datetime index
                    if hasattr(vix_history.index, 'tz') and vix_history.index.tz is not None:
                        vix_history.index = vix_history.index.tz_localize(None)
                    
                    vix_data = pd.DataFrame({'VIX': vix_history['Close']})
                    self.logger.info(f"Loaded VIX data from Yahoo Finance. Shape: {vix_data.shape}")
                
            except Exception as e:
                self.logger.error(f"Failed to load VIX from Yahoo Finance: {e}")
        
        # Fallback to synthetic data
        if vix_data is None:
            vix_data = self._generate_synthetic_vix_data(start_date, end_date)
        
        # Forward fill missing values
        vix_data = vix_data.ffill()
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = vix_data.copy()
        
        return vix_data
    
    def get_benchmark_data(self,
                          benchmark_symbol: str,
                          start_date: date,
                          end_date: date,
                          use_cache: bool = True) -> pd.DataFrame:
        """
        Get benchmark return data.
        
        Parameters:
        -----------
        benchmark_symbol : str
            Benchmark ticker symbol
        start_date : date
            Start date
        end_date : date
            End date
        use_cache : bool
            Whether to use cached data
        
        Returns:
        --------
        pd.DataFrame
            Benchmark return data
        """
        # Load price data
        price_data = self.get_price_data([benchmark_symbol], start_date, end_date, use_cache)
        
        # Calculate returns
        returns = self.calculate_returns(price_data)
        
        return returns.rename(columns={benchmark_symbol: f'{benchmark_symbol}_returns'})
    
    def _fetch_yahoo_prices(self,
                           symbols: List[str],
                           start_date: date,
                           end_date: date) -> pd.DataFrame:
        """
        Fetch price data from Yahoo Finance.
        
        Parameters:
        -----------
        symbols : List[str]
            List of ticker symbols
        start_date : date
            Start date
        end_date : date
            End date
        
        Returns:
        --------
        pd.DataFrame
            Price data
        """
        # Add buffer to ensure we get data
        buffer_start = start_date - timedelta(days=10)
        
        # Download data for all symbols
        data = yf.download(symbols, start=buffer_start, end=end_date, group_by='ticker', auto_adjust=True)
        
        # Extract adjusted close prices
        if len(symbols) == 1:
            # Single symbol case
            if 'Adj Close' in data.columns:
                prices = pd.DataFrame({symbols[0]: data['Adj Close']})
            else:
                prices = pd.DataFrame({symbols[0]: data['Close']})
        else:
            # Multiple symbols case
            prices = pd.DataFrame()
            for symbol in symbols:
                try:
                    if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                        # Multi-level columns
                        if symbol in data.columns.levels[0]:
                            if (symbol, 'Adj Close') in data.columns:
                                prices[symbol] = data[(symbol, 'Adj Close')]
                            elif (symbol, 'Close') in data.columns:
                                prices[symbol] = data[(symbol, 'Close')]
                    else:
                        # Single level columns (when downloading single symbol)
                        if 'Adj Close' in data.columns:
                            prices[symbol] = data['Adj Close']
                        elif 'Close' in data.columns:
                            prices[symbol] = data['Close']
                except Exception as e:
                    self.logger.warning(f"Could not extract data for {symbol}: {e}")
        
        if prices.empty:
            raise ValueError("No price data could be extracted from Yahoo Finance")
        
        # Ensure timezone-naive datetime index
        if hasattr(prices.index, 'tz') and prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        
        # Filter to requested date range
        prices = prices[prices.index.date >= start_date]
        
        return prices
    
    def _load_local_price_data(self,
                              symbols: List[str],
                              start_date: date,
                              end_date: date) -> Optional[pd.DataFrame]:
        """Load price data from local files."""
        price_data = pd.DataFrame()
        
        for symbol in symbols:
            local_file = self.config.data.raw_data_dir / f"{symbol}_prices.csv"
            
            if local_file.exists():
                try:
                    data = pd.read_csv(local_file, index_col=0, parse_dates=True)
                    data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]
                    
                    # Use first column as price
                    price_data[symbol] = data.iloc[:, 0]
                    
                except Exception as e:
                    self.logger.error(f"Failed to load local data for {symbol}: {e}")
        
        if not price_data.empty:
            self.logger.info(f"Loaded local price data. Shape: {price_data.shape}")
            return price_data
        
        return None
    
    def _generate_synthetic_price_data(self,
                                     symbols: List[str],
                                     start_date: date,
                                     end_date: date) -> pd.DataFrame:
        """Generate synthetic price data for testing purposes."""
        self.logger.warning("Generating synthetic price data - for testing only!")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic price movements
        np.random.seed(42)
        
        price_data = pd.DataFrame(index=date_range)
        
        for i, symbol in enumerate(symbols):
            # Different parameters for different assets
            base_price = 100 + i * 50  # Different starting prices
            drift = 0.0002 + i * 0.0001  # Slightly different expected returns
            volatility = 0.015 + i * 0.005  # Different volatilities
            
            prices = [base_price]
            current_price = base_price
            
            for _ in range(len(date_range) - 1):
                # Geometric Brownian Motion
                random_shock = np.random.normal(0, 1)
                price_change = current_price * (drift + volatility * random_shock)
                current_price += price_change
                current_price = max(current_price, 1.0)  # Prevent negative prices
                prices.append(current_price)
            
            price_data[symbol] = prices
        
        return price_data
    
    def _generate_synthetic_vix_data(self,
                                   start_date: date,
                                   end_date: date) -> pd.DataFrame:
        """Generate synthetic VIX data for testing purposes."""
        self.logger.warning("Generating synthetic VIX data - for testing only!")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate VIX-like data with mean reversion
        np.random.seed(45)
        base_vix = 20.0  # Average VIX level
        mean_reversion = 0.05
        volatility = 2.0
        
        vix_values = []
        current_vix = base_vix
        
        for _ in date_range:
            # Mean reversion with jumps
            if np.random.random() < 0.02:  # 2% chance of jump
                jump = np.random.normal(0, 10)
            else:
                jump = 0
            
            change = mean_reversion * (base_vix - current_vix) + np.random.normal(0, volatility) + jump
            current_vix += change
            current_vix = max(5.0, min(current_vix, 80.0))  # Bound VIX values
            vix_values.append(current_vix)
        
        return pd.DataFrame({'VIX': vix_values}, index=date_range)
    
    def clear_cache(self):
        """Clear data cache."""
        self._cache.clear()
        self.logger.info("Market data cache cleared")