"""
Backtesting engine for regime-aware investment strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger
import warnings

from ..config import Config
from ..data.data_loader import DataLoader
from ..models.regime_identifier import RegimeIdentifier
from ..models.regime_forecaster import RegimeForecaster
from ..models.cma_calculator import RegimeConditionalCMA
from ..optimization.portfolio_optimizer import PortfolioOptimizer, OptimizationResult
from ..utils.data_utils import enforce_point_in_time, resample_data
from ..utils.metrics import calculate_metrics


@dataclass
class BacktestResult:
    """
    Container for backtest results.
    """
    # Performance data
    portfolio_returns: pd.Series
    portfolio_weights: pd.DataFrame
    benchmark_returns: Optional[pd.Series] = None
    static_mpt_returns: Optional[pd.Series] = None
    
    # Regime data
    predicted_regimes: pd.Series = field(default_factory=pd.Series)
    actual_regimes: pd.Series = field(default_factory=pd.Series)
    regime_prediction_accuracy: float = 0.0
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    regime_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Portfolio characteristics
    turnover: pd.Series = field(default_factory=pd.Series)
    transaction_costs: pd.Series = field(default_factory=pd.Series)
    
    # Backtest metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    rebalance_frequency: str = "monthly"
    strategy_name: str = "regime_strategy"
    
    # Additional data
    optimization_results: List[OptimizationResult] = field(default_factory=list)
    risk_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


class BacktestEngine:
    """
    Main backtesting engine for regime-aware investment strategies.
    """
    
    def __init__(self, config: Config):
        """
        Initialize backtest engine.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.regime_identifier = RegimeIdentifier(config)
        self.regime_forecaster = RegimeForecaster(config)
        self.cma_calculator = RegimeConditionalCMA(config)
        self.optimizer = PortfolioOptimizer(config)
        
        # Backtest state
        self.results = None
        self.is_fitted = False
        
    def run_backtest(self,
                    start_date: datetime,
                    end_date: datetime,
                    assets: Optional[List[str]] = None,
                    benchmark: Optional[str] = None,
                    initial_capital: float = None,
                    strategy_name: str = "regime_strategy") -> BacktestResult:
        """
        Run complete backtest simulation.
        
        Parameters:
        -----------
        start_date : datetime
            Backtest start date
        end_date : datetime
            Backtest end date
        assets : List[str], optional
            Asset universe
        benchmark : str, optional
            Benchmark symbol
        initial_capital : float, optional
            Initial capital
        strategy_name : str
            Strategy name for results
        
        Returns:
        --------
        BacktestResult
            Complete backtest results
        """
        self.logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        assets = assets or self.config.data.default_assets
        benchmark = benchmark or self.config.backtest.benchmark_symbol
        initial_capital = initial_capital or self.config.backtest.initial_capital
        
        # Step 1: Load and prepare data
        self._load_backtest_data(start_date, end_date, assets, benchmark)
        
        # Step 2: Fit regime models on initial training period
        self._fit_regime_models()
        
        # Step 3: Run simulation loop
        results = self._run_simulation_loop(start_date, end_date, initial_capital, strategy_name)
        
        # Step 4: Calculate final performance metrics
        self._calculate_final_metrics(results)
        
        self.results = results
        self.is_fitted = True
        
        self.logger.info("Backtest completed successfully")
        
        return results
    
    def _load_backtest_data(self,
                           start_date: datetime,
                           end_date: datetime,
                           assets: List[str],
                           benchmark: str):
        """Load all required data for backtesting."""
        self.logger.info("Loading backtest data")
        
        # Extend start date for initial training period
        training_start = start_date - timedelta(days=365 * 2)  # 2 years of training data
        
        # Load market data
        self.market_data = self.data_loader.load_market_data(
            assets=assets,
            start_date=training_start.date(),
            end_date=end_date.date()
        )
        
        # Load economic indicators
        self.economic_data = self.data_loader.load_economic_indicators(
            start_date=training_start.date(),
            end_date=end_date.date()
        )
        
        # Load benchmark data
        if benchmark:
            self.benchmark_data = self.data_loader.market_data.get_benchmark_data(
                benchmark,
                training_start.date(),
                end_date.date()
            )
        else:
            self.benchmark_data = None
        
        # Store metadata
        self.assets = assets
        self.benchmark_symbol = benchmark
        self.backtest_start = start_date
        self.backtest_end = end_date
        
        # Calculate static MPT benchmark weights (calculated once at start)
        self.static_mpt_weights = self._calculate_static_mpt_weights()
        
        self.logger.info(f"Loaded data: {len(self.market_data)} market observations, "
                        f"{len(self.economic_data)} economic observations")
    
    def _fit_regime_models(self):
        """Fit regime identification and forecasting models."""
        self.logger.info("Fitting regime models")
        
        # Fit regime identifier on historical data
        self.regime_identifier.fit(
            economic_data=self.economic_data,
            market_data=self.market_data
        )
        
        # Get historical regimes
        self.historical_regimes = self.regime_identifier.get_regimes()
        
        # Fit CMA calculator (only use the requested assets, not auxiliary data like VIX)
        asset_returns = self.market_data[self.assets]  # Filter to only requested assets
        aligned_returns, aligned_regimes = asset_returns.align(
            self.historical_regimes, join='inner', axis=0
        )
        
        self.cma_calculator.fit(aligned_returns, aligned_regimes)
        
        # Prepare forecasting features and fit forecaster
        forecasting_features = self._create_forecasting_features(self.market_data)
        
        self.regime_forecaster.fit(
            features=forecasting_features,
            regimes=self.historical_regimes
        )
        
        self.logger.info("Regime models fitted successfully")
    
    def _run_simulation_loop(self,
                            start_date: datetime,
                            end_date: datetime,
                            initial_capital: float,
                            strategy_name: str) -> BacktestResult:
        """Run the main simulation loop."""
        self.logger.info("Starting simulation loop")
        
        # Initialize results containers
        portfolio_values = []
        portfolio_weights_list = []
        predicted_regimes_list = []
        actual_regimes_list = []
        optimization_results = []
        transaction_costs_list = []
        
        # Rebalancing dates
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        
        # Initialize portfolio
        current_weights = None
        portfolio_value = initial_capital
        
        for i, rebalance_date in enumerate(rebalance_dates):
            self.logger.debug(f"Rebalancing {i+1}/{len(rebalance_dates)}: {rebalance_date}")
            
            try:
                # Step 1: Predict regime for next period
                regime_forecast = self._predict_regime(rebalance_date)
                predicted_regime = regime_forecast['predicted_regime']
                
                # Step 2: Get regime-conditional CMA
                cma = self.cma_calculator.get_cma(predicted_regime)
                
                # Step 3: Optimize portfolio
                opt_result = self.optimizer.optimize_regime_conditional(
                    cma,
                    method=self.config.optimization.default_method,
                    regime=predicted_regime
                )
                
                # Step 4: Calculate transaction costs if rebalancing
                transaction_cost = 0.0
                if current_weights is not None:
                    weight_changes = np.abs(opt_result.weights - current_weights)
                    transaction_cost = np.sum(weight_changes) * self.config.backtest.transaction_costs
                    transaction_cost *= portfolio_value  # Convert to dollar terms
                
                # Step 5: Update portfolio
                current_weights = opt_result.weights
                portfolio_value -= transaction_cost  # Subtract transaction costs
                
                # Step 6: Calculate returns until next rebalance date
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                    period_returns = self._calculate_period_returns(
                        rebalance_date, next_date, current_weights
                    )
                    portfolio_value *= (1 + period_returns)
                
                # Step 7: Get actual regime for evaluation
                actual_regime = self._get_actual_regime(rebalance_date)
                
                # Store results
                portfolio_values.append(portfolio_value)
                portfolio_weights_list.append(current_weights.copy())
                predicted_regimes_list.append(predicted_regime)
                actual_regimes_list.append(actual_regime)
                optimization_results.append(opt_result)
                transaction_costs_list.append(transaction_cost)
                
            except Exception as e:
                self.logger.error(f"Error at {rebalance_date}: {e}")
                # Use previous weights or equal weights as fallback
                if current_weights is None:
                    current_weights = np.ones(len(self.assets)) / len(self.assets)
                
                # Store fallback results
                portfolio_values.append(portfolio_value)
                portfolio_weights_list.append(current_weights.copy())
                predicted_regimes_list.append(0)  # Default regime
                actual_regimes_list.append(0)
                transaction_costs_list.append(0.0)
        
        # Create result time series
        portfolio_returns = self._calculate_portfolio_return_series(
            rebalance_dates, portfolio_values, initial_capital
        )
        
        portfolio_weights_df = pd.DataFrame(
            portfolio_weights_list,
            index=rebalance_dates,
            columns=self.assets
        )
        
        predicted_regimes_series = pd.Series(
            predicted_regimes_list,
            index=rebalance_dates,
            name='predicted_regime'
        )
        
        actual_regimes_series = pd.Series(
            actual_regimes_list,
            index=rebalance_dates,
            name='actual_regime'
        )
        
        transaction_costs_series = pd.Series(
            transaction_costs_list,
            index=rebalance_dates,
            name='transaction_costs'
        )
        
        # Calculate turnover
        turnover_series = self._calculate_turnover(portfolio_weights_df)
        
        # Get benchmark returns if available
        benchmark_returns = None
        if self.benchmark_data is not None:
            benchmark_returns = self._align_benchmark_returns(rebalance_dates)
        
        # Calculate static MPT benchmark returns
        static_mpt_returns = self._calculate_static_mpt_returns(rebalance_dates)
        
        # Create backtest result
        result = BacktestResult(
            portfolio_returns=portfolio_returns,
            portfolio_weights=portfolio_weights_df,
            benchmark_returns=benchmark_returns,
            static_mpt_returns=static_mpt_returns,
            predicted_regimes=predicted_regimes_series,
            actual_regimes=actual_regimes_series,
            turnover=turnover_series,
            transaction_costs=transaction_costs_series,
            start_date=start_date,
            end_date=end_date,
            rebalance_frequency=self.config.backtest.rebalance_frequency,
            strategy_name=strategy_name,
            optimization_results=optimization_results
        )
        
        self.logger.info(f"Simulation completed: {len(rebalance_dates)} rebalancing periods")
        
        return result
    
    def _predict_regime(self, current_date: datetime) -> Dict[str, Any]:
        """Predict regime for given date."""
        # Get point-in-time forecasting features
        features = self.data_loader.get_point_in_time_data(
            'features', current_date
        )
        
        # Forecast regime
        forecast = self.regime_forecaster.forecast_regime(
            current_date, features, horizon=1
        )
        
        return forecast
    
    def _get_actual_regime(self, date: datetime) -> int:
        """Get actual regime for given date (for evaluation)."""
        try:
            # Find closest date in historical regimes
            available_dates = self.historical_regimes.index
            closest_date = min(available_dates, 
                             key=lambda x: abs((x - date).total_seconds()))
            
            return self.historical_regimes[closest_date]
        except:
            return 0  # Default regime if not found
    
    def _calculate_period_returns(self,
                                 start_date: datetime,
                                 end_date: datetime,
                                 weights: np.ndarray) -> float:
        """Calculate portfolio return for a specific period."""
        try:
            # Get market returns for the period (only for the requested assets)
            period_mask = ((self.market_data.index > start_date) & 
                          (self.market_data.index <= end_date))
            period_returns = self.market_data[period_mask][self.assets]
            
            if period_returns.empty:
                return 0.0
            
            # Calculate weighted portfolio returns
            portfolio_returns = period_returns.dot(weights)
            
            # Compound returns over the period
            cumulative_return = (1 + portfolio_returns).prod() - 1
            
            return cumulative_return
            
        except Exception as e:
            self.logger.error(f"Error calculating period returns: {e}")
            return 0.0
    
    def _get_rebalance_dates(self,
                            start_date: datetime,
                            end_date: datetime) -> List[datetime]:
        """Generate rebalancing dates based on frequency."""
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'quarterly': 'Q'
        }
        
        freq = freq_map.get(self.config.backtest.rebalance_frequency, 'M')
        
        date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq
        )
        
        return date_range.tolist()
    
    def _create_forecasting_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for regime forecasting."""
        features = pd.DataFrame(index=market_data.index)
        
        # Market volatility features
        for window in [5, 20, 60]:
            features[f'market_vol_{window}'] = market_data.std(axis=1).rolling(window).mean()
        
        # Market momentum features
        market_returns = market_data.mean(axis=1)
        for window in [5, 20, 60]:
            features[f'market_mom_{window}'] = market_returns.rolling(window).mean()
        
        # Individual asset features
        for asset in market_data.columns:
            features[f'{asset}_vol_20'] = market_data[asset].rolling(20).std()
            features[f'{asset}_mom_20'] = market_data[asset].rolling(20).mean()
        
        return features
    
    def _calculate_portfolio_return_series(self,
                                         dates: List[datetime],
                                         values: List[float],
                                         initial_capital: float) -> pd.Series:
        """Calculate portfolio return series from values."""
        returns = []
        
        for i in range(1, len(values)):
            ret = (values[i] - values[i-1]) / values[i-1]
            returns.append(ret)
        
        # First return is from initial capital to first value
        if len(values) > 0:
            first_return = (values[0] - initial_capital) / initial_capital
            returns.insert(0, first_return)
        
        return pd.Series(returns, index=dates[:len(returns)])
    
    def _calculate_turnover(self, weights_df: pd.DataFrame) -> pd.Series:
        """Calculate portfolio turnover."""
        turnover = pd.Series(index=weights_df.index, dtype=float)
        turnover.iloc[0] = 0.0  # No turnover for first period
        
        for i in range(1, len(weights_df)):
            weight_changes = np.abs(weights_df.iloc[i] - weights_df.iloc[i-1])
            turnover.iloc[i] = weight_changes.sum()
        
        return turnover
    
    def _align_benchmark_returns(self, rebalance_dates: List[datetime]) -> pd.Series:
        """Align benchmark returns with rebalance dates."""
        if self.benchmark_data is None:
            return None
        
        benchmark_returns = []
        
        for i in range(len(rebalance_dates)):
            try:
                if i == 0:
                    # Use return from start of backtest to first rebalance
                    period_start = self.backtest_start
                else:
                    period_start = rebalance_dates[i-1]
                
                period_end = rebalance_dates[i]
                
                period_mask = ((self.benchmark_data.index > period_start) & 
                              (self.benchmark_data.index <= period_end))
                period_returns = self.benchmark_data[period_mask].iloc[:, 0]
                
                if not period_returns.empty:
                    period_return = (1 + period_returns).prod() - 1
                else:
                    period_return = 0.0
                
                benchmark_returns.append(period_return)
                
            except Exception as e:
                self.logger.error(f"Error calculating benchmark return: {e}")
                benchmark_returns.append(0.0)
        
        return pd.Series(benchmark_returns, index=rebalance_dates)
    
    def _calculate_static_mpt_weights(self) -> np.ndarray:
        """
        Calculate static MPT portfolio weights using all historical data.
        This creates a benchmark that uses the same optimization method but 
        with static weights calculated once at the beginning.
        """
        try:
            # Use all available historical data for static calculation
            asset_returns = self.market_data[self.assets]
            
            # Calculate unconditional expected returns and covariance
            mu = asset_returns.mean().values * 252  # Annualized
            sigma = asset_returns.cov().values * 252  # Annualized
            
            # Use the same optimization method as the dynamic strategy
            result = self.optimizer.optimize(
                mu=mu, 
                sigma=sigma, 
                method=self.config.optimization.default_method
            )
            
            if result and result.success:
                self.logger.info(f"Static MPT benchmark weights calculated: "
                               f"Return={result.expected_return:.3f}, "
                               f"Vol={result.volatility:.3f}, "
                               f"Sharpe={result.sharpe_ratio:.3f}")
                return result.weights
            else:
                # Fallback to equal weights
                self.logger.warning("Static MPT optimization failed, using equal weights")
                return np.ones(len(self.assets)) / len(self.assets)
                
        except Exception as e:
            self.logger.error(f"Error calculating static MPT weights: {e}")
            # Fallback to equal weights
            return np.ones(len(self.assets)) / len(self.assets)
    
    def _calculate_static_mpt_returns(self, rebalance_dates: List[datetime]) -> pd.Series:
        """Calculate returns for static MPT benchmark portfolio."""
        static_returns = []
        
        for i, rebalance_date in enumerate(rebalance_dates):
            try:
                if i == 0:
                    # No return for first period
                    static_returns.append(0.0)
                    continue
                
                # Calculate period return using static weights
                period_start = rebalance_dates[i-1]
                period_end = rebalance_date
                
                period_return = self._calculate_period_returns(
                    period_start, period_end, self.static_mpt_weights
                )
                static_returns.append(period_return)
                
            except Exception as e:
                self.logger.error(f"Error calculating static MPT return: {e}")
                static_returns.append(0.0)
        
        return pd.Series(static_returns, index=rebalance_dates)
    
    def _calculate_final_metrics(self, result: BacktestResult):
        """Calculate final performance metrics."""
        self.logger.info("Calculating final performance metrics")
        
        # Overall performance metrics
        result.performance_metrics = calculate_metrics(
            result.portfolio_returns,
            result.benchmark_returns,
            risk_free_rate=self.config.backtest.risk_free_rate
        )
        
        # Regime prediction accuracy
        if not result.predicted_regimes.empty and not result.actual_regimes.empty:
            correct_predictions = (result.predicted_regimes == result.actual_regimes).sum()
            total_predictions = len(result.predicted_regimes)
            result.regime_prediction_accuracy = correct_predictions / total_predictions
        
        # Performance by regime
        result.regime_performance = self._calculate_regime_performance(result)
        
        self.logger.info(f"Performance metrics calculated: "
                        f"Total return: {result.performance_metrics.get('total_return', 0):.2%}, "
                        f"Sharpe ratio: {result.performance_metrics.get('sharpe_ratio', 0):.3f}, "
                        f"Regime accuracy: {result.regime_prediction_accuracy:.2%}")
    
    def _calculate_regime_performance(self, result: BacktestResult) -> Dict[int, Dict[str, float]]:
        """Calculate performance metrics by regime."""
        regime_performance = {}
        
        if result.actual_regimes.empty:
            return regime_performance
        
        unique_regimes = result.actual_regimes.unique()
        
        for regime in unique_regimes:
            regime_mask = result.actual_regimes == regime
            regime_returns = result.portfolio_returns[regime_mask]
            
            if not regime_returns.empty:
                regime_metrics = calculate_metrics(regime_returns)
                regime_performance[regime] = regime_metrics
        
        return regime_performance
    
    def get_results(self) -> Optional[BacktestResult]:
        """Get backtest results."""
        return self.results
    
    def save_results(self, filepath: str):
        """Save backtest results to file."""
        if self.results is None:
            raise ValueError("No results to save. Run backtest first.")
        
        try:
            # Create a dictionary with all result components
            results_dict = {
                'portfolio_returns': self.results.portfolio_returns,
                'portfolio_weights': self.results.portfolio_weights,
                'benchmark_returns': self.results.benchmark_returns,
                'predicted_regimes': self.results.predicted_regimes,
                'actual_regimes': self.results.actual_regimes,
                'performance_metrics': self.results.performance_metrics,
                'regime_performance': self.results.regime_performance,
                'regime_prediction_accuracy': self.results.regime_prediction_accuracy,
                'turnover': self.results.turnover,
                'transaction_costs': self.results.transaction_costs,
                'metadata': {
                    'start_date': self.results.start_date,
                    'end_date': self.results.end_date,
                    'rebalance_frequency': self.results.rebalance_frequency,
                    'strategy_name': self.results.strategy_name
                }
            }
            
            # Save to pickle for full object preservation
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(results_dict, f)
            
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def load_results(self, filepath: str):
        """Load backtest results from file."""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                results_dict = pickle.load(f)
            
            # Reconstruct BacktestResult object
            self.results = BacktestResult(
                portfolio_returns=results_dict['portfolio_returns'],
                portfolio_weights=results_dict['portfolio_weights'],
                benchmark_returns=results_dict.get('benchmark_returns'),
                predicted_regimes=results_dict['predicted_regimes'],
                actual_regimes=results_dict['actual_regimes'],
                performance_metrics=results_dict['performance_metrics'],
                regime_performance=results_dict['regime_performance'],
                regime_prediction_accuracy=results_dict['regime_prediction_accuracy'],
                turnover=results_dict['turnover'],
                transaction_costs=results_dict['transaction_costs'],
                **results_dict['metadata']
            )
            
            self.is_fitted = True
            self.logger.info(f"Results loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            raise