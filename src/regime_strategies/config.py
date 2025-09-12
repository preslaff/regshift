"""
Configuration management for the regime strategies system.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, date
import os
from pathlib import Path


class DataConfig(BaseModel):
    """Configuration for data sources and parameters."""
    
    # Data sources
    fred_api_key: Optional[str] = Field(default=None, env="FRED_API_KEY")
    alpha_vantage_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_KEY")
    
    # Data parameters
    start_date: date = Field(default=date(2000, 1, 1))
    end_date: date = Field(default=date.today())
    
    # Economic indicators
    cpi_series: str = "CPIAUCSL"  # Consumer Price Index
    cli_series: str = "OECD_MEI_CLI"  # Composite Leading Indicator
    vix_symbol: str = "^VIX"
    
    # Asset universe
    default_assets: List[str] = [
        "SPY",    # S&P 500
        "TLT",    # 20+ Year Treasury
        "GLD",    # Gold
        "IWM",    # Russell 2000
        "EFA"     # EAFE
    ]
    
    # Data storage
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")


class RegimeConfig(BaseModel):
    """Configuration for regime identification and forecasting."""
    
    # Stage 1: Historical Regime Identification
    regime_method: str = "investment_clock"  # Options: "investment_clock", "hmm", "kmeans"
    n_regimes: int = 4
    
    # Investment Clock parameters
    cpi_lookback_quarters: int = 4
    cpi_ma_months: int = 36
    cli_lookback_months: int = 3
    
    # HMM parameters
    hmm_n_iter: int = 100
    hmm_tol: float = 1e-6
    hmm_random_state: int = 42
    
    # K-means parameters
    kmeans_random_state: int = 42
    kmeans_n_init: int = 10
    
    # Stage 2: Regime Forecasting
    forecasting_models: List[str] = ["random_forest", "logistic_regression"]
    ensemble_method: str = "voting"  # Options: "voting", "weighted_average"
    
    # Model parameters
    random_forest_params: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
    logistic_regression_params: Dict[str, Any] = {
        "random_state": 42,
        "max_iter": 1000
    }
    
    # Time series parameters
    use_rolling_window: bool = False
    rolling_window_years: int = 5
    min_training_samples: int = 252  # Trading days


class OptimizationConfig(BaseModel):
    """Configuration for portfolio optimization."""
    
    # Optimization methods
    default_method: str = "max_sharpe"  # Options: "max_sharpe", "risk_parity", "min_variance"
    
    # Constraints
    long_only: bool = True
    weight_bounds: tuple = (0.0, 1.0)
    sum_to_one: bool = True
    
    # Risk parameters
    risk_free_rate: float = 0.02
    target_return: Optional[float] = None
    max_weight: float = 0.4  # Maximum weight per asset
    min_weight: float = 0.0  # Minimum weight per asset
    
    # Optimization solver parameters
    solver: str = "CLARABEL"  # CVXPY solver - more widely available than ECOS
    max_iters: int = 10000
    tolerance: float = 1e-8


class BacktestConfig(BaseModel):
    """Configuration for backtesting and evaluation."""
    
    # Backtesting parameters
    rebalance_frequency: str = "monthly"  # Options: "daily", "weekly", "monthly", "quarterly"
    initial_capital: float = 1000000.0
    transaction_costs: float = 0.001  # 10 bps
    
    # Benchmark
    benchmark_symbol: str = "SPY"
    static_benchmark: bool = True  # Use static MPT portfolio as benchmark
    
    # Performance metrics
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95  # For VaR calculation
    
    # Output
    save_results: bool = True
    results_dir: Path = Path("results")
    plot_format: str = "html"  # Options: "html", "png", "pdf"


class Config(BaseModel):
    """Main configuration class."""
    
    # Sub-configurations
    data: DataConfig = DataConfig()
    regime: RegimeConfig = RegimeConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    backtest: BacktestConfig = BacktestConfig()
    
    # General settings
    random_seed: int = 42
    log_level: str = "INFO"
    n_jobs: int = -1  # Number of parallel jobs
    
    # Environment
    environment: str = "development"  # Options: "development", "production"
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.data_dir,
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.backtest.results_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from file."""
        # This could be extended to support YAML/JSON config files
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()


# Global configuration instance
config = Config()