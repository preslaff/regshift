"""
Capital Market Assumptions (CMA) calculator for regime-conditional optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger
from scipy import stats
import warnings

from ..config import Config
from ..utils.data_utils import clean_returns
from ..utils.metrics import calculate_metrics


class RegimeConditionalCMA:
    """
    Calculator for regime-conditional Capital Market Assumptions.
    """
    
    def __init__(self, config: Config):
        """
        Initialize CMA calculator.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        
        # Storage for regime-specific CMAs
        self.regime_cmas = {}
        self.unconditional_cma = None
        self.asset_names = []
        
        # Estimation statistics
        self.estimation_stats = {}
        
    def fit(self,
            returns: pd.DataFrame,
            regimes: pd.Series,
            use_shrinkage: bool = True,
            min_observations: int = 30) -> 'RegimeConditionalCMA':
        """
        Fit regime-conditional CMAs.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset return data
        regimes : pd.Series
            Regime labels aligned with returns
        use_shrinkage : bool
            Whether to use shrinkage estimation for covariance
        min_observations : int
            Minimum observations required per regime
        
        Returns:
        --------
        RegimeConditionalCMA
            Fitted CMA calculator
        """
        self.logger.info("Fitting regime-conditional CMAs")
        
        # Align data
        aligned_returns, aligned_regimes = returns.align(regimes, join='inner', axis=0)
        self.asset_names = list(aligned_returns.columns)
        
        # Clean returns
        for column in aligned_returns.columns:
            aligned_returns[column] = clean_returns(aligned_returns[column])
        
        # Calculate unconditional CMA (baseline)
        self.unconditional_cma = self._calculate_cma(
            aligned_returns, 
            "unconditional",
            use_shrinkage=use_shrinkage
        )
        
        # Calculate regime-specific CMAs
        unique_regimes = sorted(aligned_regimes.unique())
        
        for regime in unique_regimes:
            regime_mask = aligned_regimes == regime
            regime_returns = aligned_returns[regime_mask]
            
            if len(regime_returns) < min_observations:
                self.logger.warning(
                    f"Insufficient observations for regime {regime}: "
                    f"{len(regime_returns)} < {min_observations}. Using unconditional CMA."
                )
                self.regime_cmas[regime] = self.unconditional_cma.copy()
                self.estimation_stats[regime] = {
                    'n_observations': len(regime_returns),
                    'method': 'unconditional_fallback'
                }
            else:
                self.regime_cmas[regime] = self._calculate_cma(
                    regime_returns,
                    f"regime_{regime}",
                    use_shrinkage=use_shrinkage
                )
                self.estimation_stats[regime] = {
                    'n_observations': len(regime_returns),
                    'method': 'shrinkage' if use_shrinkage else 'sample',
                    'start_date': regime_returns.index.min(),
                    'end_date': regime_returns.index.max()
                }
        
        # Validate CMAs
        self._validate_cmas()
        
        # Log summary
        self._log_cma_summary()
        
        return self
    
    def get_cma(self, 
                regime: int,
                fallback_to_unconditional: bool = True) -> Dict[str, np.ndarray]:
        """
        Get CMA for a specific regime.
        
        Parameters:
        -----------
        regime : int
            Regime identifier
        fallback_to_unconditional : bool
            Whether to fallback to unconditional CMA if regime not found
        
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with 'mu' (expected returns) and 'sigma' (covariance matrix)
        """
        if regime in self.regime_cmas:
            return self.regime_cmas[regime].copy()
        elif fallback_to_unconditional and self.unconditional_cma is not None:
            self.logger.warning(f"Regime {regime} not found, using unconditional CMA")
            return self.unconditional_cma.copy()
        else:
            raise ValueError(f"No CMA available for regime {regime}")
    
    def get_expected_returns(self, regime: int) -> np.ndarray:
        """
        Get expected returns for a specific regime.
        
        Parameters:
        -----------
        regime : int
            Regime identifier
        
        Returns:
        --------
        np.ndarray
            Expected returns vector
        """
        cma = self.get_cma(regime)
        return cma['mu']
    
    def get_covariance_matrix(self, regime: int) -> np.ndarray:
        """
        Get covariance matrix for a specific regime.
        
        Parameters:
        -----------
        regime : int
            Regime identifier
        
        Returns:
        --------
        np.ndarray
            Covariance matrix
        """
        cma = self.get_cma(regime)
        return cma['sigma']
    
    def compare_regimes(self) -> pd.DataFrame:
        """
        Compare expected returns and risk across regimes.
        
        Returns:
        --------
        pd.DataFrame
            Comparison of regime characteristics
        """
        comparison_data = []
        
        # Add unconditional baseline
        if self.unconditional_cma is not None:
            mu = self.unconditional_cma['mu']
            sigma = self.unconditional_cma['sigma']
            
            comparison_data.append({
                'regime': 'unconditional',
                'avg_return': mu.mean(),
                'avg_volatility': np.sqrt(np.diag(sigma)).mean(),
                'portfolio_vol': np.sqrt(np.ones(len(mu)) @ sigma @ np.ones(len(mu)) / len(mu)**2),
                'n_observations': len(self.asset_names)
            })
        
        # Add regime-specific data
        for regime, cma in self.regime_cmas.items():
            mu = cma['mu']
            sigma = cma['sigma']
            stats = self.estimation_stats.get(regime, {})
            
            comparison_data.append({
                'regime': regime,
                'avg_return': mu.mean(),
                'avg_volatility': np.sqrt(np.diag(sigma)).mean(),
                'portfolio_vol': np.sqrt(np.ones(len(mu)) @ sigma @ np.ones(len(mu)) / len(mu)**2),
                'n_observations': stats.get('n_observations', 0)
            })
        
        return pd.DataFrame(comparison_data).set_index('regime')
    
    def get_regime_rankings(self, regime: int) -> Dict[str, int]:
        """
        Get asset rankings for a specific regime (best to worst expected returns).
        
        Parameters:
        -----------
        regime : int
            Regime identifier
        
        Returns:
        --------
        Dict[str, int]
            Asset rankings (1 = best, len(assets) = worst)
        """
        mu = self.get_expected_returns(regime)
        
        # Create rankings (higher return = better rank)
        rankings = {}
        sorted_indices = np.argsort(-mu)  # Descending order
        
        for rank, idx in enumerate(sorted_indices, 1):
            asset_name = self.asset_names[idx]
            rankings[asset_name] = rank
        
        return rankings
    
    def _calculate_cma(self,
                      returns: pd.DataFrame,
                      label: str,
                      use_shrinkage: bool = True) -> Dict[str, np.ndarray]:
        """
        Calculate expected returns and covariance matrix.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Return data
        label : str
            Label for logging
        use_shrinkage : bool
            Whether to use shrinkage estimation
        
        Returns:
        --------
        Dict[str, np.ndarray]
            CMA dictionary with 'mu' and 'sigma'
        """
        self.logger.debug(f"Calculating CMA for {label}")
        
        # Expected returns (simple mean)
        mu = returns.mean().values
        
        # Covariance matrix
        if use_shrinkage:
            sigma = self._shrinkage_covariance(returns)
        else:
            sigma = returns.cov().values
        
        # Ensure covariance matrix is positive definite
        sigma = self._ensure_positive_definite(sigma)
        
        # Annualize (assuming daily returns)
        mu_annual = mu * 252
        sigma_annual = sigma * 252
        
        cma = {
            'mu': mu_annual,
            'sigma': sigma_annual,
            'mu_daily': mu,
            'sigma_daily': sigma,
            'n_assets': len(mu),
            'n_observations': len(returns)
        }
        
        return cma
    
    def _shrinkage_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate shrinkage covariance matrix (Ledoit-Wolf).
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Return data
        
        Returns:
        --------
        np.ndarray
            Shrinkage covariance matrix
        """
        try:
            from sklearn.covariance import LedoitWolf
            
            lw = LedoitWolf()
            lw.fit(returns.values)
            
            self.logger.debug(f"Shrinkage intensity: {lw.shrinkage_:.3f}")
            
            return lw.covariance_
            
        except ImportError:
            self.logger.warning("sklearn not available, using sample covariance")
            return returns.cov().values
        except Exception as e:
            self.logger.error(f"Shrinkage estimation failed: {e}, using sample covariance")
            return returns.cov().values
    
    def _ensure_positive_definite(self, sigma: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
        """
        Ensure covariance matrix is positive definite.
        
        Parameters:
        -----------
        sigma : np.ndarray
            Covariance matrix
        min_eigenvalue : float
            Minimum eigenvalue to ensure
        
        Returns:
        --------
        np.ndarray
            Positive definite covariance matrix
        """
        try:
            # Check if already positive definite
            eigenvalues = np.linalg.eigvals(sigma)
            
            if np.min(eigenvalues) > min_eigenvalue:
                return sigma
            
            # Fix negative eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(sigma)
            eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
            
            # Reconstruct matrix
            sigma_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            self.logger.debug("Fixed covariance matrix to be positive definite")
            
            return sigma_fixed
            
        except Exception as e:
            self.logger.error(f"Failed to fix covariance matrix: {e}")
            # Fallback: add small value to diagonal
            return sigma + np.eye(sigma.shape[0]) * min_eigenvalue
    
    def _validate_cmas(self):
        """Validate calculated CMAs."""
        issues = []
        
        for regime, cma in self.regime_cmas.items():
            mu = cma['mu']
            sigma = cma['sigma']
            
            # Check for NaN values
            if np.any(np.isnan(mu)) or np.any(np.isnan(sigma)):
                issues.append(f"Regime {regime}: NaN values in CMA")
            
            # Check for unrealistic returns (> 100% or < -50% annually)
            if np.any(mu > 1.0) or np.any(mu < -0.5):
                issues.append(f"Regime {regime}: Unrealistic expected returns")
            
            # Check covariance matrix properties
            try:
                eigenvalues = np.linalg.eigvals(sigma)
                if np.min(eigenvalues) <= 0:
                    issues.append(f"Regime {regime}: Covariance matrix not positive definite")
            except Exception:
                issues.append(f"Regime {regime}: Invalid covariance matrix")
        
        if issues:
            for issue in issues:
                self.logger.warning(issue)
        else:
            self.logger.info("All CMAs passed validation")
    
    def _log_cma_summary(self):
        """Log summary of calculated CMAs."""
        self.logger.info(f"Calculated CMAs for {len(self.regime_cmas)} regimes")
        
        for regime, cma in self.regime_cmas.items():
            mu = cma['mu']
            sigma = cma['sigma']
            stats = self.estimation_stats.get(regime, {})
            
            avg_return = mu.mean() * 100  # Convert to percentage
            avg_vol = np.sqrt(np.diag(sigma)).mean() * 100
            
            self.logger.info(
                f"Regime {regime}: Avg return {avg_return:.2f}%, "
                f"Avg volatility {avg_vol:.2f}%, "
                f"Observations: {stats.get('n_observations', 0)}"
            )
    
    def save_cmas(self, filepath: str):
        """
        Save CMAs to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save file
        """
        try:
            data_to_save = {
                'regime_cmas': self.regime_cmas,
                'unconditional_cma': self.unconditional_cma,
                'asset_names': self.asset_names,
                'estimation_stats': self.estimation_stats
            }
            
            np.savez_compressed(filepath, **data_to_save)
            self.logger.info(f"CMAs saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save CMAs: {e}")
    
    def load_cmas(self, filepath: str):
        """
        Load CMAs from file.
        
        Parameters:
        -----------
        filepath : str
            Path to load file
        """
        try:
            data = np.load(filepath, allow_pickle=True)
            
            self.regime_cmas = data['regime_cmas'].item()
            self.unconditional_cma = data['unconditional_cma'].item()
            self.asset_names = data['asset_names'].tolist()
            self.estimation_stats = data['estimation_stats'].item()
            
            self.logger.info(f"CMAs loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load CMAs: {e}")


class DynamicCMACalculator:
    """
    Calculator for dynamic CMAs that can adapt over time.
    """
    
    def __init__(self, config: Config):
        """
        Initialize dynamic CMA calculator.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        self.base_calculator = RegimeConditionalCMA(config)
        
        # Rolling window parameters
        self.rolling_window = None
        self.decay_factor = 0.95  # For exponential weighting
        
    def fit_rolling(self,
                   returns: pd.DataFrame,
                   regimes: pd.Series,
                   window_size: int = 252 * 3,  # 3 years
                   min_regime_obs: int = 30) -> 'DynamicCMACalculator':
        """
        Fit rolling regime-conditional CMAs.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset return data
        regimes : pd.Series
            Regime labels
        window_size : int
            Rolling window size in periods
        min_regime_obs : int
            Minimum observations per regime
        
        Returns:
        --------
        DynamicCMACalculator
            Fitted calculator
        """
        self.logger.info(f"Fitting rolling CMAs with window size {window_size}")
        
        self.rolling_window = window_size
        self.rolling_cmas = {}
        
        # Align data
        aligned_returns, aligned_regimes = returns.align(regimes, join='inner', axis=0)
        
        # Calculate rolling CMAs
        for i in range(window_size, len(aligned_returns)):
            end_date = aligned_returns.index[i]
            start_idx = i - window_size
            
            window_returns = aligned_returns.iloc[start_idx:i]
            window_regimes = aligned_regimes.iloc[start_idx:i]
            
            # Fit CMA for this window
            cma_calc = RegimeConditionalCMA(self.config)
            cma_calc.fit(window_returns, window_regimes, min_observations=min_regime_obs)
            
            self.rolling_cmas[end_date] = cma_calc
        
        self.logger.info(f"Calculated rolling CMAs for {len(self.rolling_cmas)} time points")
        
        return self
    
    def get_cma_at_date(self,
                       date: datetime,
                       regime: int) -> Dict[str, np.ndarray]:
        """
        Get CMA for specific date and regime.
        
        Parameters:
        -----------
        date : datetime
            Date for CMA lookup
        regime : int
            Regime identifier
        
        Returns:
        --------
        Dict[str, np.ndarray]
            CMA dictionary
        """
        if not self.rolling_cmas:
            raise ValueError("Rolling CMAs not fitted. Call fit_rolling() first.")
        
        # Find nearest available date
        available_dates = list(self.rolling_cmas.keys())
        nearest_date = min(available_dates, key=lambda x: abs((x - date).total_seconds()))
        
        cma_calc = self.rolling_cmas[nearest_date]
        return cma_calc.get_cma(regime)
    
    def get_cma_evolution(self, regime: int, asset_idx: int = 0) -> pd.Series:
        """
        Get evolution of expected return for specific regime and asset.
        
        Parameters:
        -----------
        regime : int
            Regime identifier
        asset_idx : int
            Asset index
        
        Returns:
        --------
        pd.Series
            Time series of expected returns
        """
        evolution_data = {}
        
        for date, cma_calc in self.rolling_cmas.items():
            try:
                mu = cma_calc.get_expected_returns(regime)
                evolution_data[date] = mu[asset_idx]
            except (KeyError, IndexError):
                evolution_data[date] = np.nan
        
        return pd.Series(evolution_data)