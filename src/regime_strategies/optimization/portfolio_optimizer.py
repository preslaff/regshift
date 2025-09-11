"""
Portfolio optimization using Modern Portfolio Theory with regime-conditional inputs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from loguru import logger
from dataclasses import dataclass
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("CVXPY not available. Using scipy optimization only.")

from scipy.optimize import minimize
from scipy.optimize import Bounds, LinearConstraint

from ..config import Config
from ..utils.metrics import sharpe_ratio


@dataclass
class OptimizationResult:
    """
    Container for optimization results.
    """
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    optimization_status: str
    method: str
    regime: Optional[int] = None
    optimization_time: Optional[float] = None
    additional_metrics: Optional[Dict[str, float]] = None


class MaxSharpeOptimizer:
    """
    Maximum Sharpe ratio portfolio optimizer.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Max Sharpe optimizer.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        
    def optimize(self,
                mu: np.ndarray,
                sigma: np.ndarray,
                risk_free_rate: Optional[float] = None) -> OptimizationResult:
        """
        Optimize portfolio for maximum Sharpe ratio.
        
        Parameters:
        -----------
        mu : np.ndarray
            Expected returns vector
        sigma : np.ndarray
            Covariance matrix
        risk_free_rate : float, optional
            Risk-free rate
        
        Returns:
        --------
        OptimizationResult
            Optimization results
        """
        n_assets = len(mu)
        risk_free_rate = risk_free_rate or self.config.optimization.risk_free_rate
        
        if CVXPY_AVAILABLE:
            return self._optimize_cvxpy(mu, sigma, risk_free_rate)
        else:
            return self._optimize_scipy(mu, sigma, risk_free_rate)
    
    def _optimize_cvxpy(self,
                       mu: np.ndarray,
                       sigma: np.ndarray,
                       risk_free_rate: float) -> OptimizationResult:
        """Optimize using CVXPY."""
        n_assets = len(mu)
        
        # Decision variables
        w = cp.Variable(n_assets)
        
        # Portfolio return and risk
        portfolio_return = mu.T @ w
        portfolio_risk = cp.quad_form(w, sigma)
        
        # Objective: maximize Sharpe ratio
        # We maximize (return - rf) / sqrt(risk) by maximizing (return - rf) subject to risk = 1
        # This is equivalent to the original problem
        objective = cp.Maximize(portfolio_return - risk_free_rate)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            portfolio_risk == 1,  # Normalize risk
        ]
        
        # Add bounds if specified
        if self.config.optimization.long_only:
            constraints.append(w >= 0)
        
        if hasattr(self.config.optimization, 'max_weight'):
            constraints.append(w <= self.config.optimization.max_weight)
        
        if hasattr(self.config.optimization, 'min_weight'):
            constraints.append(w >= self.config.optimization.min_weight)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(
                solver=self.config.optimization.solver,
                max_iters=self.config.optimization.max_iters
            )
            
            if problem.status == cp.OPTIMAL:
                weights = w.value
                weights = weights / np.sum(weights)  # Renormalize
                
                # Calculate portfolio metrics
                port_return = np.dot(mu, weights)
                port_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
                port_sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=port_return,
                    expected_volatility=port_vol,
                    sharpe_ratio=port_sharpe,
                    optimization_status="optimal",
                    method="max_sharpe_cvxpy"
                )
            else:
                self.logger.warning(f"CVXPY optimization failed: {problem.status}")
                return self._fallback_optimization(mu, sigma, risk_free_rate)
                
        except Exception as e:
            self.logger.error(f"CVXPY optimization error: {e}")
            return self._fallback_optimization(mu, sigma, risk_free_rate)
    
    def _optimize_scipy(self,
                       mu: np.ndarray,
                       sigma: np.ndarray,
                       risk_free_rate: float) -> OptimizationResult:
        """Optimize using scipy."""
        n_assets = len(mu)
        
        # Objective function: minimize negative Sharpe ratio
        def objective(w):
            port_return = np.dot(mu, w)
            port_vol = np.sqrt(np.dot(w, np.dot(sigma, w)))
            
            if port_vol == 0:
                return -np.inf  # Avoid division by zero
            
            sharpe = (port_return - risk_free_rate) / port_vol
            return -sharpe  # Minimize negative Sharpe
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
        
        # Bounds
        if self.config.optimization.long_only:
            bounds = [(0, self.config.optimization.max_weight) for _ in range(n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.optimization.max_iters}
            )
            
            if result.success:
                weights = result.x
                weights = weights / np.sum(weights)  # Renormalize
                
                # Calculate portfolio metrics
                port_return = np.dot(mu, weights)
                port_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
                port_sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=port_return,
                    expected_volatility=port_vol,
                    sharpe_ratio=port_sharpe,
                    optimization_status="optimal",
                    method="max_sharpe_scipy"
                )
            else:
                self.logger.warning(f"Scipy optimization failed: {result.message}")
                return self._fallback_optimization(mu, sigma, risk_free_rate)
                
        except Exception as e:
            self.logger.error(f"Scipy optimization error: {e}")
            return self._fallback_optimization(mu, sigma, risk_free_rate)
    
    def _fallback_optimization(self,
                              mu: np.ndarray,
                              sigma: np.ndarray,
                              risk_free_rate: float) -> OptimizationResult:
        """Fallback to equal weights if optimization fails."""
        n_assets = len(mu)
        weights = np.ones(n_assets) / n_assets
        
        port_return = np.dot(mu, weights)
        port_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
        port_sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        
        return OptimizationResult(
            weights=weights,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=port_sharpe,
            optimization_status="fallback",
            method="equal_weights"
        )


class MinimumVarianceOptimizer:
    """
    Minimum variance portfolio optimizer.
    """
    
    def __init__(self, config: Config):
        """
        Initialize minimum variance optimizer.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
    
    def optimize(self,
                mu: np.ndarray,
                sigma: np.ndarray) -> OptimizationResult:
        """
        Optimize portfolio for minimum variance.
        
        Parameters:
        -----------
        mu : np.ndarray
            Expected returns vector (not used directly)
        sigma : np.ndarray
            Covariance matrix
        
        Returns:
        --------
        OptimizationResult
            Optimization results
        """
        if CVXPY_AVAILABLE:
            return self._optimize_cvxpy(mu, sigma)
        else:
            return self._optimize_scipy(mu, sigma)
    
    def _optimize_cvxpy(self,
                       mu: np.ndarray,
                       sigma: np.ndarray) -> OptimizationResult:
        """Optimize using CVXPY."""
        n_assets = len(mu)
        
        # Decision variables
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance
        objective = cp.Minimize(cp.quad_form(w, sigma))
        
        # Constraints
        constraints = [cp.sum(w) == 1]
        
        if self.config.optimization.long_only:
            constraints.append(w >= 0)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=self.config.optimization.solver)
            
            if problem.status == cp.OPTIMAL:
                weights = w.value
                weights = weights / np.sum(weights)
                
                port_return = np.dot(mu, weights)
                port_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
                port_sharpe = port_return / port_vol if port_vol > 0 else 0
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=port_return,
                    expected_volatility=port_vol,
                    sharpe_ratio=port_sharpe,
                    optimization_status="optimal",
                    method="min_variance_cvxpy"
                )
            else:
                return self._fallback_optimization(mu, sigma)
                
        except Exception as e:
            self.logger.error(f"CVXPY min variance optimization error: {e}")
            return self._fallback_optimization(mu, sigma)
    
    def _optimize_scipy(self,
                       mu: np.ndarray,
                       sigma: np.ndarray) -> OptimizationResult:
        """Optimize using scipy."""
        n_assets = len(mu)
        
        # Objective: minimize variance
        def objective(w):
            return np.dot(w, np.dot(sigma, w))
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        if self.config.optimization.long_only:
            bounds = [(0, 1) for _ in range(n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                weights = weights / np.sum(weights)
                
                port_return = np.dot(mu, weights)
                port_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
                port_sharpe = port_return / port_vol if port_vol > 0 else 0
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=port_return,
                    expected_volatility=port_vol,
                    sharpe_ratio=port_sharpe,
                    optimization_status="optimal",
                    method="min_variance_scipy"
                )
            else:
                return self._fallback_optimization(mu, sigma)
                
        except Exception as e:
            self.logger.error(f"Scipy min variance optimization error: {e}")
            return self._fallback_optimization(mu, sigma)
    
    def _fallback_optimization(self,
                              mu: np.ndarray,
                              sigma: np.ndarray) -> OptimizationResult:
        """Fallback optimization."""
        n_assets = len(mu)
        weights = np.ones(n_assets) / n_assets
        
        port_return = np.dot(mu, weights)
        port_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
        port_sharpe = port_return / port_vol if port_vol > 0 else 0
        
        return OptimizationResult(
            weights=weights,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=port_sharpe,
            optimization_status="fallback",
            method="equal_weights"
        )


class RiskParityOptimizer:
    """
    Risk parity portfolio optimizer.
    """
    
    def __init__(self, config: Config):
        """
        Initialize risk parity optimizer.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
    
    def optimize(self,
                mu: np.ndarray,
                sigma: np.ndarray) -> OptimizationResult:
        """
        Optimize portfolio for equal risk contribution.
        
        Parameters:
        -----------
        mu : np.ndarray
            Expected returns vector
        sigma : np.ndarray
            Covariance matrix
        
        Returns:
        --------
        OptimizationResult
            Optimization results
        """
        return self._optimize_risk_parity(mu, sigma)
    
    def _optimize_risk_parity(self,
                             mu: np.ndarray,
                             sigma: np.ndarray) -> OptimizationResult:
        """Risk parity optimization using iterative approach."""
        n_assets = len(mu)
        
        # Objective: minimize sum of squared differences in risk contributions
        def objective(w):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(np.dot(w, np.dot(sigma, w)))
            
            if portfolio_vol == 0:
                return 1e6  # Large penalty for zero volatility
            
            marginal_contrib = np.dot(sigma, w) / portfolio_vol
            risk_contrib = w * marginal_contrib
            
            # Target: equal risk contribution
            target_contrib = portfolio_vol / n_assets
            
            # Sum of squared deviations from target
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        if self.config.optimization.long_only:
            bounds = [(0.01, 1) for _ in range(n_assets)]  # Small minimum to avoid zeros
        else:
            bounds = [(-1, 1) for _ in range(n_assets)]
        
        # Initial guess: inverse volatility weights
        try:
            vol_weights = 1 / np.sqrt(np.diag(sigma))
            x0 = vol_weights / np.sum(vol_weights)
        except:
            x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
                weights = weights / np.sum(weights)
                
                port_return = np.dot(mu, weights)
                port_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
                port_sharpe = port_return / port_vol if port_vol > 0 else 0
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=port_return,
                    expected_volatility=port_vol,
                    sharpe_ratio=port_sharpe,
                    optimization_status="optimal",
                    method="risk_parity"
                )
            else:
                self.logger.warning(f"Risk parity optimization failed: {result.message}")
                return self._fallback_optimization(mu, sigma)
                
        except Exception as e:
            self.logger.error(f"Risk parity optimization error: {e}")
            return self._fallback_optimization(mu, sigma)
    
    def _fallback_optimization(self,
                              mu: np.ndarray,
                              sigma: np.ndarray) -> OptimizationResult:
        """Fallback to inverse volatility weights."""
        try:
            vol_weights = 1 / np.sqrt(np.diag(sigma))
            weights = vol_weights / np.sum(vol_weights)
        except:
            n_assets = len(mu)
            weights = np.ones(n_assets) / n_assets
        
        port_return = np.dot(mu, weights)
        port_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
        port_sharpe = port_return / port_vol if port_vol > 0 else 0
        
        return OptimizationResult(
            weights=weights,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=port_sharpe,
            optimization_status="fallback",
            method="inverse_volatility"
        )


class PortfolioOptimizer:
    """
    Main portfolio optimizer that orchestrates different optimization methods.
    """
    
    def __init__(self, config: Config):
        """
        Initialize portfolio optimizer.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        
        # Initialize optimizers
        self.optimizers = {
            'max_sharpe': MaxSharpeOptimizer(config),
            'min_variance': MinimumVarianceOptimizer(config),
            'risk_parity': RiskParityOptimizer(config)
        }
        
        self.asset_names = []
    
    def optimize(self,
                mu: np.ndarray,
                sigma: np.ndarray,
                method: Optional[str] = None,
                asset_names: Optional[List[str]] = None,
                regime: Optional[int] = None) -> OptimizationResult:
        """
        Optimize portfolio using specified method.
        
        Parameters:
        -----------
        mu : np.ndarray
            Expected returns vector
        sigma : np.ndarray
            Covariance matrix
        method : str, optional
            Optimization method
        asset_names : List[str], optional
            Asset names for results
        regime : int, optional
            Regime identifier
        
        Returns:
        --------
        OptimizationResult
            Optimization results
        """
        method = method or self.config.optimization.default_method
        self.asset_names = asset_names or self.asset_names
        
        if method not in self.optimizers:
            raise ValueError(f"Unknown optimization method: {method}. "
                           f"Available: {list(self.optimizers.keys())}")
        
        self.logger.info(f"Optimizing portfolio using {method} method")
        
        # Validate inputs
        self._validate_inputs(mu, sigma)
        
        # Optimize
        start_time = datetime.now()
        
        try:
            if method == 'max_sharpe':
                result = self.optimizers[method].optimize(mu, sigma)
            else:
                result = self.optimizers[method].optimize(mu, sigma)
            
            # Add metadata
            result.regime = regime
            result.optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Log results
            self._log_optimization_results(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return self._emergency_fallback(mu, sigma, method, regime)
    
    def optimize_regime_conditional(self,
                                  cma_dict: Dict[str, np.ndarray],
                                  method: Optional[str] = None,
                                  regime: Optional[int] = None) -> OptimizationResult:
        """
        Optimize portfolio using regime-conditional CMAs.
        
        Parameters:
        -----------
        cma_dict : Dict[str, np.ndarray]
            CMA dictionary with 'mu' and 'sigma'
        method : str, optional
            Optimization method
        regime : int, optional
            Regime identifier
        
        Returns:
        --------
        OptimizationResult
            Optimization results
        """
        mu = cma_dict['mu']
        sigma = cma_dict['sigma']
        
        return self.optimize(mu, sigma, method=method, regime=regime)
    
    def compare_methods(self,
                       mu: np.ndarray,
                       sigma: np.ndarray,
                       methods: Optional[List[str]] = None) -> Dict[str, OptimizationResult]:
        """
        Compare multiple optimization methods.
        
        Parameters:
        -----------
        mu : np.ndarray
            Expected returns vector
        sigma : np.ndarray
            Covariance matrix
        methods : List[str], optional
            Methods to compare
        
        Returns:
        --------
        Dict[str, OptimizationResult]
            Results for each method
        """
        methods = methods or list(self.optimizers.keys())
        results = {}
        
        for method in methods:
            try:
                results[method] = self.optimize(mu, sigma, method=method)
                self.logger.info(f"{method}: Sharpe={results[method].sharpe_ratio:.3f}")
            except Exception as e:
                self.logger.error(f"Failed to optimize with {method}: {e}")
        
        return results
    
    def _validate_inputs(self, mu: np.ndarray, sigma: np.ndarray):
        """Validate optimization inputs."""
        if len(mu) != sigma.shape[0] or len(mu) != sigma.shape[1]:
            raise ValueError("Dimension mismatch between mu and sigma")
        
        if np.any(np.isnan(mu)) or np.any(np.isnan(sigma)):
            raise ValueError("NaN values in inputs")
        
        if np.any(np.isinf(mu)) or np.any(np.isinf(sigma)):
            raise ValueError("Infinite values in inputs")
        
        # Check if covariance matrix is positive semidefinite
        eigenvals = np.linalg.eigvals(sigma)
        if np.any(eigenvals < -1e-8):
            self.logger.warning("Covariance matrix is not positive semidefinite")
    
    def _log_optimization_results(self, result: OptimizationResult):
        """Log optimization results."""
        self.logger.info(
            f"Optimization completed: {result.method} | "
            f"Status: {result.optimization_status} | "
            f"Return: {result.expected_return:.4f} | "
            f"Vol: {result.expected_volatility:.4f} | "
            f"Sharpe: {result.sharpe_ratio:.4f}"
        )
        
        if len(self.asset_names) == len(result.weights):
            for i, (asset, weight) in enumerate(zip(self.asset_names, result.weights)):
                if weight > 0.01:  # Only log significant weights
                    self.logger.debug(f"{asset}: {weight:.3f}")
    
    def _emergency_fallback(self,
                           mu: np.ndarray,
                           sigma: np.ndarray,
                           method: str,
                           regime: Optional[int]) -> OptimizationResult:
        """Emergency fallback to equal weights."""
        n_assets = len(mu)
        weights = np.ones(n_assets) / n_assets
        
        port_return = np.dot(mu, weights)
        port_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
        port_sharpe = port_return / port_vol if port_vol > 0 else 0
        
        return OptimizationResult(
            weights=weights,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=port_sharpe,
            optimization_status="emergency_fallback",
            method=f"{method}_failed",
            regime=regime
        )