"""
Risk models and risk management utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger

from ..config import Config
from ..utils.metrics import calculate_metrics


class RiskModel:
    """
    Risk model for portfolio risk assessment and management.
    """
    
    def __init__(self, config: Config):
        """
        Initialize risk model.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        
    def calculate_portfolio_risk(self,
                               weights: np.ndarray,
                               sigma: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        sigma : np.ndarray
            Covariance matrix
        
        Returns:
        --------
        Dict[str, float]
            Risk metrics
        """
        portfolio_variance = np.dot(weights, np.dot(sigma, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Risk contributions
        marginal_contrib = np.dot(sigma, weights)
        risk_contributions = weights * marginal_contrib / portfolio_variance if portfolio_variance > 0 else np.zeros_like(weights)
        
        # Concentration measures
        herfindahl_index = np.sum(weights ** 2)
        effective_n_assets = 1 / herfindahl_index if herfindahl_index > 0 else len(weights)
        
        # Maximum weight
        max_weight = np.max(np.abs(weights))
        
        risk_metrics = {
            'portfolio_volatility': portfolio_volatility,
            'portfolio_variance': portfolio_variance,
            'risk_contribution_hhi': np.sum(risk_contributions ** 2),
            'weight_herfindahl': herfindahl_index,
            'effective_n_assets': effective_n_assets,
            'max_weight': max_weight,
            'max_risk_contribution': np.max(risk_contributions) if len(risk_contributions) > 0 else 0,
        }
        
        return risk_metrics
    
    def calculate_risk_attribution(self,
                                 weights: np.ndarray,
                                 sigma: np.ndarray,
                                 asset_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate detailed risk attribution.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        sigma : np.ndarray
            Covariance matrix
        asset_names : List[str], optional
            Asset names
        
        Returns:
        --------
        pd.DataFrame
            Risk attribution details
        """
        portfolio_variance = np.dot(weights, np.dot(sigma, weights))
        
        if portfolio_variance == 0:
            # Handle zero variance case
            n_assets = len(weights)
            attribution_data = {
                'weight': weights,
                'volatility': np.zeros(n_assets),
                'marginal_risk': np.zeros(n_assets),
                'risk_contribution': np.zeros(n_assets),
                'risk_contribution_pct': np.zeros(n_assets)
            }
        else:
            # Individual asset volatilities
            asset_volatilities = np.sqrt(np.diag(sigma))
            
            # Marginal risk contributions
            marginal_contrib = np.dot(sigma, weights) / np.sqrt(portfolio_variance)
            
            # Component risk contributions
            risk_contributions = weights * marginal_contrib
            risk_contrib_pct = risk_contributions / np.sum(risk_contributions) * 100
            
            attribution_data = {
                'weight': weights,
                'volatility': asset_volatilities,
                'marginal_risk': marginal_contrib,
                'risk_contribution': risk_contributions,
                'risk_contribution_pct': risk_contrib_pct
            }
        
        # Create DataFrame
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(len(weights))]
        
        attribution_df = pd.DataFrame(attribution_data, index=asset_names)
        
        return attribution_df
    
    def stress_test_portfolio(self,
                            weights: np.ndarray,
                            returns: pd.DataFrame,
                            scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Perform stress testing on portfolio.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        returns : pd.DataFrame
            Historical return data
        scenarios : Dict[str, Dict[str, float]]
            Stress scenarios (asset_name -> shock)
        
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Stress test results
        """
        results = {}
        
        # Base portfolio return
        base_returns = returns.mean().values
        base_portfolio_return = np.dot(weights, base_returns)
        
        for scenario_name, shocks in scenarios.items():
            # Apply shocks
            shocked_returns = base_returns.copy()
            
            for i, asset_name in enumerate(returns.columns):
                if asset_name in shocks:
                    shocked_returns[i] += shocks[asset_name]
            
            # Calculate stressed portfolio return
            stressed_portfolio_return = np.dot(weights, shocked_returns)
            
            results[scenario_name] = {
                'portfolio_return_change': stressed_portfolio_return - base_portfolio_return,
                'portfolio_return_stressed': stressed_portfolio_return,
                'relative_change': (stressed_portfolio_return - base_portfolio_return) / abs(base_portfolio_return) if base_portfolio_return != 0 else 0
            }
        
        return results
    
    def calculate_diversification_ratio(self,
                                      weights: np.ndarray,
                                      sigma: np.ndarray) -> float:
        """
        Calculate diversification ratio.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        sigma : np.ndarray
            Covariance matrix
        
        Returns:
        --------
        float
            Diversification ratio
        """
        # Weighted average of individual volatilities
        individual_vols = np.sqrt(np.diag(sigma))
        weighted_avg_vol = np.dot(weights, individual_vols)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
        
        if portfolio_vol == 0:
            return 1.0
        
        diversification_ratio = weighted_avg_vol / portfolio_vol
        
        return diversification_ratio
    
    def calculate_maximum_drawdown_estimate(self,
                                          weights: np.ndarray,
                                          returns: pd.DataFrame,
                                          confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Estimate maximum drawdown using historical simulation.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        returns : pd.DataFrame
            Historical return data
        confidence_level : float
            Confidence level for estimation
        
        Returns:
        --------
        Dict[str, float]
            Drawdown estimates
        """
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate cumulative returns and drawdowns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        
        # Calculate statistics
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns[drawdowns < 0].mean() if any(drawdowns < 0) else 0
        
        # VaR-style estimate at confidence level
        drawdown_var = drawdowns.quantile(1 - confidence_level)
        
        return {
            'max_drawdown_historical': max_drawdown,
            'average_drawdown': avg_drawdown,
            'drawdown_var': drawdown_var,
            'drawdown_volatility': drawdowns.std()
        }
    
    def check_risk_limits(self,
                         weights: np.ndarray,
                         sigma: np.ndarray,
                         limits: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if portfolio satisfies risk limits.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        sigma : np.ndarray
            Covariance matrix
        limits : Dict[str, float]
            Risk limits to check
        
        Returns:
        --------
        Dict[str, bool]
            Limit satisfaction (True = within limit)
        """
        risk_metrics = self.calculate_portfolio_risk(weights, sigma)
        
        limit_checks = {}
        
        for limit_name, limit_value in limits.items():
            if limit_name in risk_metrics:
                current_value = risk_metrics[limit_name]
                
                if limit_name in ['portfolio_volatility', 'max_weight', 'weight_herfindahl']:
                    # Upper limits
                    limit_checks[limit_name] = current_value <= limit_value
                elif limit_name in ['effective_n_assets']:
                    # Lower limits
                    limit_checks[limit_name] = current_value >= limit_value
                else:
                    # Default to upper limit
                    limit_checks[limit_name] = current_value <= limit_value
            else:
                self.logger.warning(f"Unknown risk limit: {limit_name}")
                limit_checks[limit_name] = True  # Assume satisfied if unknown
        
        return limit_checks
    
    def generate_risk_report(self,
                           weights: np.ndarray,
                           sigma: np.ndarray,
                           asset_names: Optional[List[str]] = None,
                           regime: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        sigma : np.ndarray
            Covariance matrix
        asset_names : List[str], optional
            Asset names
        regime : int, optional
            Current regime
        
        Returns:
        --------
        Dict[str, Any]
            Comprehensive risk report
        """
        # Basic risk metrics
        risk_metrics = self.calculate_portfolio_risk(weights, sigma)
        
        # Risk attribution
        risk_attribution = self.calculate_risk_attribution(weights, sigma, asset_names)
        
        # Diversification
        diversification_ratio = self.calculate_diversification_ratio(weights, sigma)
        
        # Create risk report
        report = {
            'regime': regime,
            'timestamp': datetime.now(),
            'risk_metrics': risk_metrics,
            'diversification_ratio': diversification_ratio,
            'risk_attribution': risk_attribution.to_dict(),
            'top_risk_contributors': risk_attribution.nlargest(3, 'risk_contribution_pct').index.tolist(),
            'concentration_warning': risk_metrics['max_weight'] > 0.3,
            'diversification_warning': diversification_ratio < 1.2
        }
        
        return report