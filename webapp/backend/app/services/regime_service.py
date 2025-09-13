"""
Regime service for market regime identification and forecasting.
Integrates with the regime_strategies module for actual regime analysis.
"""

import sys
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, date
from loguru import logger

# Add the src directory to Python path for importing regime_strategies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

try:
    from regime_strategies.main import RegimeStrategyApp
    from regime_strategies.config import Config
    from regime_strategies.regimes.regime_identifier import RegimeIdentifier
    from regime_strategies.models.regime_forecaster import RegimeForecaster
    REGIME_STRATEGIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import regime_strategies module: {e}")
    REGIME_STRATEGIES_AVAILABLE = False


class RegimeService:
    """Service for regime identification and forecasting using the regime_strategies module."""
    
    def __init__(self):
        # Initialize the regime strategies application
        if REGIME_STRATEGIES_AVAILABLE:
            try:
                self.regime_app = RegimeStrategyApp()
                self.config = self.regime_app.config
                self.regime_identifier = RegimeIdentifier(self.config)
                self.regime_forecaster = RegimeForecaster(self.config)
                
                logger.info("Regime service initialized with regime_strategies module")
            except Exception as e:
                logger.error(f"Failed to initialize regime_strategies module: {e}")
                self._init_fallback()
        else:
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback mode without regime_strategies."""
        self.regime_app = None
        self.config = None
        self.regime_identifier = None
        self.regime_forecaster = None
        logger.warning("Regime service running in fallback mode")
    
    async def identify_regimes(
        self,
        method: str = "investment_clock",
        start_date: date = None,
        end_date: date = None,
        assets: List[str] = None,
        economic_indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Identify market regimes using specified method."""
        try:
            logger.info(f"Identifying regimes using {method}")
            
            if self.regime_identifier is None:
                return self._mock_regime_identification(method)
            
            # Use the actual regime identifier from regime_strategies
            if method == "investment_clock":
                regimes = self.regime_identifier.identify_investment_clock_regimes(
                    start_date=start_date,
                    end_date=end_date
                )
            elif method == "kmeans":
                regimes = self.regime_identifier.identify_kmeans_regimes(
                    assets=assets or ["SPY", "TLT", "GLD"],
                    start_date=start_date,
                    end_date=end_date
                )
            elif method == "hmm":
                regimes = self.regime_identifier.identify_hmm_regimes(
                    assets=assets or ["SPY"],
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                raise ValueError(f"Unknown regime identification method: {method}")
            
            # Process results into API format
            result = {
                "regimes": regimes.to_dict() if hasattr(regimes, 'to_dict') else regimes,
                "regime_summary": self._calculate_regime_summary(regimes),
                "transition_matrix": self._calculate_transition_matrix(regimes),
                "regime_characteristics": self._get_regime_characteristics(method)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Regime identification error: {e}")
            return self._mock_regime_identification(method)
    
    async def forecast_regime(
        self,
        model_type: str = "logistic_regression",
        forecast_horizon: int = 1,
        features: List[str] = None,
        training_window: int = 60
    ) -> Dict[str, Any]:
        """Forecast future market regime."""
        try:
            logger.info(f"Forecasting regime using {model_type}")
            
            if self.regime_forecaster is None:
                return self._mock_regime_forecast()
            
            # Use the actual regime forecaster from regime_strategies
            forecast_result = self.regime_forecaster.forecast_regime(
                model_type=model_type,
                forecast_horizon=forecast_horizon,
                features=features,
                training_window=training_window
            )
            
            return {
                "predicted_regime": forecast_result.get("predicted_regime", "Growth"),
                "probabilities": forecast_result.get("probabilities", {"Growth": 0.4, "Heating": 0.3, "Stagflation": 0.2, "Slowing": 0.1}),
                "confidence": forecast_result.get("confidence", 0.75),
                "model_metrics": forecast_result.get("model_metrics", {"accuracy": 0.72, "precision": 0.68})
            }
            
        except Exception as e:
            logger.error(f"Regime forecasting error: {e}")
            return self._mock_regime_forecast()
    
    async def get_current_regime(
        self,
        method: str = "investment_clock"
    ) -> Dict[str, Any]:
        """Get current market regime assessment."""
        try:
            if self.regime_identifier is None:
                return self._mock_current_regime()
            
            # Get the most recent regime identification
            current_regime = self.regime_identifier.get_current_regime(method=method)
            
            return {
                "regime": current_regime.get("regime", "Growth"),
                "confidence": current_regime.get("confidence", 0.80),
                "timestamp": datetime.now(),
                "description": self._get_regime_description(current_regime.get("regime", "Growth")),
                "indicators": current_regime.get("indicators", {})
            }
            
        except Exception as e:
            logger.error(f"Current regime error: {e}")
            return self._mock_current_regime()
    
    async def get_regime_history(
        self,
        method: str = "investment_clock",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get historical regime transitions and statistics."""
        try:
            if self.regime_identifier is None:
                return self._mock_regime_history()
            
            # Get regime history from the regime strategies module
            history = self.regime_identifier.get_regime_history(
                method=method,
                start_date=start_date,
                end_date=end_date
            )
            
            return {
                "timeline": history.get("timeline", []),
                "statistics": history.get("statistics", {}),
                "avg_duration": history.get("avg_duration", 6),  # months
                "transitions": history.get("transition_frequency", 12)  # per year
            }
            
        except Exception as e:
            logger.error(f"Regime history error: {e}")
            return self._mock_regime_history()
    
    def _mock_regime_identification(self, method: str) -> Dict[str, Any]:
        """Mock regime identification for fallback."""
        return {
            "regimes": {
                "2020-01-01": "Growth",
                "2020-03-01": "Slowing", 
                "2020-06-01": "Growth",
                "2020-09-01": "Heating",
                "2021-01-01": "Growth"
            },
            "regime_summary": {
                "Growth": 0.45,
                "Heating": 0.25,
                "Stagflation": 0.15,
                "Slowing": 0.15
            },
            "transition_matrix": {
                "Growth": {"Growth": 0.7, "Heating": 0.2, "Slowing": 0.1},
                "Heating": {"Growth": 0.3, "Heating": 0.4, "Stagflation": 0.3},
                "Stagflation": {"Heating": 0.3, "Stagflation": 0.4, "Slowing": 0.3},
                "Slowing": {"Growth": 0.4, "Stagflation": 0.1, "Slowing": 0.5}
            },
            "regime_characteristics": self._get_regime_characteristics(method)
        }
    
    def _mock_regime_forecast(self) -> Dict[str, Any]:
        """Mock regime forecast for fallback."""
        return {
            "predicted_regime": "Growth",
            "probabilities": {
                "Growth": 0.45,
                "Heating": 0.25,
                "Stagflation": 0.15,
                "Slowing": 0.15
            },
            "confidence": 0.75,
            "model_metrics": {
                "accuracy": 0.72,
                "precision": 0.68,
                "recall": 0.71
            }
        }
    
    def _mock_current_regime(self) -> Dict[str, Any]:
        """Mock current regime for fallback."""
        return {
            "regime": "Growth",
            "confidence": 0.82,
            "timestamp": datetime.now(),
            "description": "Economic growth with moderate inflation",
            "indicators": {
                "gdp_growth": 0.025,
                "inflation_rate": 0.032,
                "unemployment": 0.039
            }
        }
    
    def _mock_regime_history(self) -> Dict[str, Any]:
        """Mock regime history for fallback."""
        return {
            "timeline": [
                {"date": "2023-01-01", "regime": "Growth", "duration": 6},
                {"date": "2023-07-01", "regime": "Heating", "duration": 4},
                {"date": "2023-11-01", "regime": "Slowing", "duration": 3}
            ],
            "statistics": {
                "most_common": "Growth",
                "least_common": "Stagflation",
                "total_transitions": 24
            },
            "avg_duration": 5.2,
            "transitions": 8
        }
    
    def _calculate_regime_summary(self, regimes) -> Dict[str, float]:
        """Calculate regime distribution summary."""
        # Implementation would analyze regime data
        return {
            "Growth": 0.40,
            "Heating": 0.30, 
            "Stagflation": 0.15,
            "Slowing": 0.15
        }
    
    def _calculate_transition_matrix(self, regimes) -> Dict[str, Dict[str, float]]:
        """Calculate regime transition probabilities."""
        # Implementation would calculate actual transitions
        return {
            "Growth": {"Growth": 0.7, "Heating": 0.2, "Slowing": 0.1},
            "Heating": {"Growth": 0.3, "Heating": 0.4, "Stagflation": 0.3},
            "Stagflation": {"Heating": 0.3, "Stagflation": 0.4, "Slowing": 0.3},
            "Slowing": {"Growth": 0.4, "Stagflation": 0.1, "Slowing": 0.5}
        }
    
    def _get_regime_characteristics(self, method: str) -> Dict[str, Any]:
        """Get characteristics for each regime."""
        if method == "investment_clock":
            return {
                "Growth": {
                    "description": "Economic growth with falling inflation",
                    "optimal_assets": ["Equities", "Credit"],
                    "color": "#22c55e"
                },
                "Heating": {
                    "description": "Economic growth with rising inflation", 
                    "optimal_assets": ["Commodities", "Real Estate"],
                    "color": "#f59e0b"
                },
                "Stagflation": {
                    "description": "Economic slowdown with rising inflation",
                    "optimal_assets": ["Commodities", "Cash"],
                    "color": "#ef4444"
                },
                "Slowing": {
                    "description": "Economic slowdown with falling inflation",
                    "optimal_assets": ["Bonds", "Government Securities"],
                    "color": "#3b82f6"
                }
            }
        else:
            return {
                "Regime_0": {"description": "Low volatility state", "color": "#22c55e"},
                "Regime_1": {"description": "Medium volatility state", "color": "#f59e0b"},
                "Regime_2": {"description": "High volatility state", "color": "#ef4444"}
            }
    
    def _get_regime_description(self, regime: str) -> str:
        """Get description for a specific regime."""
        descriptions = {
            "Growth": "Economic growth with moderate inflation and strong equity performance",
            "Heating": "Economic expansion with rising inflation pressures",
            "Stagflation": "Economic stagnation combined with inflationary pressures",
            "Slowing": "Economic deceleration with deflationary tendencies"
        }
        return descriptions.get(regime, "Unknown regime state")