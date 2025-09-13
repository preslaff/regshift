"""
Market regime identification and analysis endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field
from loguru import logger

from app.services.regime_service import RegimeService
from app.core.auth import get_current_user
from app.models.user import User

router = APIRouter()


class RegimeAnalysisRequest(BaseModel):
    """Request model for regime analysis."""
    method: str = Field("investment_clock", description="Regime identification method")
    start_date: date = Field(..., description="Analysis start date")
    end_date: date = Field(..., description="Analysis end date")
    assets: List[str] = Field(..., description="Assets to analyze")
    economic_indicators: Optional[List[str]] = Field(None, description="Economic indicators to use")


class RegimeForecastRequest(BaseModel):
    """Request model for regime forecasting."""
    model_type: str = Field("logistic_regression", description="Forecasting model type")
    forecast_horizon: int = Field(1, description="Forecast horizon in months")
    features: List[str] = Field(..., description="Features for prediction")
    training_window: int = Field(60, description="Training window in months")


@router.post("/identify", response_model=Dict[str, Any])
async def identify_regimes(
    request: RegimeAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Identify market regimes using specified method."""
    try:
        logger.info(f"Identifying regimes using {request.method} for user {current_user.id}")
        
        regime_service = RegimeService()
        
        # Perform regime identification
        analysis = await regime_service.identify_regimes(
            method=request.method,
            start_date=request.start_date,
            end_date=request.end_date,
            assets=request.assets,
            economic_indicators=request.economic_indicators
        )
        
        return {
            "method": request.method,
            "regimes": analysis["regimes"],
            "regime_summary": analysis["regime_summary"],
            "transition_matrix": analysis["transition_matrix"],
            "regime_characteristics": analysis["regime_characteristics"],
            "analysis_period": {
                "start_date": request.start_date,
                "end_date": request.end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error identifying regimes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast", response_model=Dict[str, Any])
async def forecast_regime(
    request: RegimeForecastRequest,
    current_user: User = Depends(get_current_user)
):
    """Forecast future market regime."""
    try:
        logger.info(f"Forecasting regime using {request.model_type} for user {current_user.id}")
        
        regime_service = RegimeService()
        
        # Perform regime forecasting
        forecast = await regime_service.forecast_regime(
            model_type=request.model_type,
            forecast_horizon=request.forecast_horizon,
            features=request.features,
            training_window=request.training_window
        )
        
        return {
            "forecast_regime": forecast["predicted_regime"],
            "probability_distribution": forecast["probabilities"],
            "confidence": forecast["confidence"],
            "model_performance": forecast["model_metrics"],
            "forecast_horizon": request.forecast_horizon,
            "model_type": request.model_type
        }
        
    except Exception as e:
        logger.error(f"Error forecasting regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current", response_model=Dict[str, Any])
async def get_current_regime(
    method: str = Query("investment_clock", description="Regime identification method"),
    current_user: User = Depends(get_current_user)
):
    """Get current market regime assessment."""
    try:
        regime_service = RegimeService()
        
        # Get current regime
        current_regime = await regime_service.get_current_regime(method=method)
        
        return {
            "current_regime": current_regime["regime"],
            "confidence": current_regime["confidence"],
            "method": method,
            "last_updated": current_regime["timestamp"],
            "regime_description": current_regime["description"],
            "key_indicators": current_regime["indicators"]
        }
        
    except Exception as e:
        logger.error(f"Error getting current regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=Dict[str, Any])
async def get_regime_history(
    method: str = Query("investment_clock", description="Regime identification method"),
    start_date: Optional[date] = Query(None, description="Start date for history"),
    end_date: Optional[date] = Query(None, description="End date for history"),
    current_user: User = Depends(get_current_user)
):
    """Get historical regime transitions and statistics."""
    try:
        regime_service = RegimeService()
        
        # Get regime history
        history = await regime_service.get_regime_history(
            method=method,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "regime_timeline": history["timeline"],
            "regime_statistics": history["statistics"],
            "average_duration": history["avg_duration"],
            "transition_frequency": history["transitions"],
            "method": method,
            "period": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting regime history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods", response_model=List[Dict[str, Any]])
async def get_available_methods():
    """Get available regime identification methods."""
    return [
        {
            "method": "investment_clock",
            "name": "Investment Clock",
            "description": "Four-regime model based on inflation and economic growth",
            "regimes": ["Growth", "Heating", "Stagflation", "Slowing"],
            "indicators": ["CPI", "CLI"]
        },
        {
            "method": "kmeans",
            "name": "K-Means Clustering", 
            "description": "Unsupervised clustering of market conditions",
            "regimes": ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"],
            "indicators": ["Returns", "Volatility", "Economic indicators"]
        },
        {
            "method": "hmm",
            "name": "Hidden Markov Model",
            "description": "Statistical model with hidden regime states",
            "regimes": ["State 1", "State 2", "State 3"],
            "indicators": ["Market returns", "Volatility"]
        }
    ]