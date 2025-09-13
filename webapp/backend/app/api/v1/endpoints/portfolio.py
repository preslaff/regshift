"""
Portfolio analysis and optimization endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field
import asyncio
from loguru import logger

from app.services.portfolio_service import PortfolioService
from app.models.portfolio import Portfolio, PortfolioAnalysis
from app.core.auth import get_current_user
from app.models.user import User

router = APIRouter()


class PortfolioRequest(BaseModel):
    """Request model for portfolio analysis."""
    name: str = Field(..., description="Portfolio name")
    assets: List[str] = Field(..., description="List of asset symbols")
    weights: Optional[List[float]] = Field(None, description="Asset weights (if None, will optimize)")
    benchmark: Optional[str] = Field("SPY", description="Benchmark symbol")
    start_date: date = Field(..., description="Analysis start date")
    end_date: date = Field(..., description="Analysis end date")
    optimization_method: str = Field("max_sharpe", description="Optimization method")
    regime_method: str = Field("investment_clock", description="Regime identification method")
    rebalance_frequency: str = Field("monthly", description="Rebalancing frequency")


class OptimizationRequest(BaseModel):
    """Request model for portfolio optimization."""
    assets: List[str] = Field(..., description="List of asset symbols")
    method: str = Field("max_sharpe", description="Optimization method")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optimization constraints")
    start_date: date = Field(..., description="Data start date")
    end_date: date = Field(..., description="Data end date")


class BacktestRequest(BaseModel):
    """Request model for backtesting."""
    portfolio_id: Optional[int] = Field(None, description="Existing portfolio ID")
    assets: List[str] = Field(..., description="List of asset symbols")
    start_date: date = Field(..., description="Backtest start date")  
    end_date: date = Field(..., description="Backtest end date")
    initial_capital: float = Field(1000000.0, description="Initial capital")
    optimization_method: str = Field("max_sharpe", description="Optimization method")
    regime_method: str = Field("investment_clock", description="Regime identification method")
    rebalance_frequency: str = Field("monthly", description="Rebalancing frequency")


@router.post("/create", response_model=Dict[str, Any])
async def create_portfolio(
    request: PortfolioRequest,
    current_user: User = Depends(get_current_user)
):
    """Create and analyze a new portfolio."""
    try:
        logger.info(f"Creating portfolio '{request.name}' for user {current_user.id}")
        
        portfolio_service = PortfolioService()
        
        # Create and analyze portfolio
        portfolio = await portfolio_service.create_portfolio(
            user_id=current_user.id,
            name=request.name,
            assets=request.assets,
            weights=request.weights,
            benchmark=request.benchmark,
            start_date=request.start_date,
            end_date=request.end_date,
            optimization_method=request.optimization_method,
            regime_method=request.regime_method
        )
        
        return {
            "portfolio_id": portfolio.id,
            "name": portfolio.name,
            "assets": portfolio.assets,
            "weights": portfolio.weights,
            "created_at": portfolio.created_at,
            "message": "Portfolio created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_portfolio(
    request: OptimizationRequest,
    current_user: User = Depends(get_current_user)
):
    """Optimize portfolio weights using specified method."""
    try:
        logger.info(f"Optimizing portfolio for user {current_user.id}")
        
        portfolio_service = PortfolioService()
        
        # Perform optimization
        result = await portfolio_service.optimize_weights(
            assets=request.assets,
            method=request.method,
            constraints=request.constraints,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        return {
            "assets": request.assets,
            "weights": result["weights"],
            "expected_return": result["expected_return"],
            "expected_volatility": result["expected_volatility"],
            "sharpe_ratio": result["sharpe_ratio"],
            "optimization_method": request.method,
            "optimization_status": result["status"]
        }
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest", response_model=Dict[str, Any])
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Run comprehensive portfolio backtest."""
    try:
        logger.info(f"Starting backtest for user {current_user.id}")
        
        portfolio_service = PortfolioService()
        
        # Start backtest (can be run in background for long analyses)
        backtest_task = await portfolio_service.run_backtest(
            user_id=current_user.id,
            assets=request.assets,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            optimization_method=request.optimization_method,
            regime_method=request.regime_method,
            rebalance_frequency=request.rebalance_frequency
        )
        
        return {
            "task_id": backtest_task.id,
            "status": "started",
            "message": "Backtest started successfully",
            "estimated_duration": "2-5 minutes",
            "results_url": f"/portfolio/backtest/{backtest_task.id}/results"
        }
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/{task_id}/results", response_model=Dict[str, Any])
async def get_backtest_results(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get backtest results by task ID."""
    try:
        portfolio_service = PortfolioService()
        
        # Get backtest results
        results = await portfolio_service.get_backtest_results(
            task_id=task_id,
            user_id=current_user.id
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="Backtest results not found")
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[Dict[str, Any]])
async def list_portfolios(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """List user's portfolios."""
    try:
        portfolio_service = PortfolioService()
        
        portfolios = await portfolio_service.get_user_portfolios(
            user_id=current_user.id,
            limit=limit,
            offset=offset
        )
        
        return [
            {
                "id": p.id,
                "name": p.name,
                "assets": p.assets,
                "created_at": p.created_at,
                "last_updated": p.updated_at,
                "performance_summary": p.performance_summary
            }
            for p in portfolios
        ]
        
    except Exception as e:
        logger.error(f"Error listing portfolios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{portfolio_id}", response_model=Dict[str, Any])
async def get_portfolio(
    portfolio_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get specific portfolio details."""
    try:
        portfolio_service = PortfolioService()
        
        portfolio = await portfolio_service.get_portfolio(
            portfolio_id=portfolio_id,
            user_id=current_user.id
        )
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return {
            "id": portfolio.id,
            "name": portfolio.name,
            "assets": portfolio.assets,
            "weights": portfolio.weights,
            "benchmark": portfolio.benchmark,
            "created_at": portfolio.created_at,
            "analysis": portfolio.latest_analysis,
            "performance_metrics": portfolio.performance_metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{portfolio_id}")
async def delete_portfolio(
    portfolio_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete a portfolio."""
    try:
        portfolio_service = PortfolioService()
        
        success = await portfolio_service.delete_portfolio(
            portfolio_id=portfolio_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return {"message": "Portfolio deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))