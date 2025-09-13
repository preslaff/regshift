"""
Backtesting and strategy evaluation endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field
import uuid
from loguru import logger

from app.services.backtesting_service import BacktestingService
from app.core.auth import get_current_user
from app.models.user import User

router = APIRouter()


class BacktestRequest(BaseModel):
    """Request model for comprehensive backtesting."""
    name: str = Field(..., description="Backtest name")
    assets: List[str] = Field(..., description="Assets to include")
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    initial_capital: float = Field(1000000.0, description="Initial capital")
    optimization_methods: List[str] = Field(["max_sharpe"], description="Optimization methods")
    regime_methods: List[str] = Field(["investment_clock"], description="Regime methods")
    rebalance_frequency: str = Field("monthly", description="Rebalancing frequency")
    benchmarks: List[str] = Field(["SPY"], description="Benchmark assets")
    transaction_costs: float = Field(0.001, description="Transaction cost percentage")


class MultiHorizonRequest(BaseModel):
    """Request model for multi-horizon analysis."""
    assets: List[str] = Field(..., description="Assets to analyze")
    start_dates: List[date] = Field(..., description="Multiple start dates")
    end_dates: List[date] = Field(..., description="Multiple end dates")
    strategies: List[str] = Field(..., description="Strategies to compare")
    regime_methods: List[str] = Field(["investment_clock"], description="Regime methods")


class ComparisonRequest(BaseModel):
    """Request model for strategy comparison."""
    backtest_ids: List[str] = Field(..., description="Backtest IDs to compare")
    metrics: List[str] = Field(["total_return", "sharpe_ratio", "max_drawdown"], description="Metrics to compare")


@router.post("/run", response_model=Dict[str, Any])
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Run comprehensive backtesting analysis."""
    try:
        logger.info(f"Starting backtest '{request.name}' for user {current_user.id}")
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        backtesting_service = BacktestingService()
        
        # Start backtest in background
        background_tasks.add_task(
            backtesting_service.run_comprehensive_backtest,
            task_id=task_id,
            user_id=current_user.id,
            name=request.name,
            assets=request.assets,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            optimization_methods=request.optimization_methods,
            regime_methods=request.regime_methods,
            rebalance_frequency=request.rebalance_frequency,
            benchmarks=request.benchmarks,
            transaction_costs=request.transaction_costs
        )
        
        return {
            "task_id": task_id,
            "name": request.name,
            "status": "started",
            "message": "Backtest analysis started",
            "estimated_duration": "5-15 minutes",
            "results_url": f"/backtesting/{task_id}/results"
        }
        
    except Exception as e:
        logger.error(f"Error starting backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{task_id}/results", response_model=Dict[str, Any])
async def get_backtest_results(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get backtest results by task ID."""
    try:
        backtesting_service = BacktestingService()
        
        results = await backtesting_service.get_backtest_results(
            task_id=task_id,
            user_id=current_user.id
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="Backtest results not found")
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{task_id}/status", response_model=Dict[str, Any])
async def get_backtest_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get backtest execution status."""
    try:
        backtesting_service = BacktestingService()
        
        status = await backtesting_service.get_backtest_status(
            task_id=task_id,
            user_id=current_user.id
        )
        
        return {
            "task_id": task_id,
            "status": status["status"],
            "progress": status["progress"],
            "message": status["message"],
            "started_at": status["started_at"],
            "estimated_completion": status.get("estimated_completion"),
            "error": status.get("error")
        }
        
    except Exception as e:
        logger.error(f"Error getting backtest status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-horizon", response_model=Dict[str, Any])
async def run_multi_horizon_analysis(
    request: MultiHorizonRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Run multi-horizon backtest analysis."""
    try:
        logger.info(f"Starting multi-horizon analysis for user {current_user.id}")
        
        task_id = str(uuid.uuid4())
        backtesting_service = BacktestingService()
        
        # Start multi-horizon analysis in background
        background_tasks.add_task(
            backtesting_service.run_multi_horizon_analysis,
            task_id=task_id,
            user_id=current_user.id,
            assets=request.assets,
            start_dates=request.start_dates,
            end_dates=request.end_dates,
            strategies=request.strategies,
            regime_methods=request.regime_methods
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "analysis_combinations": len(request.start_dates) * len(request.end_dates),
            "estimated_duration": "15-30 minutes",
            "results_url": f"/backtesting/{task_id}/multi-horizon-results"
        }
        
    except Exception as e:
        logger.error(f"Error starting multi-horizon analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=Dict[str, Any])
async def compare_strategies(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_user)
):
    """Compare multiple backtest strategies."""
    try:
        backtesting_service = BacktestingService()
        
        comparison = await backtesting_service.compare_strategies(
            backtest_ids=request.backtest_ids,
            metrics=request.metrics,
            user_id=current_user.id
        )
        
        return {
            "comparison_results": comparison["results"],
            "performance_ranking": comparison["ranking"],
            "statistical_tests": comparison["stats_tests"],
            "visualization_data": comparison["viz_data"]
        }
        
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[Dict[str, Any]])
async def get_backtest_history(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """Get user's backtest history."""
    try:
        backtesting_service = BacktestingService()
        
        history = await backtesting_service.get_user_backtests(
            user_id=current_user.id,
            limit=limit,
            offset=offset
        )
        
        return [
            {
                "task_id": bt.task_id,
                "name": bt.name,
                "status": bt.status,
                "created_at": bt.created_at,
                "completed_at": bt.completed_at,
                "assets": bt.assets,
                "performance_summary": bt.performance_summary
            }
            for bt in history
        ]
        
    except Exception as e:
        logger.error(f"Error getting backtest history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{task_id}")
async def delete_backtest(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a backtest and its results."""
    try:
        backtesting_service = BacktestingService()
        
        success = await backtesting_service.delete_backtest(
            task_id=task_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        return {"message": "Backtest deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))