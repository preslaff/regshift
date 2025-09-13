"""
Analytics and reporting endpoints for performance analysis and insights.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
from pydantic import BaseModel, Field
from loguru import logger

from app.services.analytics_service import AnalyticsService
from app.core.auth import get_current_user
from app.models.user import User

router = APIRouter()


class PerformanceAnalysisRequest(BaseModel):
    """Request model for performance analysis."""
    portfolio_ids: List[int] = Field(..., description="Portfolio IDs to analyze")
    benchmark_symbols: List[str] = Field(["SPY"], description="Benchmark symbols")
    analysis_period: int = Field(252, description="Analysis period in days")
    metrics: List[str] = Field(
        ["total_return", "sharpe_ratio", "max_drawdown", "volatility"],
        description="Performance metrics to calculate"
    )


class RiskAnalysisRequest(BaseModel):
    """Request model for risk analysis."""
    portfolio_id: int = Field(..., description="Portfolio ID")
    risk_free_rate: float = Field(0.02, description="Risk-free rate")
    confidence_levels: List[float] = Field([0.95, 0.99], description="VaR confidence levels")
    time_horizon: int = Field(21, description="Risk horizon in days")


class AttributionRequest(BaseModel):
    """Request model for performance attribution."""
    portfolio_id: int = Field(..., description="Portfolio ID")
    benchmark_symbol: str = Field("SPY", description="Benchmark symbol")
    attribution_method: str = Field("brinson", description="Attribution method")
    start_date: date = Field(..., description="Analysis start date")
    end_date: date = Field(..., description="Analysis end date")


@router.post("/performance", response_model=Dict[str, Any])
async def analyze_performance(
    request: PerformanceAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Comprehensive performance analysis."""
    try:
        logger.info(f"Running performance analysis for user {current_user.id}")
        
        analytics_service = AnalyticsService()
        
        # Perform performance analysis
        analysis = await analytics_service.analyze_performance(
            user_id=current_user.id,
            portfolio_ids=request.portfolio_ids,
            benchmark_symbols=request.benchmark_symbols,
            analysis_period=request.analysis_period,
            metrics=request.metrics
        )
        
        return {
            "performance_metrics": analysis["metrics"],
            "comparative_analysis": analysis["comparison"],
            "risk_adjusted_returns": analysis["risk_adjusted"],
            "rolling_performance": analysis["rolling_metrics"],
            "benchmark_comparison": analysis["benchmark_analysis"],
            "performance_attribution": analysis["attribution"]
        }
        
    except Exception as e:
        logger.error(f"Error in performance analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk-analysis", response_model=Dict[str, Any])
async def analyze_risk(
    request: RiskAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Comprehensive risk analysis."""
    try:
        logger.info(f"Running risk analysis for portfolio {request.portfolio_id}")
        
        analytics_service = AnalyticsService()
        
        # Perform risk analysis
        analysis = await analytics_service.analyze_risk(
            user_id=current_user.id,
            portfolio_id=request.portfolio_id,
            risk_free_rate=request.risk_free_rate,
            confidence_levels=request.confidence_levels,
            time_horizon=request.time_horizon
        )
        
        return {
            "value_at_risk": analysis["var"],
            "expected_shortfall": analysis["expected_shortfall"],
            "risk_decomposition": analysis["risk_breakdown"],
            "correlation_analysis": analysis["correlations"],
            "stress_test_results": analysis["stress_tests"],
            "risk_contribution": analysis["component_risk"]
        }
        
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/attribution", response_model=Dict[str, Any])
async def performance_attribution(
    request: AttributionRequest,
    current_user: User = Depends(get_current_user)
):
    """Performance attribution analysis."""
    try:
        logger.info(f"Running attribution analysis for portfolio {request.portfolio_id}")
        
        analytics_service = AnalyticsService()
        
        # Perform attribution analysis
        attribution = await analytics_service.performance_attribution(
            user_id=current_user.id,
            portfolio_id=request.portfolio_id,
            benchmark_symbol=request.benchmark_symbol,
            method=request.attribution_method,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        return {
            "total_attribution": attribution["total"],
            "allocation_effect": attribution["allocation"],
            "selection_effect": attribution["selection"],
            "interaction_effect": attribution["interaction"],
            "sector_attribution": attribution["sector_breakdown"],
            "asset_attribution": attribution["asset_breakdown"]
        }
        
    except Exception as e:
        logger.error(f"Error in attribution analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard", response_model=Dict[str, Any])
async def get_analytics_dashboard(
    current_user: User = Depends(get_current_user),
    period: str = Query("1M", description="Dashboard period (1W, 1M, 3M, 1Y)")
):
    """Get analytics dashboard data."""
    try:
        analytics_service = AnalyticsService()
        
        # Get dashboard data
        dashboard = await analytics_service.get_dashboard_data(
            user_id=current_user.id,
            period=period
        )
        
        return {
            "portfolio_summary": dashboard["portfolios"],
            "performance_overview": dashboard["performance"],
            "risk_metrics": dashboard["risk"],
            "recent_activity": dashboard["activity"],
            "market_insights": dashboard["insights"],
            "alerts": dashboard["alerts"]
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/generate", response_model=Dict[str, Any])
async def generate_report(
    report_type: str = Query(..., description="Report type"),
    portfolio_ids: List[int] = Query(..., description="Portfolio IDs"),
    start_date: date = Query(..., description="Report start date"),
    end_date: date = Query(..., description="Report end date"),
    current_user: User = Depends(get_current_user)
):
    """Generate comprehensive portfolio report."""
    try:
        logger.info(f"Generating {report_type} report for user {current_user.id}")
        
        analytics_service = AnalyticsService()
        
        # Generate report
        report = await analytics_service.generate_report(
            user_id=current_user.id,
            report_type=report_type,
            portfolio_ids=portfolio_ids,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "report_id": report["report_id"],
            "report_type": report_type,
            "status": "generated",
            "download_url": report["download_url"],
            "preview_data": report["preview"],
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights", response_model=List[Dict[str, Any]])
async def get_market_insights(
    current_user: User = Depends(get_current_user),
    insight_type: Optional[str] = Query(None, description="Type of insights"),
    limit: int = Query(10, description="Number of insights")
):
    """Get AI-generated market insights and recommendations."""
    try:
        analytics_service = AnalyticsService()
        
        # Get insights
        insights = await analytics_service.get_market_insights(
            user_id=current_user.id,
            insight_type=insight_type,
            limit=limit
        )
        
        return [
            {
                "id": insight["id"],
                "type": insight["type"],
                "title": insight["title"],
                "content": insight["content"],
                "confidence": insight["confidence"],
                "relevance_score": insight["relevance"],
                "created_at": insight["timestamp"],
                "related_assets": insight["assets"],
                "actionable": insight["actionable"]
            }
            for insight in insights
        ]
        
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_portfolio_alerts(
    current_user: User = Depends(get_current_user),
    alert_type: Optional[str] = Query(None, description="Alert type filter"),
    severity: Optional[str] = Query(None, description="Severity filter")
):
    """Get portfolio alerts and notifications."""
    try:
        analytics_service = AnalyticsService()
        
        # Get alerts
        alerts = await analytics_service.get_portfolio_alerts(
            user_id=current_user.id,
            alert_type=alert_type,
            severity=severity
        )
        
        return [
            {
                "id": alert["id"],
                "type": alert["type"],
                "severity": alert["severity"],
                "title": alert["title"],
                "message": alert["message"],
                "portfolio_id": alert["portfolio_id"],
                "triggered_at": alert["timestamp"],
                "acknowledged": alert["acknowledged"],
                "recommended_action": alert["recommendation"]
            }
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    current_user: User = Depends(get_current_user)
):
    """Acknowledge a portfolio alert."""
    try:
        analytics_service = AnalyticsService()
        
        # Acknowledge alert
        success = await analytics_service.acknowledge_alert(
            alert_id=alert_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"message": "Alert acknowledged successfully"}
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/summary", response_model=Dict[str, Any])
async def get_metrics_summary(
    current_user: User = Depends(get_current_user),
    portfolio_ids: Optional[List[int]] = Query(None, description="Portfolio IDs"),
    time_period: str = Query("1Y", description="Time period")
):
    """Get summarized performance metrics."""
    try:
        analytics_service = AnalyticsService()
        
        # Get metrics summary
        summary = await analytics_service.get_metrics_summary(
            user_id=current_user.id,
            portfolio_ids=portfolio_ids,
            time_period=time_period
        )
        
        return {
            "total_portfolios": summary["portfolio_count"],
            "total_aum": summary["total_assets"],
            "average_return": summary["avg_return"],
            "best_performer": summary["best_portfolio"],
            "worst_performer": summary["worst_portfolio"],
            "risk_metrics": summary["risk_summary"],
            "regime_performance": summary["regime_breakdown"]
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))