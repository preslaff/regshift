"""
Scenario analysis and stress testing endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field
from loguru import logger

from app.services.scenario_service import ScenarioService
from app.core.auth import get_current_user
from app.models.user import User

router = APIRouter()


class ScenarioRequest(BaseModel):
    """Request model for scenario analysis."""
    name: str = Field(..., description="Scenario name")
    scenario_type: str = Field(..., description="Type of scenario")
    portfolio_id: Optional[int] = Field(None, description="Portfolio ID to analyze")
    assets: List[str] = Field(..., description="Assets to include")
    weights: List[float] = Field(..., description="Asset weights")
    scenario_parameters: Dict[str, Any] = Field(..., description="Scenario-specific parameters")


class StressTestRequest(BaseModel):
    """Request model for stress testing."""
    portfolio_id: Optional[int] = Field(None, description="Portfolio ID")
    assets: List[str] = Field(..., description="Assets to test")
    weights: List[float] = Field(..., description="Asset weights")
    stress_scenarios: List[str] = Field(..., description="Stress test scenarios")
    confidence_levels: List[float] = Field([0.95, 0.99], description="VaR confidence levels")


class MonteCarloRequest(BaseModel):
    """Request model for Monte Carlo simulation."""
    portfolio_id: Optional[int] = Field(None, description="Portfolio ID")
    assets: List[str] = Field(..., description="Assets to simulate")
    weights: List[float] = Field(..., description="Asset weights")
    simulation_horizon: int = Field(252, description="Simulation horizon in days")
    num_simulations: int = Field(10000, description="Number of simulations")
    initial_value: float = Field(1000000.0, description="Initial portfolio value")


class HistoricalScenarioRequest(BaseModel):
    """Request model for historical scenario replay."""
    portfolio_id: Optional[int] = Field(None, description="Portfolio ID")
    assets: List[str] = Field(..., description="Assets to analyze")
    weights: List[float] = Field(..., description="Asset weights") 
    historical_periods: List[Dict[str, date]] = Field(..., description="Historical periods to replay")
    regime_aware: bool = Field(True, description="Use regime-aware analysis")


@router.post("/create", response_model=Dict[str, Any])
async def create_scenario(
    request: ScenarioRequest,
    current_user: User = Depends(get_current_user)
):
    """Create and analyze a custom scenario."""
    try:
        logger.info(f"Creating scenario '{request.name}' for user {current_user.id}")
        
        scenario_service = ScenarioService()
        
        # Create and analyze scenario
        scenario = await scenario_service.create_scenario(
            user_id=current_user.id,
            name=request.name,
            scenario_type=request.scenario_type,
            portfolio_id=request.portfolio_id,
            assets=request.assets,
            weights=request.weights,
            parameters=request.scenario_parameters
        )
        
        return {
            "scenario_id": scenario.id,
            "name": scenario.name,
            "scenario_type": scenario.scenario_type,
            "results": scenario.analysis_results,
            "created_at": scenario.created_at,
            "message": "Scenario created and analyzed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stress-test", response_model=Dict[str, Any])
async def run_stress_test(
    request: StressTestRequest,
    current_user: User = Depends(get_current_user)
):
    """Run comprehensive stress testing."""
    try:
        logger.info(f"Running stress test for user {current_user.id}")
        
        scenario_service = ScenarioService()
        
        # Perform stress testing
        results = await scenario_service.run_stress_test(
            user_id=current_user.id,
            portfolio_id=request.portfolio_id,
            assets=request.assets,
            weights=request.weights,
            stress_scenarios=request.stress_scenarios,
            confidence_levels=request.confidence_levels
        )
        
        return {
            "stress_test_results": results["scenarios"],
            "value_at_risk": results["var_estimates"],
            "expected_shortfall": results["expected_shortfall"],
            "worst_case_scenarios": results["worst_cases"],
            "portfolio_resilience": results["resilience_metrics"]
        }
        
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monte-carlo", response_model=Dict[str, Any])
async def run_monte_carlo(
    request: MonteCarloRequest,
    current_user: User = Depends(get_current_user)
):
    """Run Monte Carlo simulation."""
    try:
        logger.info(f"Running Monte Carlo simulation for user {current_user.id}")
        
        scenario_service = ScenarioService()
        
        # Perform Monte Carlo simulation
        results = await scenario_service.run_monte_carlo(
            user_id=current_user.id,
            portfolio_id=request.portfolio_id,
            assets=request.assets,
            weights=request.weights,
            horizon=request.simulation_horizon,
            num_simulations=request.num_simulations,
            initial_value=request.initial_value
        )
        
        return {
            "simulation_summary": results["summary"],
            "percentile_outcomes": results["percentiles"],
            "probability_distributions": results["distributions"],
            "risk_metrics": results["risk_metrics"],
            "simulation_paths": results["sample_paths"],
            "confidence_intervals": results["confidence_intervals"]
        }
        
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/historical-replay", response_model=Dict[str, Any])
async def replay_historical_scenarios(
    request: HistoricalScenarioRequest,
    current_user: User = Depends(get_current_user)
):
    """Replay historical market scenarios."""
    try:
        logger.info(f"Replaying historical scenarios for user {current_user.id}")
        
        scenario_service = ScenarioService()
        
        # Perform historical scenario replay
        results = await scenario_service.replay_historical_scenarios(
            user_id=current_user.id,
            portfolio_id=request.portfolio_id,
            assets=request.assets,
            weights=request.weights,
            historical_periods=request.historical_periods,
            regime_aware=request.regime_aware
        )
        
        return {
            "scenario_results": results["period_results"],
            "performance_summary": results["summary"],
            "regime_analysis": results["regime_breakdown"] if request.regime_aware else None,
            "comparative_metrics": results["comparative_metrics"],
            "lessons_learned": results["insights"]
        }
        
    except Exception as e:
        logger.error(f"Error replaying historical scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates", response_model=List[Dict[str, Any]])
async def get_scenario_templates():
    """Get available scenario templates."""
    return [
        {
            "name": "2008 Financial Crisis",
            "type": "historical_crisis",
            "description": "Replay the 2008-2009 financial crisis conditions",
            "parameters": {
                "start_date": "2007-10-01",
                "end_date": "2009-03-31",
                "key_events": ["Lehman Brothers collapse", "Bank bailouts", "Market crash"]
            }
        },
        {
            "name": "COVID-19 Market Shock",
            "type": "pandemic_shock",
            "description": "March 2020 pandemic-induced market volatility",
            "parameters": {
                "start_date": "2020-02-01",
                "end_date": "2020-05-31",
                "volatility_multiplier": 3.0
            }
        },
        {
            "name": "Interest Rate Shock",
            "type": "rate_shock",
            "description": "Sudden interest rate change scenario",
            "parameters": {
                "rate_change": 2.0,
                "duration": "6_months",
                "affected_sectors": ["financials", "utilities", "real_estate"]
            }
        },
        {
            "name": "Inflation Surge",
            "type": "inflation_shock",
            "description": "High inflation environment scenario",
            "parameters": {
                "inflation_rate": 0.08,
                "duration": "12_months",
                "affected_assets": ["bonds", "growth_stocks"]
            }
        }
    ]


@router.get("/list", response_model=List[Dict[str, Any]])
async def list_scenarios(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """List user's scenarios."""
    try:
        scenario_service = ScenarioService()
        
        scenarios = await scenario_service.get_user_scenarios(
            user_id=current_user.id,
            limit=limit,
            offset=offset
        )
        
        return [
            {
                "id": s.id,
                "name": s.name,
                "scenario_type": s.scenario_type,
                "created_at": s.created_at,
                "results_summary": s.results_summary
            }
            for s in scenarios
        ]
        
    except Exception as e:
        logger.error(f"Error listing scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{scenario_id}", response_model=Dict[str, Any])
async def get_scenario(
    scenario_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get specific scenario details."""
    try:
        scenario_service = ScenarioService()
        
        scenario = await scenario_service.get_scenario(
            scenario_id=scenario_id,
            user_id=current_user.id
        )
        
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        return {
            "id": scenario.id,
            "name": scenario.name,
            "scenario_type": scenario.scenario_type,
            "parameters": scenario.parameters,
            "results": scenario.analysis_results,
            "created_at": scenario.created_at,
            "assets": scenario.assets,
            "weights": scenario.weights
        }
        
    except Exception as e:
        logger.error(f"Error getting scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{scenario_id}")
async def delete_scenario(
    scenario_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete a scenario."""
    try:
        scenario_service = ScenarioService()
        
        success = await scenario_service.delete_scenario(
            scenario_id=scenario_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        return {"message": "Scenario deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))