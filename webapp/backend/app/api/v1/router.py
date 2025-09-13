"""
Main API router that includes all endpoint routers.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import (
    auth,
    portfolio,
    regimes,
    backtesting,
    scenarios,
    market_data,
    users,
    analytics
)

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"]
)

api_router.include_router(
    portfolio.router,
    prefix="/portfolio",
    tags=["Portfolio"]
)

api_router.include_router(
    regimes.router,
    prefix="/regimes",
    tags=["Market Regimes"]
)

api_router.include_router(
    backtesting.router,
    prefix="/backtesting",
    tags=["Backtesting"]
)

api_router.include_router(
    scenarios.router,
    prefix="/scenarios",
    tags=["Scenario Analysis"]
)

api_router.include_router(
    market_data.router,
    prefix="/market-data",
    tags=["Market Data"]
)

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["Analytics & Reports"]
)