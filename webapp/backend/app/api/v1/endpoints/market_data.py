"""
Market data retrieval and management endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field
from loguru import logger

from app.services.market_data_service import MarketDataService
from app.core.auth import get_current_user
from app.models.user import User

router = APIRouter()


class DataRequest(BaseModel):
    """Request model for market data."""
    symbols: List[str] = Field(..., description="Asset symbols")
    start_date: date = Field(..., description="Start date")
    end_date: date = Field(..., description="End date")
    data_type: str = Field("prices", description="Type of data to retrieve")
    frequency: str = Field("daily", description="Data frequency")


class BenchmarkRequest(BaseModel):
    """Request model for benchmark data."""
    benchmark_type: str = Field(..., description="Benchmark type")
    start_date: date = Field(..., description="Start date")
    end_date: date = Field(..., description="End date")


@router.post("/prices", response_model=Dict[str, Any])
async def get_price_data(
    request: DataRequest,
    current_user: User = Depends(get_current_user)
):
    """Get historical price data for assets."""
    try:
        logger.info(f"Fetching price data for {len(request.symbols)} symbols for user {current_user.id}")
        
        market_data_service = MarketDataService()
        
        # Fetch price data
        data = await market_data_service.get_price_data(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            frequency=request.frequency
        )
        
        return {
            "symbols": request.symbols,
            "price_data": data["prices"],
            "metadata": data["metadata"],
            "data_quality": data["quality_metrics"],
            "period": {
                "start_date": request.start_date,
                "end_date": request.end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching price data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/returns", response_model=Dict[str, Any])
async def get_return_data(
    request: DataRequest,
    current_user: User = Depends(get_current_user)
):
    """Calculate and return asset return data."""
    try:
        logger.info(f"Calculating returns for {len(request.symbols)} symbols")
        
        market_data_service = MarketDataService()
        
        # Calculate returns
        returns = await market_data_service.calculate_returns(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            frequency=request.frequency
        )
        
        return {
            "symbols": request.symbols,
            "returns": returns["returns"],
            "statistics": returns["statistics"],
            "correlations": returns["correlations"],
            "period": {
                "start_date": request.start_date,
                "end_date": request.end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/economic-indicators", response_model=Dict[str, Any])
async def get_economic_indicators(
    indicators: List[str] = Query(..., description="Economic indicators to fetch"),
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date"),
    current_user: User = Depends(get_current_user)
):
    """Get economic indicator data from FRED."""
    try:
        logger.info(f"Fetching economic indicators: {indicators}")
        
        market_data_service = MarketDataService()
        
        # Fetch economic data
        data = await market_data_service.get_economic_indicators(
            indicators=indicators,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "indicators": indicators,
            "data": data["indicator_data"],
            "transformations": data["transformations"],
            "metadata": data["metadata"],
            "period": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching economic indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmarks", response_model=Dict[str, Any])
async def get_benchmark_data(
    request: BenchmarkRequest,
    current_user: User = Depends(get_current_user)
):
    """Get benchmark index data."""
    try:
        market_data_service = MarketDataService()
        
        # Fetch benchmark data
        data = await market_data_service.get_benchmark_data(
            benchmark_type=request.benchmark_type,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        return {
            "benchmark_type": request.benchmark_type,
            "data": data["benchmark_data"],
            "performance_metrics": data["metrics"],
            "period": {
                "start_date": request.start_date,
                "end_date": request.end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching benchmark data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols/search", response_model=List[Dict[str, Any]])
async def search_symbols(
    query: str = Query(..., description="Search query"),
    asset_type: Optional[str] = Query(None, description="Asset type filter"),
    limit: int = Query(50, description="Maximum results")
):
    """Search for asset symbols."""
    try:
        market_data_service = MarketDataService()
        
        # Search symbols
        results = await market_data_service.search_symbols(
            query=query,
            asset_type=asset_type,
            limit=limit
        )
        
        return [
            {
                "symbol": result["symbol"],
                "name": result["name"],
                "type": result["type"],
                "exchange": result["exchange"],
                "sector": result.get("sector"),
                "description": result.get("description")
            }
            for result in results
        ]
        
    except Exception as e:
        logger.error(f"Error searching symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols/popular", response_model=List[Dict[str, Any]])
async def get_popular_symbols():
    """Get popular asset symbols by category."""
    return {
        "equities": [
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "category": "Large Cap"},
            {"symbol": "QQQ", "name": "Invesco QQQ ETF", "category": "Technology"},
            {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "category": "Small Cap"},
            {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "category": "Total Market"}
        ],
        "bonds": [
            {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "category": "Long Treasury"},
            {"symbol": "IEF", "name": "iShares 7-10 Year Treasury Bond ETF", "category": "Intermediate Treasury"},
            {"symbol": "HYG", "name": "iShares iBoxx High Yield Corporate Bond ETF", "category": "High Yield"},
            {"symbol": "LQD", "name": "iShares iBoxx Investment Grade Corporate Bond ETF", "category": "Investment Grade"}
        ],
        "commodities": [
            {"symbol": "GLD", "name": "SPDR Gold Shares", "category": "Precious Metals"},
            {"symbol": "USO", "name": "United States Oil Fund", "category": "Energy"},
            {"symbol": "DBA", "name": "Invesco DB Agriculture Fund", "category": "Agriculture"},
            {"symbol": "UNG", "name": "United States Natural Gas Fund", "category": "Natural Gas"}
        ],
        "international": [
            {"symbol": "EFA", "name": "iShares MSCI EAFE ETF", "category": "Developed Markets"},
            {"symbol": "EEM", "name": "iShares MSCI Emerging Markets ETF", "category": "Emerging Markets"},
            {"symbol": "VEA", "name": "Vanguard FTSE Developed Markets ETF", "category": "Developed Markets"},
            {"symbol": "VWO", "name": "Vanguard FTSE Emerging Markets ETF", "category": "Emerging Markets"}
        ]
    }


@router.get("/data-quality/{symbol}", response_model=Dict[str, Any])
async def check_data_quality(
    symbol: str,
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date"),
    current_user: User = Depends(get_current_user)
):
    """Check data quality for a specific symbol."""
    try:
        market_data_service = MarketDataService()
        
        # Check data quality
        quality = await market_data_service.check_data_quality(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "symbol": symbol,
            "data_availability": quality["availability"],
            "missing_data_percentage": quality["missing_percentage"],
            "data_gaps": quality["gaps"],
            "quality_score": quality["quality_score"],
            "recommendations": quality["recommendations"]
        }
        
    except Exception as e:
        logger.error(f"Error checking data quality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/status", response_model=Dict[str, Any])
async def get_cache_status(
    current_user: User = Depends(get_current_user)
):
    """Get market data cache status."""
    try:
        market_data_service = MarketDataService()
        
        status = await market_data_service.get_cache_status()
        
        return {
            "cache_size": status["size"],
            "cache_hit_rate": status["hit_rate"],
            "last_updated": status["last_updated"],
            "symbols_cached": status["symbols_count"],
            "storage_usage": status["storage_usage"]
        }
        
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/refresh")
async def refresh_cache(
    symbols: List[str] = Query(..., description="Symbols to refresh"),
    current_user: User = Depends(get_current_user)
):
    """Refresh cache for specific symbols."""
    try:
        market_data_service = MarketDataService()
        
        result = await market_data_service.refresh_cache(symbols=symbols)
        
        return {
            "symbols_refreshed": result["refreshed"],
            "refresh_status": result["status"],
            "message": "Cache refresh completed"
        }
        
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))