"""
FastAPI main application entry point.
Dynamic Investment Strategies Web Application Backend.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger

from app.core.config import settings
from app.core.database import database
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting Dynamic Investment Strategies API")
    await database.connect()
    logger.info("Database connected successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Dynamic Investment Strategies API")
    await database.disconnect()
    logger.info("Database disconnected")


# Create FastAPI application
app = FastAPI(
    title="Dynamic Investment Strategies API",
    description="""
    A comprehensive API for regime-aware investment strategy analysis and portfolio optimization.
    
    ## Features
    
    * **Regime Analysis**: Market regime identification using Investment Clock, K-Means, and HMM methods
    * **Portfolio Optimization**: Max Sharpe, Risk Parity, and Minimum Variance optimization
    * **Multi-Horizon Backtesting**: Test strategies across different time periods
    * **Real Economic Data**: Integration with FRED API for authentic economic indicators
    * **Interactive Scenarios**: Custom scenario analysis and what-if modeling
    
    ## Authentication
    
    Most endpoints require JWT authentication. Use `/auth/login` to obtain access tokens.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Serve static files (for frontend if needed)
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Dynamic Investment Strategies API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "operational"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "database": "connected" if database.is_connected else "disconnected",
        "timestamp": str(pd.Timestamp.now())
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )