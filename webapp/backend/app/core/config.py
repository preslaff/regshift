"""
Application configuration management.
"""

from pydantic import BaseSettings, Field
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "Dynamic Investment Strategies API"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    
    # Database
    DATABASE_URL: str = Field(env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=10, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Authentication
    JWT_SECRET_KEY: str = Field(env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    
    # External APIs
    FRED_API_KEY: Optional[str] = Field(default=None, env="FRED_API_KEY")
    ALPHA_VANTAGE_KEY: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_KEY")
    
    # Redis/Celery (for background tasks)
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # File Storage
    UPLOAD_DIR: Path = Field(default=Path("uploads"), env="UPLOAD_DIR")
    MAX_UPLOAD_SIZE: int = Field(default=10 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 10MB
    
    # Portfolio Analysis
    DEFAULT_ASSETS: List[str] = Field(
        default=["SPY", "TLT", "GLD", "IWM", "EFA"], 
        env="DEFAULT_ASSETS"
    )
    MAX_ASSETS_PER_PORTFOLIO: int = Field(default=20, env="MAX_ASSETS_PER_PORTFOLIO")
    DEFAULT_BENCHMARK: str = Field(default="SPY", env="DEFAULT_BENCHMARK")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Validate required settings
def validate_settings():
    """Validate critical settings on startup."""
    if not settings.JWT_SECRET_KEY:
        raise ValueError("JWT_SECRET_KEY environment variable is required")
    
    if not settings.DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is required")
    
    # Create upload directory if it doesn't exist
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Validate on import
validate_settings()