"""
Portfolio models for investment strategy management.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base


class Portfolio(Base):
    """Portfolio model."""
    
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    assets = Column(JSON, nullable=False)  # List of asset symbols
    weights = Column(JSON, nullable=True)  # Asset weights
    benchmark = Column(String(20), default="SPY")
    
    optimization_method = Column(String(50), default="max_sharpe")
    regime_method = Column(String(50), default="investment_clock")
    
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User")
    analyses = relationship("PortfolioAnalysis", back_populates="portfolio")
    
    @property
    def latest_analysis(self):
        """Get the most recent analysis."""
        if self.analyses:
            return sorted(self.analyses, key=lambda x: x.created_at, reverse=True)[0]
        return None
    
    @property
    def performance_summary(self):
        """Get performance summary from latest analysis."""
        latest = self.latest_analysis
        if latest and latest.results:
            return latest.results.get("performance_summary", {})
        return {}
    
    @property
    def performance_metrics(self):
        """Get performance metrics from latest analysis."""
        latest = self.latest_analysis
        if latest and latest.results:
            return latest.results.get("performance_metrics", {})
        return {}


class PortfolioAnalysis(Base):
    """Portfolio analysis results model."""
    
    __tablename__ = "portfolio_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    
    analysis_type = Column(String(50), nullable=False)  # backtest, optimization, etc.
    
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    parameters = Column(JSON, nullable=True)  # Analysis parameters
    results = Column(JSON, nullable=True)     # Analysis results
    
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="analyses")