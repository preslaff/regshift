"""
Dynamic Investment Strategies with Market Regimes

A sophisticated system for implementing regime-aware portfolio optimization.
"""

__version__ = "1.0.0"
__author__ = "Dynamic Investment Strategies Team"

from .config import Config
from .utils.logger import setup_logger

__all__ = ["Config", "setup_logger"]