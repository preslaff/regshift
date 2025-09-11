"""
Logging utilities for the regime strategies system.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "1 week",
    retention: str = "1 month"
) -> None:
    """
    Setup logging configuration using loguru.
    
    Parameters:
    -----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    log_file : str, optional
        Path to log file. If None, logs only to console
    rotation : str
        Log rotation policy
    retention : str
        Log retention policy
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    logger.info(f"Logger initialized with level: {log_level}")


def get_logger(name: str):
    """
    Get a logger instance for a specific module.
    
    Parameters:
    -----------
    name : str
        Logger name (typically __name__)
    
    Returns:
    --------
    logger
        Configured logger instance
    """
    return logger.bind(name=name)