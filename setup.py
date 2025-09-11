"""
Setup script for Dynamic Investment Strategies with Market Regimes.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_path = Path(__file__).parent / filename
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

# Read version
version = "1.0.0"

setup(
    name="regime-strategies",
    version=version,
    author="Dynamic Investment Strategies Team",
    author_email="team@regime-strategies.com",
    description="Dynamic Investment Strategies with Market Regimes - A sophisticated system for regime-aware portfolio optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/regime-strategies/dynamic-investment-strategies",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "web": [
            "streamlit>=1.20.0",
            "dash>=2.10.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "regime-strategies=regime_strategies.main:main",
            "regime-backtest=regime_strategies.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "regime_strategies": [
            "config/*.yaml",
            "data/*.csv",
        ],
    },
    zip_safe=False,
    keywords=[
        "finance",
        "investment",
        "portfolio",
        "optimization", 
        "regime",
        "machine learning",
        "backtesting",
        "quantitative finance",
        "asset allocation",
        "modern portfolio theory"
    ],
    project_urls={
        "Bug Reports": "https://github.com/regime-strategies/dynamic-investment-strategies/issues",
        "Source": "https://github.com/regime-strategies/dynamic-investment-strategies",
        "Documentation": "https://regime-strategies.readthedocs.io/",
    },
)