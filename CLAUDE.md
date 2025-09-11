# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a **Dynamic Investment Strategies with Market Regimes** system - a Python-based solution that dynamically adjusts portfolio allocations based on predicted market regimes, improving upon traditional Modern Portfolio Theory (MPT) by addressing its sensitivity to static input assumptions.

## Development Architecture

The system follows a **5-stage modular architecture**:

### Stage 1: Historical Regime Identification (Unsupervised Learning)
- **Purpose**: Automatically label historical market data into distinct market regimes
- **Key Components**: 
  - Investment Clock theory implementation (4 regimes: heating, growing, stagflation, slowing)
  - Feature engineering for CPI and CLI transformations
  - Clustering models (K-Means, PCA) and HMMs
- **Core Libraries**: `pandas`, `scikit-learn`, `hmmlearn`
- **Output**: Time-series of historical regime labels

### Stage 2: Regime Forecasting (Supervised Learning)
- **Purpose**: Predict next period's market regime using current market information
- **Key Components**:
  - Supervised classification models (Logistic Regression, Random Forest, XGBoost)
  - Time-series considerations with rolling windows
  - Ensemble methods and naive predictor benchmarking
- **Core Libraries**: `scikit-learn`, `xgboost`
- **Critical Implementation**: Point-in-time data handling to avoid look-ahead bias

### Stage 3: Regime-Conditional Capital Market Assumptions (CMAs)
- **Purpose**: Calculate expected returns (μ) and covariance matrix (Σ) conditional on predicted regime
- **Key Components**: Dynamic CMA calculation using regime-filtered historical data
- **Focus**: Relative asset performance order within regimes

### Stage 4: Portfolio Optimization (Modern Portfolio Theory)
- **Purpose**: Construct optimal portfolio weights using regime-conditional CMAs
- **Optimization Methods**: 
  - Max Sharpe Ratio (primary focus for MVP)
  - Risk Parity
  - Minimum Variance
- **Core Libraries**: `scipy.optimize`, `cvxpy`
- **Constraints**: Sum to 1, long-only positions

### Stage 5: Strategy Simulation (Backtesting) and Evaluation
- **Purpose**: Simulate and evaluate dynamic investment strategy performance
- **Key Components**:
  - Point-in-time backtesting loop
  - Performance visualization (cumulative returns, weight evolution, multi-horizon plots)
  - Benchmark comparisons
- **Core Libraries**: `matplotlib`, `plotly`

## Data Sources and Requirements

- **Financial Data**: End-of-day (EOD) data from Yahoo Finance, Alpha Vantage
- **Macroeconomic Data**: Federal Reserve Economic Data (FRED) for CPI and CLI
- **Market Data**: VIX index, interest rates, other predictive features
- **Critical Requirement**: Point-in-time data integrity (no look-ahead bias)

## MVP Development Guidelines

### Start Simple Strategy
1. **Stage 1**: Use Investment Clock model (CPI/CLI based) over complex HMMs
2. **Stage 2**: Single robust classifier (Random Forest/Logistic Regression) with naive predictor benchmark
3. **Stage 3**: Full historical regime data for CMA calculation
4. **Stage 4**: Focus on MaxSharpe optimization with 3-5 asset classes
5. **Stage 5**: Basic cumulative returns and portfolio weight evolution plots

### Key Implementation Principles
- **Modular Design**: Separate functions/classes for each stage
- **Point-in-Time Data**: Paramount importance - only use data available at decision time
- **Benchmarking**: Always implement naive predictor and static MPT benchmarks
- **Interpretability**: Prefer explainable models for initial development
- **Robust Error Handling**: Comprehensive logging for data processing and model training

## Performance Success Metrics

- Regime-switching strategy shows higher cumulative returns than static benchmarks
- Multi-horizon plots show >70% outperformance scenarios
- MaxSharpe with regime info: target 73.4% outperformance rate
- Risk Parity with regime info: target >99% outperformance rate

## Development Best Practices

### Critical Data Handling
- **Point-in-Time Rule**: At decision point T, only use information known at T for T+1 decisions
- **Macroeconomic Data**: Account for data revisions and publication lags
- **No Look-Ahead Bias**: Most critical factor for realistic backtest results

### Code Organization
- Structure into distinct modules for each of the 5 stages
- Implement comprehensive error handling and logging
- Use version control from day one
- Create interactive visualizations with `Plotly` or `Bokeh`

### Model Development
- Start with interpretable models before moving to complex ones
- Implement ensemble methods for stability
- Use time-series appropriate cross-validation (e.g., `TimeSeriesSplit`)
- Focus on relative asset performance order within regimes

## Common Development Pitfalls to Avoid

1. **Look-Ahead Bias**: Using future information in historical decisions
2. **Data Snooping**: Over-optimizing on historical data
3. **Regime Persistence**: Underestimating the "stickiness" of market regimes
4. **Static Assumptions**: Falling back to traditional MPT approaches
5. **Complexity First**: Starting with complex models before establishing baselines