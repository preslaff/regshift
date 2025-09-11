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

## Git Workflow and Version Control

### Repository Structure
This repository follows Git best practices with comprehensive `.gitignore` for Python financial projects:

```
.gitignore includes:
- Python artifacts (__pycache__, *.pyc, etc.)
- Virtual environments (.venv, venv/)
- Data files (*.csv, *.h5, *.parquet)
- API keys and secrets (.env, secrets.json)
- IDE files (.vscode/, .idea/)
- Large result files (backtest_results/, simulation_outputs/)
- Financial data cache (market_data_cache/)
```

### Branching Strategy
- **main/master**: Production-ready code only
- **develop**: Integration branch for features
- **feature/***: Individual feature development
- **hotfix/***: Critical production fixes

### Commit Message Guidelines
Follow conventional commit format:
```
type(scope): brief description

Detailed explanation of changes made, why they were necessary,
and any important implementation details.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Common types**: 
- `feat`: New feature implementation
- `fix`: Bug fixes
- `refactor`: Code restructuring without behavior changes
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `docs`: Documentation updates
- `chore`: Maintenance tasks

### Development Workflow
1. **Start new feature**:
   ```bash
   git checkout -b feature/regime-forecasting-improvements
   ```

2. **Regular commits during development**:
   ```bash
   git add .
   git commit -m "feat(forecasting): add XGBoost ensemble model
   
   - Implement XGBoost classifier for regime prediction
   - Add hyperparameter tuning with grid search
   - Improve prediction accuracy by 3.2%
   
   🤖 Generated with [Claude Code](https://claude.ai/code)
   
   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

3. **Before merging**:
   ```bash
   # Ensure tests pass
   python -m pytest tests/
   
   # Run linting
   python -m flake8 src/
   
   # Update documentation if needed
   git add .
   git commit -m "docs: update README with XGBoost requirements"
   ```

4. **Merge to develop**:
   ```bash
   git checkout develop
   git merge feature/regime-forecasting-improvements
   git branch -d feature/regime-forecasting-improvements
   ```

### Important Git Practices for Financial Code

#### 1. Never Commit Sensitive Data
```bash
# Check what's being committed
git diff --cached

# Items to NEVER commit:
- API keys (FRED_API_KEY, etc.)
- Real trading credentials  
- Proprietary financial data
- Personal account information
- Large data files (>100MB)
```

#### 2. Point-in-Time Data Integrity
- Commit data snapshots with timestamps
- Tag releases with backtest periods: `git tag -a v1.0-backtest-2020-2023`
- Document data sources and versions in commit messages

#### 3. Reproducible Research
```bash
# Tag major backtests
git tag -a backtest-2024-q1 -m "Backtest results for Q1 2024 regime strategy"

# Include environment info in commits
pip freeze > requirements.txt
git add requirements.txt
git commit -m "chore: update dependencies for reproducible environment"
```

#### 4. Code Review Checklist
Before merging any financial modeling code:
- [ ] No look-ahead bias in backtesting logic
- [ ] Point-in-time data handling verified
- [ ] Performance metrics properly calculated
- [ ] Risk management constraints implemented
- [ ] Documentation updated for strategy changes
- [ ] Tests cover edge cases (market crashes, missing data)

### Remote Repository Management
When working with GitHub/GitLab:

1. **Push feature branches**:
   ```bash
   git push -u origin feature/portfolio-optimization
   ```

2. **Create pull requests** with detailed descriptions:
   - What financial problem does this solve?
   - What models/strategies are implemented?
   - Backtest performance summary
   - Risk considerations
   - Breaking changes (if any)

3. **Protect main branch** with required reviews for financial code

### Emergency Procedures
For critical fixes in production strategies:
```bash
# Hotfix workflow
git checkout -b hotfix/sharpe-calculation-fix main
# Make critical fix
git commit -m "fix(optimization): correct Sharpe ratio calculation"
git checkout main
git merge hotfix/sharpe-calculation-fix
git tag -a v1.0.1-hotfix -m "Fix critical Sharpe ratio bug"
```

## Common Development Pitfalls to Avoid

1. **Look-Ahead Bias**: Using future information in historical decisions
2. **Data Snooping**: Over-optimizing on historical data
3. **Regime Persistence**: Underestimating the "stickiness" of market regimes
4. **Static Assumptions**: Falling back to traditional MPT approaches
5. **Complexity First**: Starting with complex models before establishing baselines
6. **Version Control Neglect**: Not committing data handling logic or model parameters
7. **Credential Exposure**: Accidentally committing API keys or sensitive financial data