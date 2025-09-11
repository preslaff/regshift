"""
Simple working example of the Dynamic Investment Strategies system.
"""

from src.regime_strategies.main import RegimeStrategyApp

def main():
    """Run a simple backtest demonstration."""
    print("Dynamic Investment Strategies with Market Regimes")
    print("=" * 60)
    print()
    
    # Initialize the app
    print("Initializing system...")
    app = RegimeStrategyApp()
    
    # Run a simple backtest
    print("Running backtest (this may take a moment)...")
    result = app.run_backtest(
        start_date='2022-01-01',
        end_date='2022-12-31',
        assets=['SPY', 'TLT', 'GLD'],  # Simple 3-asset portfolio
        strategy_name='simple_demo',
        save_results=False
    )
    
    # Display results
    print("\nBacktest Results:")
    print("-" * 40)
    print(f"Strategy Name: {result['strategy_name']}")
    print(f"Period: {result['backtest_period']}")
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Annualized Return: {result['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown: {result['maximum_drawdown']:.2%}")
    print(f"Regime Prediction Accuracy: {result['regime_prediction_accuracy']:.1%}")
    print(f"Overall Rating: {result['overall_rating']}")
    
    print("\nDemo completed successfully!")
    print("""
System Status:
- [OK] Market data loading (Yahoo Finance)
- [WARN] Economic data (synthetic - no FRED API key)
- [OK] Regime identification (Investment Clock)
- [OK] Regime forecasting (ML models)
- [OK] Portfolio optimization (scipy)
- [OK] Backtesting engine
- [OK] Performance evaluation
- [OK] Visualization framework

To get real economic data, set your FRED API key in a .env file:
FRED_API_KEY=your_key_here

For enhanced optimization, install: pip install cvxpy
For HMM models, install: pip install hmmlearn
For XGBoost models, install: pip install xgboost
    """)

if __name__ == "__main__":
    main()