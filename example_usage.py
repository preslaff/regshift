"""
Example usage of the Dynamic Investment Strategies with Market Regimes system.

This script demonstrates how to use the various components of the system
for different types of analysis and backtesting scenarios.
"""

from datetime import datetime, date
import pandas as pd
import numpy as np
from pathlib import Path

# Import the main components
from src.regime_strategies.main import RegimeStrategyApp
from src.regime_strategies.config import Config
from src.regime_strategies.utils.logger import setup_logger


def basic_backtest_example():
    """
    Example 1: Basic backtest with default settings.
    """
    print("=" * 60)
    print("Example 1: Basic Backtest")
    print("=" * 60)
    
    # Initialize the application
    app = RegimeStrategyApp()
    
    # Run a basic backtest
    result = app.run_backtest(
        start_date="2015-01-01",
        end_date="2023-12-31",
        strategy_name="basic_regime_strategy",
        save_results=True
    )
    
    print(f"\nBasic Backtest Results:")
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Annualized Return: {result['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown: {result['maximum_drawdown']:.2%}")
    print(f"Regime Prediction Accuracy: {result['regime_prediction_accuracy']:.2%}")
    print(f"Overall Rating: {result['overall_rating']}")
    
    return result


def custom_asset_backtest_example():
    """
    Example 2: Backtest with custom asset universe.
    """
    print("\n" + "=" * 60)
    print("Example 2: Custom Asset Universe Backtest")
    print("=" * 60)
    
    app = RegimeStrategyApp()
    
    # Define custom asset universe
    custom_assets = ["SPY", "QQQ", "IWM", "TLT", "GLD", "VNQ"]
    
    result = app.run_backtest(
        start_date="2020-01-01",
        end_date="2023-12-31",
        assets=custom_assets,
        benchmark="SPY",
        strategy_name="custom_assets_strategy",
        save_results=True
    )
    
    print(f"\nCustom Assets Backtest Results:")
    print(f"Assets: {', '.join(custom_assets)}")
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"Regime Accuracy: {result['regime_prediction_accuracy']:.2%}")
    
    return result


def strategy_comparison_example():
    """
    Example 3: Compare multiple strategy configurations.
    """
    print("\n" + "=" * 60)
    print("Example 3: Strategy Comparison")
    print("=" * 60)
    
    app = RegimeStrategyApp()
    
    # Define different strategy configurations
    strategy_configs = [
        {
            "name": "conservative_strategy",
            "start_date": "2015-01-01",
            "end_date": "2023-12-31",
            "assets": ["SPY", "TLT", "GLD"],
            "benchmark": "SPY"
        },
        {
            "name": "aggressive_strategy",
            "start_date": "2015-01-01",
            "end_date": "2023-12-31",
            "assets": ["QQQ", "IWM", "EFA", "EEM"],
            "benchmark": "SPY"
        },
        {
            "name": "balanced_strategy",
            "start_date": "2015-01-01",
            "end_date": "2023-12-31",
            "assets": ["SPY", "TLT", "GLD", "VNQ", "DBC"],
            "benchmark": "SPY"
        }
    ]
    
    # Compare strategies
    comparison = app.compare_strategies(strategy_configs, save_results=True)
    
    print(f"\nStrategy Comparison Results:")
    print(f"Strategies compared: {len(comparison['strategies'])}")
    
    # Show Sharpe ratio rankings
    if 'sharpe_ratio' in comparison['rankings']:
        print(f"\nSharpe Ratio Rankings:")
        for i, strategy in enumerate(comparison['rankings']['sharpe_ratio'], 1):
            sharpe = comparison['performance_comparison']['sharpe_ratio'].get(strategy, 'N/A')
            print(f"  {i}. {strategy}: {sharpe:.3f}")
    
    return comparison


def custom_configuration_example():
    """
    Example 4: Using custom configuration.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = Config()
    
    # Modify configuration settings
    config.regime.regime_method = "investment_clock"  # Use Investment Clock method
    config.optimization.default_method = "max_sharpe"  # Maximum Sharpe ratio optimization
    config.backtest.rebalance_frequency = "quarterly"  # Quarterly rebalancing
    config.backtest.transaction_costs = 0.002  # 20 bps transaction costs
    
    # Initialize app with custom config
    app = RegimeStrategyApp()
    app.config = config
    
    result = app.run_backtest(
        start_date="2018-01-01",
        end_date="2023-12-31",
        strategy_name="custom_config_strategy",
        save_results=True
    )
    
    print(f"\nCustom Configuration Results:")
    print(f"Regime Method: {config.regime.regime_method}")
    print(f"Optimization: {config.optimization.default_method}")
    print(f"Rebalance Frequency: {config.backtest.rebalance_frequency}")
    print(f"Transaction Costs: {config.backtest.transaction_costs:.1%}")
    print(f"\nPerformance:")
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    
    return result


def regime_analysis_example():
    """
    Example 5: Detailed regime analysis.
    """
    print("\n" + "=" * 60)
    print("Example 5: Detailed Regime Analysis")
    print("=" * 60)
    
    app = RegimeStrategyApp()
    
    # Run backtest to get detailed results
    result = app.run_backtest(
        start_date="2015-01-01",
        end_date="2023-12-31",
        strategy_name="regime_analysis",
        save_results=False
    )
    
    # Get detailed backtest results
    backtest_result = app.backtest_engine.get_results()
    
    if backtest_result:
        print(f"\nRegime Analysis:")
        print(f"Regime Prediction Accuracy: {backtest_result.regime_prediction_accuracy:.2%}")
        
        # Show regime distribution
        if not backtest_result.actual_regimes.empty:
            regime_counts = backtest_result.actual_regimes.value_counts().sort_index()
            print(f"\nRegime Distribution:")
            for regime, count in regime_counts.items():
                percentage = count / len(backtest_result.actual_regimes) * 100
                print(f"  Regime {regime}: {count} periods ({percentage:.1f}%)")
        
        # Show performance by regime
        if backtest_result.regime_performance:
            print(f"\nPerformance by Regime:")
            for regime, metrics in backtest_result.regime_performance.items():
                annualized_return = metrics.get('annualized_return', 0)
                sharpe_ratio = metrics.get('sharpe_ratio', 0)
                print(f"  Regime {regime}: Return {annualized_return:.2%}, Sharpe {sharpe_ratio:.3f}")
    
    return backtest_result


def live_simulation_example():
    """
    Example 6: Live simulation (out-of-sample testing).
    """
    print("\n" + "=" * 60)
    print("Example 6: Live Simulation")
    print("=" * 60)
    
    app = RegimeStrategyApp()
    
    # Run live simulation for recent period
    result = app.run_live_simulation(
        start_date="2023-01-01",
        end_date="2023-12-31",
        rebalance_frequency="monthly",
        assets=["SPY", "TLT", "GLD", "VNQ"]
    )
    
    print(f"\nLive Simulation Results:")
    print(f"Period: {result['backtest_period']}")
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"Regime Accuracy: {result['regime_prediction_accuracy']:.2%}")
    
    return result


def main():
    """
    Run all examples.
    """
    print("Dynamic Investment Strategies with Market Regimes - Examples")
    print("=" * 80)
    print("""
This script demonstrates various usage patterns for the regime-aware
investment strategy system. Each example shows different capabilities:

1. Basic backtest with default settings
2. Custom asset universe
3. Strategy comparison
4. Custom configuration
5. Detailed regime analysis  
6. Live simulation
    """)
    
    # Setup logging
    setup_logger(log_level="INFO")
    
    try:
        # Run examples
        basic_result = basic_backtest_example()
        custom_result = custom_asset_backtest_example() 
        comparison_result = strategy_comparison_example()
        custom_config_result = custom_configuration_example()
        regime_analysis_result = regime_analysis_example()
        live_simulation_result = live_simulation_example()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print(f"""
Summary of Results:
- Basic Strategy Total Return: {basic_result['total_return']:.2%}
- Custom Assets Total Return: {custom_result['total_return']:.2%}
- Best Strategy from Comparison: {comparison_result['rankings']['total_return'][0] if 'total_return' in comparison_result['rankings'] else 'N/A'}
- Custom Config Total Return: {custom_config_result['total_return']:.2%}
- Live Simulation Total Return: {live_simulation_result['total_return']:.2%}

Check the 'results/' directory for saved outputs, visualizations, and detailed reports.
        """)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This might be due to missing data or configuration issues.")
        print("Please ensure you have proper data access (FRED API key, etc.) configured.")


if __name__ == "__main__":
    main()