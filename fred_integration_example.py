"""
FRED API Integration Example for Real Economic Data.

This example demonstrates how to integrate real economic data from the 
Federal Reserve Economic Data (FRED) API to improve regime identification
and forecasting accuracy.

To use this example:
1. Get a free FRED API key from: https://fred.stlouisfed.org/
2. Set environment variable: FRED_API_KEY=your_key_here
3. Install fredapi: pip install fredapi
4. Run this example to see the difference between synthetic and real data
"""

import os
from src.regime_strategies.main import RegimeStrategyApp
import pandas as pd

def check_fred_setup():
    """Check if FRED API is properly configured."""
    try:
        from fredapi import Fred
        fredapi_available = True
    except ImportError:
        fredapi_available = False
    
    fred_api_key = os.getenv('FRED_API_KEY')
    
    print("FRED API Setup Status:")
    print("=" * 40)
    print(f"fredapi library: {'[OK] Available' if fredapi_available else '[MISSING] (pip install fredapi)'}")
    print(f"FRED API Key: {'[OK] Configured' if fred_api_key else '[MISSING]'}")
    
    if fred_api_key and fredapi_available:
        try:
            fred = Fred(api_key=fred_api_key)
            # Test API connection with a simple query
            test_data = fred.get_series('GDP', limit=1)
            print(f"API Connection: [OK] Working")
            return True
        except Exception as e:
            print(f"API Connection: [FAILED] ({e})")
            return False
    
    return False

def demonstrate_economic_data_comparison():
    """Compare synthetic vs real economic data impact."""
    
    print("\nEconomic Data Comparison Demo")
    print("=" * 50)
    
    app = RegimeStrategyApp()
    
    # Check what data sources are being used
    print("\nData Sources:")
    print(f"- CPI Series: {app.config.data.cpi_series}")
    print(f"- FRED API Key: {'Configured' if app.config.data.fred_api_key else 'Not configured'}")
    
    try:
        # Load economic indicators to see data source
        print("\nLoading economic indicators...")
        data_loader = app.data_loader
        
        # Load economic indicators for a specific period
        econ_data = data_loader.load_economic_indicators(
            start_date=pd.to_datetime('2020-01-01').date(),
            end_date=pd.to_datetime('2023-12-31').date()
        )
        
        print(f"Economic data loaded: {econ_data.shape}")
        print("\nAvailable indicators:")
        for col in econ_data.columns:
            print(f"  - {col}")
        
        # Show data quality
        print(f"\nData quality:")
        print(f"- Total periods: {len(econ_data)}")
        print(f"- Missing values: {econ_data.isnull().sum().sum()}")
        print(f"- Date range: {econ_data.index[0]} to {econ_data.index[-1]}")
        
        # Display sample data
        print(f"\nSample data (latest 5 periods):")
        sample_data = econ_data.tail()
        for col in sample_data.columns:
            if not sample_data[col].isnull().all():
                print(f"  {col}:")
                for idx, val in sample_data[col].tail(3).items():
                    if pd.notna(val):
                        print(f"    {idx.strftime('%Y-%m')}: {val:.3f}")
        
    except Exception as e:
        print(f"Error loading economic data: {e}")
        return False
    
    return True

def run_fred_enhanced_backtest():
    """Run backtest with FRED data if available."""
    
    print("\nFRED-Enhanced Backtest")
    print("=" * 40)
    
    app = RegimeStrategyApp()
    
    if not app.config.data.fred_api_key:
        print("[WARNING] FRED API key not configured - using synthetic data")
        print("To use real economic data:")
        print("1. Get free API key: https://fred.stlouisfed.org/")
        print("2. Set environment: FRED_API_KEY=your_key_here")
        print("3. Install library: pip install fredapi")
    else:
        print("[OK] FRED API configured - using real economic data")
    
    # Run backtest with current configuration
    print("\nRunning regime-aware backtest...")
    result = app.run_backtest(
        start_date='2022-01-01',
        end_date='2022-12-31',
        assets=['SPY', 'TLT', 'GLD'],
        strategy_name='fred_demo',
        save_results=False
    )
    
    # Display results with data source information
    print("\nBacktest Results:")
    print("-" * 30)
    print(f"Strategy: {result['strategy_name']}")
    print(f"Period: {result['backtest_period']}")
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {result['maximum_drawdown']:.2%}")
    print(f"Regime Accuracy: {result['regime_prediction_accuracy']:.1%}")
    
    # Benchmark comparison
    print(f"\nBenchmark Comparison:")
    print(f"Static MPT Outperformance: {result.get('static_mpt_outperformance', 0):.1%}")
    print(f"Regime Return Advantage: {result.get('regime_advantage_return', 0):+.2%}")
    print(f"Regime Sharpe Advantage: {result.get('regime_advantage_sharpe', 0):+.3f}")
    
    # Data source summary
    data_source = "Real FRED Data" if app.config.data.fred_api_key else "Synthetic Data"
    print(f"\nData Source Used: {data_source}")
    
    return result

def setup_fred_integration():
    """Provide setup instructions for FRED integration."""
    
    print("\n" + "=" * 60)
    print("FRED API Integration Setup Guide")
    print("=" * 60)
    
    print("""
 FRED (Federal Reserve Economic Data) provides access to:
   - Consumer Price Index (CPI) for inflation measurement
   - Composite Leading Indicator (CLI) for economic growth
   - Interest rates and other macroeconomic indicators
   - High-quality, official economic data for regime identification

 Setup Steps:

1. Get Free API Key:
   -> Visit: https://fred.stlouisfed.org/
   -> Create account (free)
   -> Go to "My Account" -> "API Keys"
   -> Generate new API key

2. Install Required Library:
   pip install fredapi

3. Configure API Key (choose one method):
   
   Method A - Environment Variable:
   -> Windows: set FRED_API_KEY=your_key_here
   -> Linux/Mac: export FRED_API_KEY=your_key_here
   
   Method B - .env File:
   -> Create .env file in project root
   -> Add: FRED_API_KEY=your_key_here
   
   Method C - Direct Configuration:
   -> Edit config.py: fred_api_key = "your_key_here"

4. Verify Setup:
   -> Run this script to test connection
   -> Check for "[OK] Working" status messages

 Benefits of Real Economic Data:
   - Improved regime identification accuracy
   - Better forecasting performance  
   - More realistic backtest results
   - Professional-grade economic indicators
   - Historical data consistency

 Current Configuration:
""")
    
    # Show current status
    fred_configured = bool(os.getenv('FRED_API_KEY'))
    print(f"   FRED API Key: {'[OK] Configured' if fred_configured else '[MISSING] Not configured'}")
    
    try:
        from fredapi import Fred
        print(f"   fredapi Library: [OK] Available")
    except ImportError:
        print(f"   fredapi Library: [MISSING] Missing")
        print(f"   -> Install with: pip install fredapi")

def main():
    """Main demo function."""
    
    print("FRED API Integration for Economic Data")
    print("=" * 50)
    print()
    
    # Check setup status
    fred_ready = check_fred_setup()
    
    # Demonstrate economic data loading
    if demonstrate_economic_data_comparison():
        print("[OK] Economic data loading successful")
    
    # Run enhanced backtest
    result = run_fred_enhanced_backtest()
    
    # Show setup guide
    setup_fred_integration()
    
    print(f"\nDemo completed!")
    if not fred_ready:
        print("\nTip: Tip: Set up FRED API for enhanced regime identification accuracy!")

if __name__ == "__main__":
    main()