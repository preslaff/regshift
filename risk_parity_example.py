"""
Risk Parity vs Max Sharpe comparison example.

This example demonstrates the difference between Risk Parity and Max Sharpe
optimization methods in regime-aware portfolio strategies.
"""

from src.regime_strategies.main import RegimeStrategyApp

def main():
    """Run Risk Parity vs Max Sharpe comparison."""
    print("Risk Parity vs Max Sharpe Comparison")
    print("=" * 60)
    print()
    
    # Initialize the app
    print("Initializing system...")
    app = RegimeStrategyApp()
    
    # Test assets
    assets = ['SPY', 'TLT', 'GLD']
    period = {
        'start_date': '2022-01-01',
        'end_date': '2022-12-31'
    }
    
    # Run Max Sharpe strategy
    print("\nRunning Max Sharpe strategy...")
    app.config.optimization.default_method = "max_sharpe"
    max_sharpe_result = app.run_backtest(
        assets=assets,
        strategy_name='max_sharpe_demo',
        save_results=False,
        **period
    )
    
    # Run Risk Parity strategy  
    print("\nRunning Risk Parity strategy...")
    app.config.optimization.default_method = "risk_parity"
    risk_parity_result = app.run_backtest(
        assets=assets,
        strategy_name='risk_parity_demo',
        save_results=False,
        **period
    )
    
    # Display comparison
    print("\nStrategy Comparison Results:")
    print("=" * 60)
    
    print(f"\nMax Sharpe Strategy:")
    print(f"  Total Return: {max_sharpe_result['total_return']:.2%}")
    print(f"  Sharpe Ratio: {max_sharpe_result['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {max_sharpe_result['maximum_drawdown']:.2%}")
    print(f"  Static MPT Outperformance: {max_sharpe_result.get('static_mpt_outperformance', 0):.1%}")
    print(f"  Regime Advantage (Return): {max_sharpe_result.get('regime_advantage_return', 0):+.2%}")
    
    print(f"\nRisk Parity Strategy:")
    print(f"  Total Return: {risk_parity_result['total_return']:.2%}")
    print(f"  Sharpe Ratio: {risk_parity_result['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {risk_parity_result['maximum_drawdown']:.2%}")
    print(f"  Static MPT Outperformance: {risk_parity_result.get('static_mpt_outperformance', 0):.1%}")
    print(f"  Regime Advantage (Return): {risk_parity_result.get('regime_advantage_return', 0):+.2%}")
    
    # Calculate winner
    print(f"\nPerformance Summary:")
    print(f"-" * 40)
    
    if risk_parity_result['total_return'] > max_sharpe_result['total_return']:
        winner = "Risk Parity"
        return_advantage = risk_parity_result['total_return'] - max_sharpe_result['total_return']
    else:
        winner = "Max Sharpe"
        return_advantage = max_sharpe_result['total_return'] - risk_parity_result['total_return']
    
    if risk_parity_result['sharpe_ratio'] > max_sharpe_result['sharpe_ratio']:
        sharpe_winner = "Risk Parity"
        sharpe_advantage = risk_parity_result['sharpe_ratio'] - max_sharpe_result['sharpe_ratio']
    else:
        sharpe_winner = "Max Sharpe"
        sharpe_advantage = max_sharpe_result['sharpe_ratio'] - risk_parity_result['sharpe_ratio']
    
    print(f"Return Winner: {winner} (+{return_advantage:.2%})")
    print(f"Risk-Adjusted Winner: {sharpe_winner} (+{sharpe_advantage:.3f} Sharpe)")
    
    # Static MPT comparison
    rp_static_adv = risk_parity_result.get('static_mpt_outperformance', 0)
    ms_static_adv = max_sharpe_result.get('static_mpt_outperformance', 0)
    
    if rp_static_adv > ms_static_adv:
        static_winner = "Risk Parity"
        static_advantage = rp_static_adv - ms_static_adv
    else:
        static_winner = "Max Sharpe"
        static_advantage = ms_static_adv - rp_static_adv
    
    print(f"Better vs Static MPT: {static_winner} (+{static_advantage:.1%} outperformance)")
    
    print("\nDemo completed successfully!")
    print("""
Key Insights:
- Risk Parity aims for equal risk contribution from each asset
- Max Sharpe optimizes for highest risk-adjusted returns
- Both strategies benefit from regime-aware allocation
- Compare static MPT outperformance to see regime value
    """)

if __name__ == "__main__":
    main()