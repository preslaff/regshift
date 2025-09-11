"""
Interactive visualization demonstration for regime-aware strategies.

This example creates comprehensive interactive charts showing:
1. Portfolio weight evolution over time
2. Regime transitions and forecasting accuracy  
3. Performance comparison vs benchmarks
4. Risk metrics evolution
"""

from src.regime_strategies.main import RegimeStrategyApp
from src.regime_strategies.backtesting.visualization import PerformanceVisualizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_enhanced_dashboard(result, static_mpt_result, regime_result):
    """Create comprehensive interactive dashboard."""
    
    # Create subplot structure
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Portfolio Weight Evolution', 'Cumulative Returns Comparison',
            'Regime Transitions & Predictions', 'Rolling Sharpe Ratios',
            'Drawdown Analysis', 'Risk Contribution Analysis',
            'Static MPT vs Regime Strategy', 'Regime Prediction Accuracy'
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"colspan": 2}, None]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # 1. Portfolio Weight Evolution (Stacked Area)
    weights = regime_result.portfolio_weights
    for i, asset in enumerate(weights.columns):
        fig.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[asset],
                mode='lines',
                stackgroup='weights',
                name=f'{asset} Weight',
                hovertemplate=f'{asset}: %{{y:.1%}}<extra></extra>',
                line=dict(width=1)
            ),
            row=1, col=1
        )
    
    # 2. Cumulative Returns Comparison
    # Generate cumulative returns for demonstration
    dates = regime_result.portfolio_returns.index
    regime_cum_returns = (1 + regime_result.portfolio_returns).cumprod()
    static_cum_returns = (1 + static_mpt_result.portfolio_returns).cumprod()
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=regime_cum_returns,
            mode='lines',
            name='Regime Strategy',
            line=dict(color='blue', width=2),
            hovertemplate='Regime: %{y:.1%}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=static_cum_returns,
            mode='lines',
            name='Static MPT',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='Static MPT: %{y:.1%}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Regime Transitions & Predictions
    predicted_regimes = regime_result.predicted_regimes
    actual_regimes = regime_result.actual_regimes
    
    # Add regime background colors
    regime_colors = {0: 'lightcoral', 1: 'lightblue', 2: 'lightgreen', 3: 'lightyellow'}
    
    fig.add_trace(
        go.Scatter(
            x=predicted_regimes.index,
            y=predicted_regimes,
            mode='markers',
            name='Predicted Regimes',
            marker=dict(size=8, color='blue'),
            hovertemplate='Predicted: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=actual_regimes.index,
            y=actual_regimes,
            mode='markers',
            name='Actual Regimes',
            marker=dict(size=8, color='red', symbol='x'),
            hovertemplate='Actual: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Rolling Sharpe Ratios (20-period rolling)
    rolling_window = min(20, len(regime_result.portfolio_returns))
    if rolling_window > 1:
        regime_rolling_sharpe = regime_result.portfolio_returns.rolling(rolling_window).mean() / regime_result.portfolio_returns.rolling(rolling_window).std() * (252**0.5)
        static_rolling_sharpe = static_mpt_result.portfolio_returns.rolling(rolling_window).mean() / static_mpt_result.portfolio_returns.rolling(rolling_window).std() * (252**0.5)
        
        fig.add_trace(
            go.Scatter(
                x=regime_rolling_sharpe.index,
                y=regime_rolling_sharpe,
                mode='lines',
                name='Regime Sharpe',
                line=dict(color='blue'),
                hovertemplate='Regime Sharpe: %{y:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=static_rolling_sharpe.index,
                y=static_rolling_sharpe,
                mode='lines',
                name='Static Sharpe',
                line=dict(color='red', dash='dash'),
                hovertemplate='Static Sharpe: %{y:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # 5. Drawdown Analysis
    regime_drawdown = (regime_cum_returns / regime_cum_returns.expanding().max()) - 1
    static_drawdown = (static_cum_returns / static_cum_returns.expanding().max()) - 1
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=regime_drawdown * 100,
            mode='lines',
            name='Regime DD',
            fill='tonexty',
            line=dict(color='blue'),
            hovertemplate='Regime DD: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=static_drawdown * 100,
            mode='lines',
            name='Static DD',
            line=dict(color='red', dash='dash'),
            hovertemplate='Static DD: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. Risk Contribution Analysis (simplified)
    if len(weights.columns) > 0:
        final_weights = weights.iloc[-1]
        fig.add_trace(
            go.Bar(
                x=list(final_weights.index),
                y=list(final_weights.values),
                name='Final Allocation',
                marker_color='lightblue',
                hovertemplate='%{x}: %{y:.1%}<extra></extra>'
            ),
            row=3, col=2
        )
    
    # 7. Performance Summary (text annotations in a combined chart)
    regime_total_return = regime_cum_returns.iloc[-1] - 1
    static_total_return = static_cum_returns.iloc[-1] - 1
    regime_sharpe = regime_result.portfolio_returns.mean() / regime_result.portfolio_returns.std() * (252**0.5)
    static_sharpe = static_mpt_result.portfolio_returns.mean() / static_mpt_result.portfolio_returns.std() * (252**0.5)
    
    metrics_text = f"""
    <b>Performance Comparison</b><br>
    <br>
    <b>Regime Strategy:</b><br>
    Total Return: {regime_total_return:.2%}<br>
    Sharpe Ratio: {regime_sharpe:.3f}<br>
    Max Drawdown: {regime_drawdown.min():.2%}<br>
    <br>
    <b>Static MPT:</b><br>
    Total Return: {static_total_return:.2%}<br>
    Sharpe Ratio: {static_sharpe:.3f}<br>
    Max Drawdown: {static_drawdown.min():.2%}<br>
    <br>
    <b>Regime Advantage:</b><br>
    Return Diff: {regime_total_return - static_total_return:+.2%}<br>
    Sharpe Diff: {regime_sharpe - static_sharpe:+.3f}
    """
    
    fig.add_annotation(
        text=metrics_text,
        xref="paper", yref="paper",
        x=0.25, y=0.25,
        xanchor="center", yanchor="middle",
        align="left",
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=10)
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Regime-Aware Investment Strategy Dashboard</b><br><span style='font-size:14px'>Interactive Analysis of Portfolio Evolution and Performance</span>",
            x=0.5,
            xanchor='center'
        ),
        height=1200,
        showlegend=True,
        template="plotly_white",
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_yaxes(title_text="Weight %", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", tickformat=".0%", row=1, col=2)
    fig.update_yaxes(title_text="Regime", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
    fig.update_yaxes(title_text="Weight %", tickformat=".0%", row=3, col=2)
    
    fig.update_xaxes(title_text="Date", row=4, col=1)
    
    return fig

def main():
    """Run interactive visualization demo."""
    print("Interactive Visualization Demo")
    print("=" * 50)
    print()
    
    # Initialize the app
    print("Initializing system...")
    app = RegimeStrategyApp()
    
    # Run regime strategy
    print("Running regime-aware strategy...")
    regime_result = app.run_backtest(
        start_date='2022-01-01',
        end_date='2022-12-31',
        assets=['SPY', 'TLT', 'GLD'],
        strategy_name='interactive_demo',
        save_results=False
    )
    
    # Get the backtest result object for visualizations
    backtest_result = app.backtest_engine.results
    
    # Run static MPT for comparison
    print("Running static MPT comparison...")
    app.config.optimization.default_method = "max_sharpe"
    static_result = app.run_backtest(
        start_date='2022-01-01',
        end_date='2022-12-31',
        assets=['SPY', 'TLT', 'GLD'],
        strategy_name='static_mpt_demo',
        save_results=False
    )
    static_backtest_result = app.backtest_engine.results
    
    # Create visualizations
    print("Creating interactive visualizations...")
    
    # 1. Standard weight evolution plot
    visualizer = PerformanceVisualizer(app.config)
    weight_plot = visualizer.plot_weight_evolution(backtest_result)
    if weight_plot:
        weight_plot.write_html("portfolio_weights.html")
        print("[OK] Portfolio weight evolution saved to 'portfolio_weights.html'")
    
    # 2. Enhanced dashboard
    try:
        dashboard = create_enhanced_dashboard(
            regime_result, 
            static_result, 
            backtest_result
        )
        dashboard.write_html("regime_strategy_dashboard.html")
        print("[OK] Interactive dashboard saved to 'regime_strategy_dashboard.html'")
        print()
        print("Dashboard Features:")
        print("- Portfolio weight evolution (stacked area chart)")
        print("- Cumulative returns comparison (regime vs static MPT)")
        print("- Regime transition visualization")
        print("- Rolling Sharpe ratio analysis")  
        print("- Drawdown comparison")
        print("- Risk contribution breakdown")
        print("- Performance metrics summary")
        
    except Exception as e:
        print(f"Enhanced dashboard creation failed: {e}")
    
    print("\nVisualization demo completed!")
    print("Open the HTML files in your browser to explore the interactive charts.")

if __name__ == "__main__":
    main()