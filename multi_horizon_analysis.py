"""
Multi-Horizon Backtest Analysis for Regime-Aware Strategies.

This analysis tests the strategy across different start and end dates to evaluate
robustness and identify optimal periods. Creates the multi-horizon plots mentioned
in the development plan showing where regime strategies outperform vs underperform.

Key Features:
- Tests multiple start/end date combinations
- Creates heat map visualization of outperformance
- Identifies best and worst performing periods
- Analyzes regime strategy consistency across time horizons
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from src.regime_strategies.main import RegimeStrategyApp

class MultiHorizonAnalyzer:
    """Analyzer for multi-horizon backtest analysis."""
    
    def __init__(self, app: RegimeStrategyApp):
        """Initialize analyzer with regime strategy app."""
        self.app = app
        self.results = {}
        
    def generate_date_ranges(self, 
                           overall_start: str = '2020-01-01',
                           overall_end: str = '2023-12-31',
                           min_period_months: int = 12,
                           step_months: int = 6) -> List[Tuple[str, str]]:
        """Generate list of (start_date, end_date) pairs for analysis."""
        
        start_date = pd.to_datetime(overall_start)
        end_date = pd.to_datetime(overall_end)
        
        date_ranges = []
        
        # Generate start dates
        current_start = start_date
        while current_start < end_date - pd.DateOffset(months=min_period_months):
            
            # Generate end dates for this start date
            min_end = current_start + pd.DateOffset(months=min_period_months)
            current_end = min_end
            
            while current_end <= end_date:
                date_ranges.append((
                    current_start.strftime('%Y-%m-%d'),
                    current_end.strftime('%Y-%m-%d')
                ))
                current_end += pd.DateOffset(months=step_months)
            
            current_start += pd.DateOffset(months=step_months)
        
        return date_ranges
    
    def run_backtest_pair(self, start_date: str, end_date: str) -> Dict:
        """Run both regime and static MPT backtests for comparison."""
        
        assets = ['SPY', 'TLT', 'GLD']  # Standard asset mix
        
        try:
            # Run regime-aware strategy
            regime_result = self.app.run_backtest(
                start_date=start_date,
                end_date=end_date,
                assets=assets,
                strategy_name=f'regime_{start_date}_{end_date}',
                save_results=False
            )
            
            # Extract key metrics
            result = {
                'start_date': start_date,
                'end_date': end_date,
                'period_days': (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days,
                'regime_total_return': regime_result['total_return'],
                'regime_sharpe': regime_result['sharpe_ratio'],
                'regime_max_drawdown': regime_result['maximum_drawdown'],
                'regime_prediction_accuracy': regime_result['regime_prediction_accuracy'],
                'static_mpt_outperformance': regime_result.get('static_mpt_outperformance', 0),
                'regime_return_advantage': regime_result.get('regime_advantage_return', 0),
                'regime_sharpe_advantage': regime_result.get('regime_advantage_sharpe', 0),
                'success': True
            }
            
            # Determine outperformance
            result['outperformed_static'] = result['regime_return_advantage'] > 0
            
            return result
            
        except Exception as e:
            print(f"Error for period {start_date} to {end_date}: {e}")
            return {
                'start_date': start_date,
                'end_date': end_date,
                'success': False,
                'error': str(e)
            }
    
    def run_multi_horizon_analysis(self, 
                                 overall_start: str = '2021-01-01',
                                 overall_end: str = '2023-12-31',
                                 max_combinations: int = 20) -> pd.DataFrame:
        """Run comprehensive multi-horizon analysis."""
        
        print("Multi-Horizon Backtest Analysis")
        print("=" * 50)
        print(f"Analysis Period: {overall_start} to {overall_end}")
        print(f"Maximum Combinations: {max_combinations}")
        print()
        
        # Generate date ranges
        date_ranges = self.generate_date_ranges(
            overall_start=overall_start,
            overall_end=overall_end,
            min_period_months=12,
            step_months=6
        )
        
        # Limit combinations for demo
        if len(date_ranges) > max_combinations:
            # Select evenly distributed sample
            indices = np.linspace(0, len(date_ranges)-1, max_combinations, dtype=int)
            date_ranges = [date_ranges[i] for i in indices]
        
        print(f"Testing {len(date_ranges)} period combinations...")
        print()
        
        # Run analysis for each period
        results = []
        for i, (start_date, end_date) in enumerate(date_ranges):
            print(f"[{i+1}/{len(date_ranges)}] Testing: {start_date} to {end_date}")
            
            result = self.run_backtest_pair(start_date, end_date)
            results.append(result)
            
            if result['success']:
                outperf = "OUT" if result['outperformed_static'] else "UND"
                print(f"  Result: {outperf} | Return Advantage: {result['regime_return_advantage']:+.2%}")
            else:
                print(f"  Result: FAILED")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        successful_results = results_df[results_df['success'] == True].copy()
        
        if len(successful_results) == 0:
            print("No successful backtests completed.")
            return results_df
        
        # Calculate summary statistics
        outperformance_rate = successful_results['outperformed_static'].mean()
        avg_return_advantage = successful_results['regime_return_advantage'].mean()
        avg_sharpe_advantage = successful_results['regime_sharpe_advantage'].mean()
        
        print("\n" + "=" * 50)
        print("MULTI-HORIZON ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Successful Backtests: {len(successful_results)}/{len(results)}")
        print(f"Outperformance Rate: {outperformance_rate:.1%}")
        print(f"Average Return Advantage: {avg_return_advantage:+.2%}")
        print(f"Average Sharpe Advantage: {avg_sharpe_advantage:+.3f}")
        
        # Best and worst periods
        if len(successful_results) > 0:
            best_period = successful_results.loc[successful_results['regime_return_advantage'].idxmax()]
            worst_period = successful_results.loc[successful_results['regime_return_advantage'].idxmin()]
            
            print(f"\nBest Period:")
            print(f"  {best_period['start_date']} to {best_period['end_date']}")
            print(f"  Return Advantage: {best_period['regime_return_advantage']:+.2%}")
            print(f"  Sharpe Advantage: {best_period['regime_sharpe_advantage']:+.3f}")
            
            print(f"\nWorst Period:")
            print(f"  {worst_period['start_date']} to {worst_period['end_date']}")
            print(f"  Return Advantage: {worst_period['regime_return_advantage']:+.2%}")
            print(f"  Sharpe Advantage: {worst_period['regime_sharpe_advantage']:+.3f}")
        
        return results_df
    
    def create_multi_horizon_heatmap(self, results_df: pd.DataFrame) -> go.Figure:
        """Create multi-horizon performance heatmap."""
        
        # Filter successful results
        df = results_df[results_df['success'] == True].copy()
        
        if len(df) == 0:
            print("No successful results to plot.")
            return None
        
        # Prepare data for heatmap
        df['start_year'] = pd.to_datetime(df['start_date']).dt.year
        df['end_year'] = pd.to_datetime(df['end_date']).dt.year
        df['outperformance_numeric'] = df['outperformed_static'].astype(int)
        
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            values='regime_return_advantage',
            index='start_year',
            columns='end_year',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values * 100,  # Convert to percentage
            x=[f'{int(col)}' for col in pivot_data.columns],
            y=[f'{int(idx)}' for idx in pivot_data.index],
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title="Return Advantage (%)")
        ))
        
        fig.update_layout(
            title="Multi-Horizon Performance: Regime vs Static MPT<br><sub>Green = Regime Outperforms, Red = Static MPT Outperforms</sub>",
            xaxis_title="End Year",
            yaxis_title="Start Year",
            width=800,
            height=600
        )
        
        return fig
    
    def create_performance_distribution(self, results_df: pd.DataFrame) -> go.Figure:
        """Create distribution plot of return advantages."""
        
        df = results_df[results_df['success'] == True].copy()
        
        if len(df) == 0:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Return Advantage Distribution',
                'Sharpe Advantage Distribution', 
                'Regime Prediction Accuracy',
                'Period Length vs Performance'
            ]
        )
        
        # Return advantage histogram
        fig.add_trace(
            go.Histogram(
                x=df['regime_return_advantage'] * 100,
                nbinsx=20,
                name='Return Advantage',
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        # Sharpe advantage histogram
        fig.add_trace(
            go.Histogram(
                x=df['regime_sharpe_advantage'],
                nbinsx=20,
                name='Sharpe Advantage',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # Prediction accuracy
        fig.add_trace(
            go.Histogram(
                x=df['regime_prediction_accuracy'] * 100,
                nbinsx=10,
                name='Prediction Accuracy',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # Period length vs performance
        fig.add_trace(
            go.Scatter(
                x=df['period_days'],
                y=df['regime_return_advantage'] * 100,
                mode='markers',
                name='Period vs Performance',
                marker=dict(
                    color=df['regime_prediction_accuracy'],
                    colorscale='Viridis',
                    size=8,
                    showscale=True,
                    colorbar=dict(title="Prediction<br>Accuracy")
                ),
                hovertemplate='Days: %{x}<br>Return Adv: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Return Advantage (%)", row=1, col=1)
        fig.update_xaxes(title_text="Sharpe Advantage", row=1, col=2)
        fig.update_xaxes(title_text="Prediction Accuracy (%)", row=2, col=1)
        fig.update_xaxes(title_text="Period Length (Days)", row=2, col=2)
        fig.update_yaxes(title_text="Return Advantage (%)", row=2, col=2)
        
        fig.update_layout(
            title="Multi-Horizon Performance Analysis",
            height=800,
            showlegend=False
        )
        
        return fig

def main():
    """Run multi-horizon analysis demo."""
    
    print("Multi-Horizon Backtest Analysis Demo")
    print("=" * 60)
    print()
    
    # Initialize system
    app = RegimeStrategyApp()
    analyzer = MultiHorizonAnalyzer(app)
    
    # Run analysis (limited scope for demo)
    results_df = analyzer.run_multi_horizon_analysis(
        overall_start='2021-01-01',
        overall_end='2023-12-31', 
        max_combinations=12  # Limited for demo speed
    )
    
    # Create visualizations if we have results
    successful_results = results_df[results_df['success'] == True]
    
    if len(successful_results) > 0:
        print(f"\nCreating visualizations...")
        
        # Multi-horizon heatmap
        heatmap = analyzer.create_multi_horizon_heatmap(results_df)
        if heatmap:
            heatmap.write_html("multi_horizon_heatmap.html")
            print("  [OK] Multi-horizon heatmap saved to 'multi_horizon_heatmap.html'")
        
        # Performance distribution analysis
        distribution = analyzer.create_performance_distribution(results_df)
        if distribution:
            distribution.write_html("performance_distribution.html")
            print("  [OK] Performance distribution saved to 'performance_distribution.html'")
    
    # Save results to CSV for further analysis
    results_df.to_csv("multi_horizon_results.csv", index=False)
    print("  [OK] Results data saved to 'multi_horizon_results.csv'")
    
    print(f"\nMulti-horizon analysis completed!")
    print(f"Open HTML files in browser to explore interactive visualizations.")
    
    # Key insights summary
    if len(successful_results) > 0:
        outperf_rate = successful_results['outperformed_static'].mean()
        
        print(f"\n" + "=" * 60)
        print("KEY INSIGHTS")
        print("=" * 60)
        
        if outperf_rate >= 0.7:
            print(f"[EXCELLENT] {outperf_rate:.1%} outperformance rate exceeds 70% target")
        elif outperf_rate >= 0.5:
            print(f"[GOOD] {outperf_rate:.1%} outperformance rate shows regime value")
        else:
            print(f"[REVIEW] {outperf_rate:.1%} outperformance rate below expectations")
        
        high_accuracy_periods = successful_results[successful_results['regime_prediction_accuracy'] > 0.8]
        print(f"High Accuracy Periods (>80%): {len(high_accuracy_periods)}/{len(successful_results)}")
        
        consistent_outperformance = successful_results[successful_results['regime_return_advantage'] > 0]
        print(f"Consistent Outperformance: {len(consistent_outperformance)}/{len(successful_results)}")

if __name__ == "__main__":
    main()