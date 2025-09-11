"""
Visualization tools for backtesting results and performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from loguru import logger
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Plotting functionality will be limited.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not available. Interactive plotting functionality will be limited.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from ..config import Config
from .backtest_engine import BacktestResult


class PerformanceVisualizer:
    """
    Visualization tools for regime-aware strategy analysis.
    """
    
    def __init__(self, config: Config):
        """
        Initialize performance visualizer.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        
        # Set up plotting style
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
            
        if SEABORN_AVAILABLE:
            sns.set_palette("husl")
    
    def create_performance_dashboard(self, 
                                   result: BacktestResult,
                                   save_path: Optional[str] = None) -> Any:
        """
        Create comprehensive performance dashboard.
        
        Parameters:
        -----------
        result : BacktestResult
            Backtest results
        save_path : str, optional
            Path to save the dashboard
        
        Returns:
        --------
        Dashboard figure object
        """
        self.logger.info("Creating performance dashboard")
        
        if PLOTLY_AVAILABLE:
            return self._create_plotly_dashboard(result, save_path)
        elif MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_dashboard(result, save_path)
        else:
            self.logger.error("No plotting libraries available")
            return None
    
    def _create_plotly_dashboard(self, 
                               result: BacktestResult,
                               save_path: Optional[str] = None):
        """Create interactive Plotly dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cumulative Returns', 'Portfolio Weights Evolution',
                'Regime Predictions vs Actual', 'Rolling Sharpe Ratio',
                'Drawdown Analysis', 'Performance by Regime'
            ),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "bar"}]],
            vertical_spacing=0.08
        )
        
        # 1. Cumulative Returns
        if not result.portfolio_returns.empty:
            cumulative_returns = (1 + result.portfolio_returns).cumprod()
            
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values,
                    name='Strategy',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            if result.benchmark_returns is not None:
                benchmark_cumulative = (1 + result.benchmark_returns).cumprod()
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_cumulative.index,
                        y=benchmark_cumulative.values,
                        name='Benchmark',
                        line=dict(color='gray', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
        
        # 2. Portfolio Weights Evolution
        if not result.portfolio_weights.empty:
            for i, asset in enumerate(result.portfolio_weights.columns):
                fig.add_trace(
                    go.Scatter(
                        x=result.portfolio_weights.index,
                        y=result.portfolio_weights[asset].values,
                        name=asset,
                        stackgroup='weights',
                        mode='none'
                    ),
                    row=1, col=2
                )
        
        # 3. Regime Predictions vs Actual
        if not result.predicted_regimes.empty and not result.actual_regimes.empty:
            fig.add_trace(
                go.Scatter(
                    x=result.actual_regimes.index,
                    y=result.actual_regimes.values,
                    name='Actual Regime',
                    mode='lines+markers',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=result.predicted_regimes.index,
                    y=result.predicted_regimes.values,
                    name='Predicted Regime',
                    mode='lines+markers',
                    line=dict(color='red', width=1, dash='dot')
                ),
                row=2, col=1
            )
        
        # 4. Rolling Sharpe Ratio
        if not result.portfolio_returns.empty and len(result.portfolio_returns) > 60:
            rolling_sharpe = result.portfolio_returns.rolling(60).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    name='Rolling Sharpe (60d)',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
        
        # 5. Drawdown Analysis
        if not result.portfolio_returns.empty:
            cumulative = (1 + result.portfolio_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            
            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns.values,
                    name='Drawdown',
                    fill='tonexty',
                    line=dict(color='red'),
                    fillcolor='rgba(255,0,0,0.3)'
                ),
                row=3, col=1
            )
        
        # 6. Performance by Regime
        if result.regime_performance:
            regimes = list(result.regime_performance.keys())
            returns = [result.regime_performance[r].get('annualized_return', 0) for r in regimes]
            
            fig.add_trace(
                go.Bar(
                    x=[f'Regime {r}' for r in regimes],
                    y=returns,
                    name='Returns by Regime',
                    marker_color='lightblue'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Performance Dashboard - {result.strategy_name}",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        # Save if requested
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            self.logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def _create_matplotlib_dashboard(self, 
                                   result: BacktestResult,
                                   save_path: Optional[str] = None):
        """Create matplotlib dashboard."""
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, 0])
        if not result.portfolio_returns.empty:
            cumulative_returns = (1 + result.portfolio_returns).cumprod()
            ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                    label='Strategy', color='blue', linewidth=2)
            
            if result.benchmark_returns is not None:
                benchmark_cumulative = (1 + result.benchmark_returns).cumprod()
                ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                        label='Benchmark', color='gray', linestyle='--', alpha=0.7)
            
            ax1.set_title('Cumulative Returns')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio Weights Evolution
        ax2 = fig.add_subplot(gs[0, 1])
        if not result.portfolio_weights.empty:
            result.portfolio_weights.plot.area(ax=ax2, alpha=0.7)
            ax2.set_title('Portfolio Weights Evolution')
            ax2.set_ylabel('Weight')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Regime Predictions vs Actual
        ax3 = fig.add_subplot(gs[1, 0])
        if not result.predicted_regimes.empty and not result.actual_regimes.empty:
            ax3.plot(result.actual_regimes.index, result.actual_regimes.values, 
                    'o-', label='Actual', color='green', markersize=4)
            ax3.plot(result.predicted_regimes.index, result.predicted_regimes.values, 
                    's--', label='Predicted', color='red', markersize=3, alpha=0.7)
            ax3.set_title('Regime Predictions vs Actual')
            ax3.set_ylabel('Regime')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Rolling Sharpe Ratio
        ax4 = fig.add_subplot(gs[1, 1])
        if not result.portfolio_returns.empty and len(result.portfolio_returns) > 60:
            rolling_sharpe = result.portfolio_returns.rolling(60).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            ax4.plot(rolling_sharpe.index, rolling_sharpe.values, 
                    color='purple', linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title('Rolling Sharpe Ratio (60d)')
            ax4.set_ylabel('Sharpe Ratio')
            ax4.grid(True, alpha=0.3)
        
        # 5. Drawdown Analysis
        ax5 = fig.add_subplot(gs[2, 0])
        if not result.portfolio_returns.empty:
            cumulative = (1 + result.portfolio_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            
            ax5.fill_between(drawdowns.index, drawdowns.values, 0, 
                           color='red', alpha=0.3, label='Drawdown')
            ax5.plot(drawdowns.index, drawdowns.values, color='red', linewidth=1)
            ax5.set_title('Drawdown Analysis')
            ax5.set_ylabel('Drawdown')
            ax5.grid(True, alpha=0.3)
        
        # 6. Performance by Regime
        ax6 = fig.add_subplot(gs[2, 1])
        if result.regime_performance:
            regimes = list(result.regime_performance.keys())
            returns = [result.regime_performance[r].get('annualized_return', 0) for r in regimes]
            
            bars = ax6.bar([f'Regime {r}' for r in regimes], returns, 
                          color='lightblue', alpha=0.7)
            ax6.set_title('Returns by Regime')
            ax6.set_ylabel('Annualized Return')
            ax6.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, returns):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1%}', ha='center', va='bottom')
        
        plt.suptitle(f'Performance Dashboard - {result.strategy_name}', 
                    fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def plot_regime_evolution(self, 
                            result: BacktestResult,
                            save_path: Optional[str] = None):
        """
        Plot regime evolution over time.
        
        Parameters:
        -----------
        result : BacktestResult
            Backtest results
        save_path : str, optional
            Path to save plot
        
        Returns:
        --------
        Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("matplotlib not available for plotting")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Plot regimes
        if not result.actual_regimes.empty:
            ax1.plot(result.actual_regimes.index, result.actual_regimes.values, 
                    'o-', label='Actual Regime', color='green', markersize=4)
        
        if not result.predicted_regimes.empty:
            ax1.plot(result.predicted_regimes.index, result.predicted_regimes.values, 
                    's--', label='Predicted Regime', color='red', markersize=3, alpha=0.8)
        
        ax1.set_ylabel('Regime')
        ax1.set_title('Regime Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot portfolio returns with regime background
        if not result.portfolio_returns.empty:
            ax2.plot(result.portfolio_returns.index, 
                    (1 + result.portfolio_returns).cumprod().values,
                    label='Cumulative Returns', color='blue', linewidth=2)
            
            # Add regime background colors
            if not result.actual_regimes.empty:
                regime_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
                
                for i in range(len(result.actual_regimes)):
                    if i < len(result.actual_regimes) - 1:
                        start_date = result.actual_regimes.index[i]
                        end_date = result.actual_regimes.index[i + 1]
                        regime = result.actual_regimes.iloc[i]
                        
                        color = regime_colors[int(regime) % len(regime_colors)]
                        ax2.axvspan(start_date, end_date, alpha=0.2, color=color)
        
        ax2.set_ylabel('Cumulative Return')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Regime evolution plot saved to {save_path}")
        
        return fig
    
    def create_multi_horizon_heatmap(self, 
                                   results_dict: Dict[str, BacktestResult],
                                   save_path: Optional[str] = None):
        """
        Create multi-horizon performance heatmap.
        
        Parameters:
        -----------
        results_dict : Dict[str, BacktestResult]
            Dictionary of backtest results for different horizons
        save_path : str, optional
            Path to save heatmap
        
        Returns:
        --------
        Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("matplotlib not available for plotting")
            return None
        
        # Create performance matrix
        strategies = list(results_dict.keys())
        
        # Calculate performance metrics for each strategy
        performance_data = []
        for strategy_name, result in results_dict.items():
            if not result.portfolio_returns.empty:
                total_return = (1 + result.portfolio_returns).prod() - 1
                sharpe = result.performance_metrics.get('sharpe_ratio', 0)
                max_dd = result.performance_metrics.get('maximum_drawdown', 0)
                
                performance_data.append({
                    'strategy': strategy_name,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd
                })
        
        if not performance_data:
            self.logger.warning("No performance data available for heatmap")
            return None
        
        df = pd.DataFrame(performance_data)
        df = df.set_index('strategy')
        
        # Create heatmap
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        titles = ['Total Return', 'Sharpe Ratio', 'Maximum Drawdown']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if SEABORN_AVAILABLE:
                sns.heatmap(df[[metric]].T, annot=True, fmt='.3f', 
                           cmap='RdYlGn' if metric != 'max_drawdown' else 'RdYlGn_r',
                           ax=axes[i], cbar=True)
            else:
                im = axes[i].imshow([df[metric].values], aspect='auto', 
                                  cmap='RdYlGn' if metric != 'max_drawdown' else 'RdYlGn_r')
                axes[i].set_xticks(range(len(strategies)))
                axes[i].set_xticklabels(strategies, rotation=45)
                axes[i].set_yticks([0])
                axes[i].set_yticklabels([title])
                
                # Add text annotations
                for j, value in enumerate(df[metric].values):
                    axes[i].text(j, 0, f'{value:.3f}', ha='center', va='center')
            
            axes[i].set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Multi-horizon heatmap saved to {save_path}")
        
        return fig
    
    def plot_weight_evolution(self, 
                            result: BacktestResult,
                            save_path: Optional[str] = None):
        """
        Plot portfolio weight evolution over time.
        
        Parameters:
        -----------
        result : BacktestResult
            Backtest results
        save_path : str, optional
            Path to save plot
        
        Returns:
        --------
        Figure object
        """
        if result.portfolio_weights.empty:
            self.logger.warning("No portfolio weights data available")
            return None
        
        if PLOTLY_AVAILABLE:
            return self._plot_weight_evolution_plotly(result, save_path)
        elif MATPLOTLIB_AVAILABLE:
            return self._plot_weight_evolution_matplotlib(result, save_path)
        else:
            self.logger.error("No plotting libraries available")
            return None
    
    def _plot_weight_evolution_plotly(self, result: BacktestResult, save_path: Optional[str] = None):
        """Plot weight evolution using Plotly."""
        fig = go.Figure()
        
        # Add traces for each asset
        for asset in result.portfolio_weights.columns:
            fig.add_trace(go.Scatter(
                x=result.portfolio_weights.index,
                y=result.portfolio_weights[asset],
                mode='lines',
                stackgroup='weights',
                name=asset,
                hovertemplate=f'{asset}: %{{y:.1%}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Portfolio Weight Evolution',
            xaxis_title='Date',
            yaxis_title='Weight',
            yaxis=dict(tickformat='.0%'),
            hovermode='x unified',
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Weight evolution plot saved to {save_path}")
        
        return fig
    
    def _plot_weight_evolution_matplotlib(self, result: BacktestResult, save_path: Optional[str] = None):
        """Plot weight evolution using matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        result.portfolio_weights.plot.area(ax=ax, alpha=0.7)
        ax.set_title('Portfolio Weight Evolution')
        ax.set_ylabel('Weight')
        ax.set_xlabel('Date')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Weight evolution plot saved to {save_path}")
        
        return fig
    
    def create_regime_performance_comparison(self, 
                                          result: BacktestResult,
                                          save_path: Optional[str] = None):
        """
        Create regime performance comparison chart.
        
        Parameters:
        -----------
        result : BacktestResult
            Backtest results
        save_path : str, optional
            Path to save chart
        
        Returns:
        --------
        Figure object
        """
        if not result.regime_performance:
            self.logger.warning("No regime performance data available")
            return None
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("matplotlib not available for plotting")
            return None
        
        regimes = list(result.regime_performance.keys())
        metrics = ['annualized_return', 'sharpe_ratio', 'maximum_drawdown', 'win_rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [result.regime_performance[r].get(metric, 0) for r in regimes]
            
            bars = axes[i].bar([f'Regime {r}' for r in regimes], values, 
                              alpha=0.7, color=plt.cm.Set3(np.arange(len(regimes))))
            
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if metric in ['annualized_return', 'win_rate']:
                    label = f'{value:.1%}'
                elif metric == 'maximum_drawdown':
                    label = f'{value:.2%}'
                else:
                    label = f'{value:.3f}'
                
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.suptitle('Performance by Regime', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Regime performance comparison saved to {save_path}")
        
        return fig