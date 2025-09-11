"""
Performance evaluation and analysis for backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger
import warnings

from ..config import Config
from ..utils.metrics import calculate_metrics, sharpe_ratio, maximum_drawdown
from .backtest_engine import BacktestResult


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation for regime-aware strategies.
    """
    
    def __init__(self, config: Config):
        """
        Initialize performance evaluator.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
    
    def evaluate_strategy(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Comprehensive strategy evaluation.
        
        Parameters:
        -----------
        result : BacktestResult
            Backtest results to evaluate
        
        Returns:
        --------
        Dict[str, Any]
            Comprehensive evaluation report
        """
        self.logger.info("Starting comprehensive strategy evaluation")
        
        evaluation = {
            'strategy_name': result.strategy_name,
            'evaluation_date': datetime.now(),
            'backtest_period': {
                'start_date': result.start_date,
                'end_date': result.end_date,
                'duration_years': self._calculate_duration_years(result.start_date, result.end_date)
            },
            'basic_metrics': self._calculate_basic_metrics(result),
            'risk_metrics': self._calculate_risk_metrics(result),
            'regime_analysis': self._analyze_regime_performance(result),
            'portfolio_characteristics': self._analyze_portfolio_characteristics(result),
            'transaction_analysis': self._analyze_transaction_costs(result),
            'benchmark_comparison': self._compare_to_benchmark(result),
            'drawdown_analysis': self._analyze_drawdowns(result),
            'rolling_performance': self._calculate_rolling_performance(result),
            'regime_transition_analysis': self._analyze_regime_transitions(result)
        }
        
        # Overall assessment
        evaluation['overall_assessment'] = self._generate_overall_assessment(evaluation)
        
        self.logger.info("Strategy evaluation completed")
        
        return evaluation
    
    def _calculate_basic_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        if result.portfolio_returns.empty:
            return {}
        
        metrics = calculate_metrics(
            result.portfolio_returns,
            result.benchmark_returns,
            risk_free_rate=self.config.backtest.risk_free_rate
        )
        
        # Additional metrics
        metrics['total_periods'] = len(result.portfolio_returns)
        metrics['positive_periods'] = (result.portfolio_returns > 0).sum()
        metrics['negative_periods'] = (result.portfolio_returns < 0).sum()
        metrics['win_rate'] = metrics['positive_periods'] / metrics['total_periods'] if metrics['total_periods'] > 0 else 0
        
        return metrics
    
    def _calculate_risk_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        if result.portfolio_returns.empty:
            return {}
        
        returns = result.portfolio_returns
        
        risk_metrics = {
            'volatility_annualized': returns.std() * np.sqrt(252),
            'downside_deviation': returns[returns < 0].std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': returns.quantile(0.05),
            'var_99': returns.quantile(0.01),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
            'cvar_99': returns[returns <= returns.quantile(0.01)].mean(),
        }
        
        # Maximum drawdown analysis
        max_dd, dd_start, dd_end = maximum_drawdown(returns)
        risk_metrics['maximum_drawdown'] = max_dd
        risk_metrics['max_dd_duration'] = (dd_end - dd_start).days if dd_start and dd_end else 0
        
        # Rolling risk metrics
        risk_metrics['volatility_stability'] = returns.rolling(60).std().std() if len(returns) > 60 else np.nan
        
        return risk_metrics
    
    def _analyze_regime_performance(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze performance by regime."""
        regime_analysis = {
            'regime_prediction_accuracy': result.regime_prediction_accuracy,
            'regime_distribution': {},
            'performance_by_regime': {},
            'regime_consistency': {}
        }
        
        # Regime distribution
        if not result.actual_regimes.empty:
            regime_counts = result.actual_regimes.value_counts().sort_index()
            total_periods = len(result.actual_regimes)
            
            for regime, count in regime_counts.items():
                regime_analysis['regime_distribution'][regime] = {
                    'periods': count,
                    'percentage': count / total_periods * 100
                }
        
        # Performance by actual regime
        if not result.actual_regimes.empty and not result.portfolio_returns.empty:
            aligned_regimes, aligned_returns = result.actual_regimes.align(
                result.portfolio_returns, join='inner'
            )
            
            for regime in aligned_regimes.unique():
                regime_mask = aligned_regimes == regime
                regime_returns = aligned_returns[regime_mask]
                
                if not regime_returns.empty:
                    regime_metrics = calculate_metrics(regime_returns)
                    regime_analysis['performance_by_regime'][regime] = regime_metrics
        
        # Regime prediction consistency
        if not result.predicted_regimes.empty and not result.actual_regimes.empty:
            aligned_pred, aligned_actual = result.predicted_regimes.align(
                result.actual_regimes, join='inner'
            )
            
            # Confusion matrix
            confusion_matrix = pd.crosstab(
                aligned_actual, aligned_pred,
                rownames=['Actual'], colnames=['Predicted']
            )
            
            regime_analysis['confusion_matrix'] = confusion_matrix.to_dict()
            
            # Regime-specific prediction accuracy
            for regime in aligned_actual.unique():
                regime_mask = aligned_actual == regime
                regime_predictions = aligned_pred[regime_mask]
                regime_actuals = aligned_actual[regime_mask]
                
                accuracy = (regime_predictions == regime_actuals).mean()
                regime_analysis['regime_consistency'][regime] = accuracy
        
        return regime_analysis
    
    def _analyze_portfolio_characteristics(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze portfolio characteristics over time."""
        if result.portfolio_weights.empty:
            return {}
        
        weights_df = result.portfolio_weights
        
        characteristics = {
            'average_weights': weights_df.mean().to_dict(),
            'weight_volatility': weights_df.std().to_dict(),
            'weight_ranges': {
                asset: {'min': weights_df[asset].min(), 'max': weights_df[asset].max()}
                for asset in weights_df.columns
            },
            'concentration_metrics': {},
            'diversification_metrics': {}
        }
        
        # Concentration metrics
        herfindahl_index = (weights_df ** 2).sum(axis=1)
        characteristics['concentration_metrics'] = {
            'avg_herfindahl_index': herfindahl_index.mean(),
            'max_herfindahl_index': herfindahl_index.max(),
            'avg_effective_assets': (1 / herfindahl_index).mean(),
            'max_single_weight': weights_df.max(axis=1).mean()
        }
        
        # Weight stability
        if len(weights_df) > 1:
            weight_changes = weights_df.diff().abs().sum(axis=1)
            characteristics['weight_stability'] = {
                'avg_total_change': weight_changes.mean(),
                'max_total_change': weight_changes.max(),
                'stability_ratio': 1 - weight_changes.mean()
            }
        
        return characteristics
    
    def _analyze_transaction_costs(self, result: BacktestResult) -> Dict[str, float]:
        """Analyze transaction costs and turnover."""
        analysis = {}
        
        if not result.transaction_costs.empty:
            analysis['total_transaction_costs'] = result.transaction_costs.sum()
            analysis['avg_transaction_cost_per_period'] = result.transaction_costs.mean()
            analysis['transaction_cost_percentage'] = result.transaction_costs.sum() / self.config.backtest.initial_capital * 100
        
        if not result.turnover.empty:
            analysis['avg_turnover'] = result.turnover.mean()
            analysis['max_turnover'] = result.turnover.max()
            analysis['turnover_volatility'] = result.turnover.std()
        
        return analysis
    
    def _compare_to_benchmark(self, result: BacktestResult) -> Dict[str, Any]:
        """Compare strategy performance to benchmarks."""
        comparison = {}
        strategy_metrics = calculate_metrics(result.portfolio_returns)
        comparison['strategy_metrics'] = strategy_metrics
        
        # Compare to external benchmark (e.g., SPY)
        if result.benchmark_returns is not None and not result.benchmark_returns.empty:
            benchmark_metrics = calculate_metrics(result.benchmark_returns)
            excess_returns = result.portfolio_returns - result.benchmark_returns
            outperformance_periods = (result.portfolio_returns > result.benchmark_returns).sum()
            total_periods = len(result.portfolio_returns)
            
            comparison['external_benchmark'] = {
                'available': True,
                'metrics': benchmark_metrics,
                'excess_return_metrics': calculate_metrics(excess_returns),
                'outperformance_rate': outperformance_periods / total_periods if total_periods > 0 else 0,
                'tracking_error': excess_returns.std() * np.sqrt(252)
            }
        else:
            comparison['external_benchmark'] = {'available': False}
        
        # Compare to static MPT benchmark (always available)
        if result.static_mpt_returns is not None and not result.static_mpt_returns.empty:
            static_mpt_metrics = calculate_metrics(result.static_mpt_returns)
            excess_returns_mpt = result.portfolio_returns - result.static_mpt_returns
            outperformance_periods_mpt = (result.portfolio_returns > result.static_mpt_returns).sum()
            total_periods_mpt = len(result.portfolio_returns)
            
            comparison['static_mpt_benchmark'] = {
                'available': True,
                'metrics': static_mpt_metrics,
                'excess_return_metrics': calculate_metrics(excess_returns_mpt),
                'outperformance_rate': outperformance_periods_mpt / total_periods_mpt if total_periods_mpt > 0 else 0,
                'tracking_error': excess_returns_mpt.std() * np.sqrt(252),
                'regime_advantage': {
                    'total_return_diff': strategy_metrics['total_return'] - static_mpt_metrics['total_return'],
                    'sharpe_diff': strategy_metrics['sharpe_ratio'] - static_mpt_metrics['sharpe_ratio'],
                    'max_drawdown_diff': strategy_metrics['maximum_drawdown'] - static_mpt_metrics['maximum_drawdown']
                }
            }
        else:
            comparison['static_mpt_benchmark'] = {'available': False}
        
        # Legacy support for existing code
        benchmark_comparison = comparison.get('external_benchmark', {})
        if benchmark_comparison.get('available', False):
            comparison.update({
                'benchmark_available': True,
                'benchmark_metrics': benchmark_comparison['metrics'],
                'excess_return_metrics': benchmark_comparison['excess_return_metrics'],
                'outperformance_rate': benchmark_comparison['outperformance_rate'],
                'tracking_error': benchmark_comparison['tracking_error']
            })
        else:
            comparison['benchmark_available'] = False
        
        # Information ratio
        if comparison['tracking_error'] > 0:
            comparison['information_ratio'] = excess_returns.mean() * 252 / comparison['tracking_error']
        else:
            comparison['information_ratio'] = 0
        
        return comparison
    
    def _analyze_drawdowns(self, result: BacktestResult) -> Dict[str, Any]:
        """Detailed drawdown analysis."""
        if result.portfolio_returns.empty:
            return {}
        
        returns = result.portfolio_returns
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        
        # Find all drawdown periods
        is_drawdown = drawdowns < 0
        drawdown_periods = []
        
        if is_drawdown.any():
            # Find drawdown starts and ends
            drawdown_starts = is_drawdown & ~is_drawdown.shift(1, fill_value=False)
            drawdown_ends = ~is_drawdown & is_drawdown.shift(1, fill_value=False)
            
            start_dates = drawdowns[drawdown_starts].index
            end_dates = drawdowns[drawdown_ends].index
            
            # Handle case where drawdown continues to end of period
            if len(start_dates) > len(end_dates):
                end_dates = end_dates.append(pd.Index([drawdowns.index[-1]]))
            
            for start, end in zip(start_dates, end_dates):
                period_drawdowns = drawdowns[start:end]
                max_dd = period_drawdowns.min()
                duration = (end - start).days
                
                drawdown_periods.append({
                    'start_date': start,
                    'end_date': end,
                    'max_drawdown': max_dd,
                    'duration_days': duration
                })
        
        analysis = {
            'total_drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration': np.mean([dd['duration_days'] for dd in drawdown_periods]) if drawdown_periods else 0,
            'max_drawdown_duration': max([dd['duration_days'] for dd in drawdown_periods]) if drawdown_periods else 0,
            'avg_drawdown_magnitude': np.mean([dd['max_drawdown'] for dd in drawdown_periods]) if drawdown_periods else 0,
            'drawdown_periods': drawdown_periods[:5]  # Top 5 worst drawdowns
        }
        
        return analysis
    
    def _calculate_rolling_performance(self, result: BacktestResult, window_months: int = 12) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics."""
        if result.portfolio_returns.empty:
            return {}
        
        returns = result.portfolio_returns
        window_periods = window_months * 21  # Approximate monthly periods
        
        if len(returns) < window_periods:
            return {'insufficient_data': True}
        
        rolling_metrics = {
            'rolling_return': returns.rolling(window_periods).apply(lambda x: (1 + x).prod() - 1),
            'rolling_volatility': returns.rolling(window_periods).std() * np.sqrt(252),
            'rolling_sharpe': returns.rolling(window_periods).apply(
                lambda x: sharpe_ratio(x, self.config.backtest.risk_free_rate, 252)
            ),
            'rolling_max_drawdown': returns.rolling(window_periods).apply(
                lambda x: maximum_drawdown(x)[0]
            )
        }
        
        return rolling_metrics
    
    def _analyze_regime_transitions(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze regime transition patterns and their impact."""
        if result.actual_regimes.empty or result.predicted_regimes.empty:
            return {}
        
        analysis = {}
        
        # Transition frequency
        actual_transitions = (result.actual_regimes != result.actual_regimes.shift(1)).sum()
        predicted_transitions = (result.predicted_regimes != result.predicted_regimes.shift(1)).sum()
        
        analysis['transition_frequency'] = {
            'actual_transitions': actual_transitions,
            'predicted_transitions': predicted_transitions,
            'total_periods': len(result.actual_regimes)
        }
        
        # Performance around regime transitions
        if not result.portfolio_returns.empty:
            transition_dates = result.actual_regimes[
                result.actual_regimes != result.actual_regimes.shift(1)
            ].index
            
            pre_transition_performance = []
            post_transition_performance = []
            
            for date in transition_dates:
                try:
                    # Get performance 5 periods before and after transition
                    date_idx = result.portfolio_returns.index.get_loc(date)
                    
                    if date_idx >= 5:
                        pre_perf = result.portfolio_returns.iloc[date_idx-5:date_idx].mean()
                        pre_transition_performance.append(pre_perf)
                    
                    if date_idx < len(result.portfolio_returns) - 5:
                        post_perf = result.portfolio_returns.iloc[date_idx:date_idx+5].mean()
                        post_transition_performance.append(post_perf)
                        
                except (KeyError, IndexError):
                    continue
            
            analysis['transition_performance'] = {
                'avg_pre_transition_return': np.mean(pre_transition_performance) if pre_transition_performance else 0,
                'avg_post_transition_return': np.mean(post_transition_performance) if post_transition_performance else 0,
                'transition_impact': np.mean(post_transition_performance) - np.mean(pre_transition_performance) if pre_transition_performance and post_transition_performance else 0
            }
        
        return analysis
    
    def _generate_overall_assessment(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall strategy assessment."""
        assessment = {
            'rating': 'Unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        basic_metrics = evaluation.get('basic_metrics', {})
        risk_metrics = evaluation.get('risk_metrics', {})
        regime_analysis = evaluation.get('regime_analysis', {})
        benchmark_comparison = evaluation.get('benchmark_comparison', {})
        
        # Rating based on Sharpe ratio and regime prediction accuracy
        sharpe = basic_metrics.get('sharpe_ratio', 0)
        regime_accuracy = regime_analysis.get('regime_prediction_accuracy', 0)
        
        if sharpe > 1.0 and regime_accuracy > 0.6:
            assessment['rating'] = 'Excellent'
        elif sharpe > 0.5 and regime_accuracy > 0.5:
            assessment['rating'] = 'Good'
        elif sharpe > 0.0 and regime_accuracy > 0.4:
            assessment['rating'] = 'Fair'
        else:
            assessment['rating'] = 'Poor'
        
        # Identify strengths
        if sharpe > 0.5:
            assessment['strengths'].append('Strong risk-adjusted returns')
        if regime_accuracy > 0.55:
            assessment['strengths'].append('Good regime prediction accuracy')
        if risk_metrics.get('maximum_drawdown', 0) > -0.2:
            assessment['strengths'].append('Controlled drawdowns')
        if benchmark_comparison.get('outperformance_rate', 0) > 0.6:
            assessment['strengths'].append('Consistent benchmark outperformance')
        
        # Identify weaknesses
        if sharpe < 0.3:
            assessment['weaknesses'].append('Low risk-adjusted returns')
        if regime_accuracy < 0.45:
            assessment['weaknesses'].append('Poor regime prediction accuracy')
        if risk_metrics.get('maximum_drawdown', 0) < -0.3:
            assessment['weaknesses'].append('Large drawdowns')
        if basic_metrics.get('win_rate', 0) < 0.5:
            assessment['weaknesses'].append('Low win rate')
        
        # Generate recommendations
        if regime_accuracy < 0.5:
            assessment['recommendations'].append('Improve regime forecasting models')
        if risk_metrics.get('volatility_annualized', 0) > 0.2:
            assessment['recommendations'].append('Consider risk management improvements')
        if evaluation.get('transaction_analysis', {}).get('avg_turnover', 0) > 1.0:
            assessment['recommendations'].append('Reduce portfolio turnover to minimize costs')
        
        return assessment
    
    def _calculate_duration_years(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate duration in years."""
        if start_date and end_date:
            return (end_date - start_date).days / 365.25
        return 0.0
    
    def compare_strategies(self, 
                          results: Dict[str, BacktestResult]) -> Dict[str, Any]:
        """
        Compare multiple strategies.
        
        Parameters:
        -----------
        results : Dict[str, BacktestResult]
            Dictionary of strategy results
        
        Returns:
        --------
        Dict[str, Any]
            Strategy comparison report
        """
        self.logger.info(f"Comparing {len(results)} strategies")
        
        comparison = {
            'strategies': list(results.keys()),
            'comparison_date': datetime.now(),
            'performance_comparison': {},
            'risk_comparison': {},
            'regime_comparison': {},
            'rankings': {}
        }
        
        # Calculate metrics for each strategy
        strategy_evaluations = {}
        for name, result in results.items():
            strategy_evaluations[name] = self.evaluate_strategy(result)
        
        # Performance comparison
        performance_metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'win_rate']
        for metric in performance_metrics:
            comparison['performance_comparison'][metric] = {
                name: eval_data['basic_metrics'].get(metric, np.nan)
                for name, eval_data in strategy_evaluations.items()
            }
        
        # Risk comparison
        risk_metrics = ['volatility_annualized', 'maximum_drawdown', 'var_95']
        for metric in risk_metrics:
            comparison['risk_comparison'][metric] = {
                name: eval_data['risk_metrics'].get(metric, np.nan)
                for name, eval_data in strategy_evaluations.items()
            }
        
        # Regime comparison
        comparison['regime_comparison'] = {
            name: eval_data['regime_analysis'].get('regime_prediction_accuracy', np.nan)
            for name, eval_data in strategy_evaluations.items()
        }
        
        # Create rankings
        ranking_metrics = ['sharpe_ratio', 'total_return', 'regime_prediction_accuracy']
        for metric in ranking_metrics:
            if metric == 'regime_prediction_accuracy':
                values = comparison['regime_comparison']
            else:
                values = comparison['performance_comparison'].get(metric, {})
            
            # Sort by value (descending for positive metrics)
            sorted_strategies = sorted(
                values.items(), 
                key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, 
                reverse=True
            )
            
            comparison['rankings'][metric] = [name for name, _ in sorted_strategies]
        
        self.logger.info("Strategy comparison completed")
        
        return comparison