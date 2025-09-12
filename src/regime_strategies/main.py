"""
Main application for Dynamic Investment Strategies with Market Regimes.
"""

import argparse
import sys
from datetime import datetime, date
from pathlib import Path
from loguru import logger

from .config import Config
from .utils.logger import setup_logger
from .backtesting.backtest_engine import BacktestEngine
from .backtesting.performance_evaluator import PerformanceEvaluator
from .backtesting.visualization import PerformanceVisualizer


class RegimeStrategyApp:
    """
    Main application class for regime-aware investment strategies.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the application.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file
        """
        # Load configuration
        if config_path:
            self.config = Config.from_file(config_path)
        else:
            self.config = Config()
        
        # Setup logging
        setup_logger(
            log_level=self.config.log_level,
            log_file="logs/regime_strategies.log" if self.config.environment == "production" else None
        )
        
        self.logger = logger.bind(name=__name__)
        self.logger.info(f"Initialized Regime Strategy App - {self.config.environment} mode")
        
        # Initialize components
        self.backtest_engine = BacktestEngine(self.config)
        self.evaluator = PerformanceEvaluator(self.config)
        self.visualizer = PerformanceVisualizer(self.config)
    
    def run_backtest(self,
                    start_date: str,
                    end_date: str,
                    assets: list = None,
                    benchmark: str = None,
                    strategy_name: str = "regime_strategy",
                    save_results: bool = True) -> dict:
        """
        Run a complete backtest.
        
        Parameters:
        -----------
        start_date : str
            Backtest start date (YYYY-MM-DD)
        end_date : str
            Backtest end date (YYYY-MM-DD)
        assets : list, optional
            List of assets to include
        benchmark : str, optional
            Benchmark symbol
        strategy_name : str
            Name for the strategy
        save_results : bool
            Whether to save results
        
        Returns:
        --------
        dict
            Backtest results summary
        """
        self.logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        try:
            # Parse dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Use default assets if none provided
            assets = assets or self.config.data.default_assets
            benchmark = benchmark or self.config.backtest.benchmark_symbol
            
            # Run backtest
            result = self.backtest_engine.run_backtest(
                start_date=start_dt,
                end_date=end_dt,
                assets=assets,
                benchmark=benchmark,
                strategy_name=strategy_name
            )
            
            # Evaluate performance
            evaluation = self.evaluator.evaluate_strategy(result)
            
            # Create visualizations
            dashboard = self.visualizer.create_performance_dashboard(
                result,
                save_path=f"results/{strategy_name}_dashboard.html" if save_results else None
            )
            
            # Save results if requested
            if save_results:
                results_path = f"results/{strategy_name}_results.pkl"
                self.backtest_engine.save_results(results_path)
                
                # Save evaluation report
                import json
                eval_path = f"results/{strategy_name}_evaluation.json"
                with open(eval_path, 'w') as f:
                    # Convert numpy types for JSON serialization
                    eval_json = self._convert_for_json(evaluation)
                    json.dump(eval_json, f, indent=2, default=str)
                
                self.logger.info(f"Results saved to {results_path}")
                self.logger.info(f"Evaluation saved to {eval_path}")
            
            # Return summary including benchmark comparison
            benchmark_comparison = evaluation.get('benchmark_comparison', {})
            static_mpt = benchmark_comparison.get('static_mpt_benchmark', {})
            
            summary = {
                'strategy_name': strategy_name,
                'backtest_period': f"{start_date} to {end_date}",
                'total_return': evaluation['basic_metrics'].get('total_return', 0),
                'annualized_return': evaluation['basic_metrics'].get('annualized_return', 0),
                'annualized_volatility': evaluation['basic_metrics'].get('annualized_volatility', 0),
                'sharpe_ratio': evaluation['basic_metrics'].get('sharpe_ratio', 0),
                'maximum_drawdown': evaluation['risk_metrics'].get('maximum_drawdown', 0),
                'regime_prediction_accuracy': evaluation['regime_analysis'].get('regime_prediction_accuracy', 0),
                'overall_rating': evaluation['overall_assessment'].get('rating', 'Unknown'),
                # Static MPT benchmark comparison
                'static_mpt_outperformance': static_mpt.get('outperformance_rate', 0),
                'regime_advantage_return': static_mpt.get('regime_advantage', {}).get('total_return_diff', 0),
                'regime_advantage_sharpe': static_mpt.get('regime_advantage', {}).get('sharpe_diff', 0)
            }
            
            self.logger.info(f"Backtest completed successfully: {summary}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    def compare_strategies(self,
                         strategy_configs: list,
                         save_results: bool = True) -> dict:
        """
        Compare multiple strategy configurations.
        
        Parameters:
        -----------
        strategy_configs : list
            List of strategy configuration dictionaries
        save_results : bool
            Whether to save comparison results
        
        Returns:
        --------
        dict
            Strategy comparison results
        """
        self.logger.info(f"Comparing {len(strategy_configs)} strategies")
        
        results = {}
        
        for config in strategy_configs:
            strategy_name = config.get('name', 'unnamed_strategy')
            
            try:
                # Run individual backtest
                result = self.run_backtest(
                    start_date=config['start_date'],
                    end_date=config['end_date'],
                    assets=config.get('assets'),
                    benchmark=config.get('benchmark'),
                    strategy_name=strategy_name,
                    save_results=False  # Save at end of comparison
                )
                
                results[strategy_name] = self.backtest_engine.get_results()
                
            except Exception as e:
                self.logger.error(f"Failed to run strategy {strategy_name}: {e}")
                continue
        
        if not results:
            raise ValueError("No successful strategy runs for comparison")
        
        # Compare strategies
        comparison = self.evaluator.compare_strategies(results)
        
        # Create comparison visualizations
        if len(results) > 1:
            heatmap = self.visualizer.create_multi_horizon_heatmap(
                results,
                save_path="results/strategy_comparison_heatmap.png" if save_results else None
            )
        
        # Save comparison results
        if save_results:
            import json
            comparison_path = "results/strategy_comparison.json"
            with open(comparison_path, 'w') as f:
                comparison_json = self._convert_for_json(comparison)
                json.dump(comparison_json, f, indent=2, default=str)
            
            self.logger.info(f"Strategy comparison saved to {comparison_path}")
        
        return comparison
    
    def run_live_simulation(self,
                          start_date: str,
                          end_date: str,
                          rebalance_frequency: str = "monthly",
                          assets: list = None) -> dict:
        """
        Run live simulation with regime updates.
        
        Parameters:
        -----------
        start_date : str
            Simulation start date
        end_date : str
            Simulation end date
        rebalance_frequency : str
            Rebalancing frequency
        assets : list, optional
            Asset universe
        
        Returns:
        --------
        dict
            Simulation results
        """
        self.logger.info(f"Starting live simulation: {start_date} to {end_date}")
        
        # For now, this is similar to backtest but could be extended
        # for real-time data feeds and live trading
        
        # Update configuration for live simulation
        original_freq = self.config.backtest.rebalance_frequency
        self.config.backtest.rebalance_frequency = rebalance_frequency
        
        try:
            result = self.run_backtest(
                start_date=start_date,
                end_date=end_date,
                assets=assets,
                strategy_name="live_simulation",
                save_results=True
            )
            
            return result
            
        finally:
            # Restore original frequency
            self.config.backtest.rebalance_frequency = original_freq
    
    def _convert_for_json(self, obj):
        """Convert numpy types and other non-serializable objects for JSON."""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return self._convert_for_json(obj.__dict__)
        else:
            return obj


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Dynamic Investment Strategies with Market Regimes")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--assets', nargs='+', help='Asset symbols')
    backtest_parser.add_argument('--benchmark', help='Benchmark symbol')
    backtest_parser.add_argument('--strategy-name', default='regime_strategy', help='Strategy name')
    backtest_parser.add_argument('--config', help='Configuration file path')
    backtest_parser.add_argument('--no-save', action='store_true', help='Don\'t save results')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare strategies')
    compare_parser.add_argument('--config-file', required=True, help='Strategy comparison config file')
    compare_parser.add_argument('--no-save', action='store_true', help='Don\'t save results')
    
    # Live simulation command
    live_parser = subparsers.add_parser('live', help='Run live simulation')
    live_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    live_parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    live_parser.add_argument('--frequency', default='monthly', help='Rebalancing frequency')
    live_parser.add_argument('--assets', nargs='+', help='Asset symbols')
    live_parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize app
        app = RegimeStrategyApp(args.config if hasattr(args, 'config') else None)
        
        if args.command == 'backtest':
            result = app.run_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                assets=args.assets,
                benchmark=args.benchmark,
                strategy_name=args.strategy_name,
                save_results=not args.no_save
            )
            
            print("\nBacktest Results:")
            print(f"Strategy: {result['strategy_name']}")
            print(f"Period: {result['backtest_period']}")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Annualized Return: {result['annualized_return']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(f"Maximum Drawdown: {result['maximum_drawdown']:.2%}")
            print(f"Regime Accuracy: {result['regime_prediction_accuracy']:.2%}")
            print(f"Overall Rating: {result['overall_rating']}")
        
        elif args.command == 'compare':
            import json
            with open(args.config_file, 'r') as f:
                strategy_configs = json.load(f)
            
            comparison = app.compare_strategies(
                strategy_configs,
                save_results=not args.no_save
            )
            
            print(f"\nStrategy Comparison Results ({len(comparison['strategies'])} strategies):")
            
            # Print rankings
            for metric, ranking in comparison['rankings'].items():
                print(f"\n{metric.replace('_', ' ').title()} Ranking:")
                for i, strategy in enumerate(ranking, 1):
                    value = comparison['performance_comparison'].get(metric, {}).get(strategy, 'N/A')
                    if isinstance(value, float):
                        print(f"  {i}. {strategy}: {value:.3f}")
                    else:
                        print(f"  {i}. {strategy}: {value}")
        
        elif args.command == 'live':
            result = app.run_live_simulation(
                start_date=args.start_date,
                end_date=args.end_date,
                rebalance_frequency=args.frequency,
                assets=args.assets
            )
            
            print(f"\nLive Simulation Results:")
            print(f"Period: {result['backtest_period']}")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(f"Regime Accuracy: {result['regime_prediction_accuracy']:.2%}")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()