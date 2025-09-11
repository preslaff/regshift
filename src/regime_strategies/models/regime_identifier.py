"""
Historical regime identification using unsupervised learning methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available. HMM models will not be supported.")

from ..config import Config


class InvestmentClockIdentifier:
    """
    Investment Clock regime identification based on economic growth and inflation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Investment Clock identifier.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        self.regime_labels = {
            0: "Slowing",     # Low growth, Low inflation
            1: "Heating",     # Low growth, High inflation  
            2: "Growing",     # High growth, Low inflation
            3: "Stagflation"  # High growth, High inflation
        }
    
    def identify_regimes(self, economic_data: pd.DataFrame) -> pd.Series:
        """
        Identify regimes using Investment Clock methodology.
        
        Parameters:
        -----------
        economic_data : pd.DataFrame
            Economic data with CPI_signal and CLI_signal columns
        
        Returns:
        --------
        pd.Series
            Regime labels for each time period
        """
        self.logger.info("Identifying regimes using Investment Clock methodology")
        
        if 'CPI_signal' not in economic_data.columns or 'CLI_signal' not in economic_data.columns:
            raise ValueError("Economic data must contain CPI_signal and CLI_signal columns")
        
        # Get signals
        cpi_signal = economic_data['CPI_signal']
        cli_signal = economic_data['CLI_signal']
        
        # Initialize regime series
        regimes = pd.Series(index=economic_data.index, dtype=int)
        
        # Apply Investment Clock rules
        for i in economic_data.index:
            inflation_high = cpi_signal[i] > 0 if not pd.isna(cpi_signal[i]) else False
            growth_high = cli_signal[i] > 0 if not pd.isna(cli_signal[i]) else False
            
            if not inflation_high and not growth_high:
                regime = 0  # Slowing
            elif inflation_high and not growth_high:
                regime = 1  # Heating
            elif not inflation_high and growth_high:
                regime = 2  # Growing
            else:  # inflation_high and growth_high
                regime = 3  # Stagflation
            
            regimes[i] = regime
        
        # Apply smoothing to reduce regime switching noise
        regimes = self._apply_smoothing(regimes, window=3)
        
        # Log regime distribution
        regime_counts = regimes.value_counts().sort_index()
        for regime_id, count in regime_counts.items():
            regime_name = self.regime_labels.get(regime_id, f"Regime_{regime_id}")
            percentage = (count / len(regimes)) * 100
            self.logger.info(f"{regime_name}: {count} periods ({percentage:.1f}%)")
        
        return regimes
    
    def _apply_smoothing(self, regimes: pd.Series, window: int = 3) -> pd.Series:
        """
        Apply smoothing to reduce regime switching noise.
        
        Parameters:
        -----------
        regimes : pd.Series
            Raw regime series
        window : int
            Smoothing window size
        
        Returns:
        --------
        pd.Series
            Smoothed regime series
        """
        smoothed = regimes.copy()
        
        for i in range(window, len(regimes) - window):
            # Take mode of surrounding window
            window_regimes = regimes.iloc[i-window:i+window+1]
            mode_regime = window_regimes.mode().iloc[0]
            smoothed.iloc[i] = mode_regime
        
        return smoothed


class KMeansRegimeIdentifier:
    """
    K-Means clustering regime identification.
    """
    
    def __init__(self, config: Config):
        """
        Initialize K-Means regime identifier.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        self.model = None
        self.scaler = None
        self.pca = None
        
    def identify_regimes(self, features: pd.DataFrame) -> pd.Series:
        """
        Identify regimes using K-Means clustering.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix for regime identification
        
        Returns:
        --------
        pd.Series
            Regime labels for each time period
        """
        self.logger.info("Identifying regimes using K-Means clustering")
        
        # Prepare features
        X = features.dropna().values
        valid_indices = features.dropna().index
        
        # Create pipeline
        pipeline_steps = [('scaler', StandardScaler())]
        
        # Add PCA if many features
        if X.shape[1] > 10:
            pipeline_steps.append(('pca', PCA(n_components=0.95)))
        
        pipeline_steps.append(('kmeans', KMeans(
            n_clusters=self.config.regime.n_regimes,
            random_state=self.config.regime.kmeans_random_state,
            n_init=self.config.regime.kmeans_n_init
        )))
        
        # Fit pipeline
        self.model = Pipeline(pipeline_steps)
        labels = self.model.fit_predict(X)
        
        # Create regime series
        regimes = pd.Series(index=features.index, dtype=float)
        regimes[valid_indices] = labels
        
        # Forward fill missing values
        regimes = regimes.fillna(method='ffill')
        regimes = regimes.fillna(method='bfill')
        regimes = regimes.astype(int)
        
        # Log regime distribution
        regime_counts = regimes.value_counts().sort_index()
        for regime_id, count in regime_counts.items():
            percentage = (count / len(regimes)) * 100
            self.logger.info(f"Regime {regime_id}: {count} periods ({percentage:.1f}%)")
        
        return regimes


class HMMRegimeIdentifier:
    """
    Hidden Markov Model regime identification.
    """
    
    def __init__(self, config: Config):
        """
        Initialize HMM regime identifier.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required for HMM regime identification")
        
        self.config = config
        self.logger = logger.bind(name=__name__)
        self.model = None
        self.scaler = None
        
    def identify_regimes(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Identify regimes using Hidden Markov Model.
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market return data
        
        Returns:
        --------
        pd.Series
            Regime labels for each time period
        """
        self.logger.info("Identifying regimes using Hidden Markov Model")
        
        # Prepare features (returns and volatilities)
        features_list = []
        
        for column in market_data.columns:
            if not market_data[column].isna().all():
                # Add returns
                features_list.append(market_data[column])
                
                # Add rolling volatility
                rolling_vol = market_data[column].rolling(window=20).std()
                features_list.append(rolling_vol)
        
        X = pd.concat(features_list, axis=1).dropna()
        valid_indices = X.index
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.values)
        
        # Fit HMM model
        self.model = hmm.GaussianHMM(
            n_components=self.config.regime.n_regimes,
            covariance_type="full",
            n_iter=self.config.regime.hmm_n_iter,
            tol=self.config.regime.hmm_tol,
            random_state=self.config.regime.hmm_random_state
        )
        
        self.model.fit(X_scaled)
        
        # Predict regimes
        labels = self.model.predict(X_scaled)
        
        # Create regime series
        regimes = pd.Series(index=market_data.index, dtype=float)
        regimes[valid_indices] = labels
        
        # Forward fill missing values
        regimes = regimes.fillna(method='ffill')
        regimes = regimes.fillna(method='bfill')
        regimes = regimes.astype(int)
        
        # Log regime distribution and transition matrix
        regime_counts = regimes.value_counts().sort_index()
        for regime_id, count in regime_counts.items():
            percentage = (count / len(regimes)) * 100
            self.logger.info(f"Regime {regime_id}: {count} periods ({percentage:.1f}%)")
        
        # Log transition matrix
        self.logger.info("Transition Matrix:")
        transition_matrix = pd.DataFrame(
            self.model.transmat_,
            index=[f"From_{i}" for i in range(self.config.regime.n_regimes)],
            columns=[f"To_{i}" for i in range(self.config.regime.n_regimes)]
        )
        self.logger.info(f"\n{transition_matrix}")
        
        return regimes


class RegimeIdentifier:
    """
    Main regime identification class that orchestrates different methods.
    """
    
    def __init__(self, config: Config):
        """
        Initialize regime identifier.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        
        # Initialize method-specific identifiers
        self.investment_clock = InvestmentClockIdentifier(config)
        self.kmeans_identifier = KMeansRegimeIdentifier(config)
        
        if HMM_AVAILABLE:
            self.hmm_identifier = HMMRegimeIdentifier(config)
        else:
            self.hmm_identifier = None
        
        # Store results
        self.regimes_ = None
        self.method_used_ = None
        
    def fit(self,
            economic_data: Optional[pd.DataFrame] = None,
            market_data: Optional[pd.DataFrame] = None) -> 'RegimeIdentifier':
        """
        Fit regime identification model.
        
        Parameters:
        -----------
        economic_data : pd.DataFrame, optional
            Economic indicator data
        market_data : pd.DataFrame, optional
            Market return data
        
        Returns:
        --------
        RegimeIdentifier
            Fitted regime identifier
        """
        method = self.config.regime.regime_method
        self.method_used_ = method
        
        self.logger.info(f"Fitting regime identifier using {method} method")
        
        if method == "investment_clock":
            if economic_data is None:
                raise ValueError("Economic data required for Investment Clock method")
            self.regimes_ = self.investment_clock.identify_regimes(economic_data)
            
        elif method == "kmeans":
            if market_data is None:
                raise ValueError("Market data required for K-Means method")
            
            # Create feature matrix
            features = self._create_feature_matrix(market_data)
            self.regimes_ = self.kmeans_identifier.identify_regimes(features)
            
        elif method == "hmm":
            if self.hmm_identifier is None:
                raise ImportError("hmmlearn is required for HMM method")
            if market_data is None:
                raise ValueError("Market data required for HMM method")
            self.regimes_ = self.hmm_identifier.identify_regimes(market_data)
            
        else:
            raise ValueError(f"Unknown regime identification method: {method}")
        
        self.logger.info(f"Regime identification completed. Total regimes identified: {len(self.regimes_)}")
        
        return self
    
    def get_regimes(self) -> pd.Series:
        """
        Get identified regimes.
        
        Returns:
        --------
        pd.Series
            Regime labels
        """
        if self.regimes_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return self.regimes_.copy()
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about identified regimes.
        
        Returns:
        --------
        Dict[str, Any]
            Regime statistics
        """
        if self.regimes_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        stats = {
            'method_used': self.method_used_,
            'n_regimes': len(self.regimes_.unique()),
            'total_periods': len(self.regimes_),
            'regime_counts': self.regimes_.value_counts().sort_index().to_dict(),
            'regime_persistence': self._calculate_persistence(),
            'transition_matrix': self._calculate_transition_matrix()
        }
        
        return stats
    
    def _create_feature_matrix(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature matrix for clustering methods.
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market return data
        
        Returns:
        --------
        pd.DataFrame
            Feature matrix
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Add market returns
        features = pd.concat([features, market_data], axis=1)
        
        # Add volatility features
        for column in market_data.columns:
            features[f'{column}_vol_5'] = market_data[column].rolling(5).std()
            features[f'{column}_vol_20'] = market_data[column].rolling(20).std()
            features[f'{column}_vol_60'] = market_data[column].rolling(60).std()
        
        # Add momentum features
        for column in market_data.columns:
            features[f'{column}_mom_5'] = market_data[column].rolling(5).mean()
            features[f'{column}_mom_20'] = market_data[column].rolling(20).mean()
        
        return features
    
    def _calculate_persistence(self) -> float:
        """Calculate average regime persistence."""
        if self.regimes_ is None:
            return 0.0
        
        # Calculate how often regime stays the same
        same_regime = (self.regimes_ == self.regimes_.shift(1)).sum()
        total_transitions = len(self.regimes_) - 1
        
        return same_regime / total_transitions if total_transitions > 0 else 0.0
    
    def _calculate_transition_matrix(self) -> pd.DataFrame:
        """Calculate regime transition matrix."""
        if self.regimes_ is None:
            return pd.DataFrame()
        
        unique_regimes = sorted(self.regimes_.unique())
        n_regimes = len(unique_regimes)
        
        transition_matrix = pd.DataFrame(
            np.zeros((n_regimes, n_regimes)),
            index=unique_regimes,
            columns=unique_regimes
        )
        
        for i in range(len(self.regimes_) - 1):
            from_regime = self.regimes_.iloc[i]
            to_regime = self.regimes_.iloc[i + 1]
            transition_matrix.loc[from_regime, to_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix.div(row_sums, axis=0).fillna(0)
        
        return transition_matrix