"""
Regime forecasting using supervised learning methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. XGBoost models will not be supported.")

from ..config import Config
from ..utils.data_utils import create_feature_matrix, enforce_point_in_time


class NaiveRegimePredictor:
    """
    Naive regime predictor that predicts the next regime will be the same as current.
    This serves as a strong baseline due to regime persistence.
    """
    
    def __init__(self, config: Config):
        """
        Initialize naive predictor.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        self.last_regime_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NaiveRegimePredictor':
        """
        Fit naive predictor (just stores the last regime).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features (not used)
        y : pd.Series
            Target regime labels
        
        Returns:
        --------
        NaiveRegimePredictor
            Fitted predictor
        """
        if not y.empty:
            self.last_regime_ = y.iloc[-1]
        
        self.logger.info("Naive predictor fitted")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict regimes (always returns last known regime).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features (not used)
        
        Returns:
        --------
        np.ndarray
            Predicted regimes
        """
        if self.last_regime_ is None:
            # Return most common regime as fallback
            predictions = np.zeros(len(X))
        else:
            predictions = np.full(len(X), self.last_regime_)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features (not used)
        
        Returns:
        --------
        np.ndarray
            Predicted probabilities
        """
        n_samples = len(X)
        n_regimes = self.config.regime.n_regimes
        
        # Create probability matrix with high confidence for last regime
        proba = np.ones((n_samples, n_regimes)) * 0.1 / (n_regimes - 1)
        
        if self.last_regime_ is not None:
            regime_idx = int(self.last_regime_)
            if 0 <= regime_idx < n_regimes:
                proba[:, regime_idx] = 0.9
        
        return proba


class RegimeForecaster:
    """
    Main regime forecasting class using ensemble of supervised learning models.
    """
    
    def __init__(self, config: Config):
        """
        Initialize regime forecaster.
        
        Parameters:
        -----------
        config : Config
            System configuration
        """
        self.config = config
        self.logger = logger.bind(name=__name__)
        
        # Initialize models
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.naive_predictor = NaiveRegimePredictor(config)
        
        # Model performance tracking
        self.performance_metrics = {}
        self.feature_importance = {}
        
        # Initialize individual models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize individual prediction models."""
        
        # Random Forest
        if "random_forest" in self.config.regime.forecasting_models:
            self.models["random_forest"] = RandomForestClassifier(
                **self.config.regime.random_forest_params
            )
        
        # Logistic Regression
        if "logistic_regression" in self.config.regime.forecasting_models:
            self.models["logistic_regression"] = LogisticRegression(
                **self.config.regime.logistic_regression_params
            )
        
        # XGBoost
        if "xgboost" in self.config.regime.forecasting_models and XGBOOST_AVAILABLE:
            self.models["xgboost"] = xgb.XGBClassifier(
                objective='multi:softprob',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.regime.random_forest_params.get("random_state", 42)
            )
        
        self.logger.info(f"Initialized {len(self.models)} forecasting models")
    
    def fit(self,
            features: pd.DataFrame,
            regimes: pd.Series,
            validation_split: float = 0.2) -> 'RegimeForecaster':
        """
        Fit regime forecasting models.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix for forecasting
        regimes : pd.Series
            Historical regime labels
        validation_split : float
            Fraction of data to use for validation
        
        Returns:
        --------
        RegimeForecaster
            Fitted forecaster
        """
        self.logger.info("Fitting regime forecasting models")
        
        # Prepare data
        X, y = self._prepare_training_data(features, regimes)
        
        if len(X) == 0:
            raise ValueError("No valid training data available")
        
        # Split data for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Fit individual models
        fitted_models = {}
        for name, model in self.models.items():
            try:
                self.logger.info(f"Fitting {name} model")
                model.fit(X_train_scaled, y_train)
                fitted_models[name] = model
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val_scaled)
                val_accuracy = accuracy_score(y_val, val_predictions)
                self.performance_metrics[name] = {
                    'validation_accuracy': val_accuracy,
                    'training_samples': len(X_train)
                }
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
                self.logger.info(f"{name} validation accuracy: {val_accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to fit {name} model: {e}")
        
        self.models = fitted_models
        
        # Fit naive predictor
        self.naive_predictor.fit(pd.DataFrame(X_train), pd.Series(y_train))
        naive_predictions = self.naive_predictor.predict(pd.DataFrame(X_val))
        naive_accuracy = accuracy_score(y_val, naive_predictions)
        self.performance_metrics['naive'] = {'validation_accuracy': naive_accuracy}
        self.logger.info(f"Naive predictor validation accuracy: {naive_accuracy:.3f}")
        
        # Create ensemble model
        if len(fitted_models) > 1:
            self._create_ensemble_model()
            
            # Evaluate ensemble
            if self.ensemble_model:
                ensemble_predictions = self.ensemble_model.predict(X_val_scaled)
                ensemble_accuracy = accuracy_score(y_val, ensemble_predictions)
                self.performance_metrics['ensemble'] = {'validation_accuracy': ensemble_accuracy}
                self.logger.info(f"Ensemble validation accuracy: {ensemble_accuracy:.3f}")
        
        # Cross-validation for more robust evaluation
        self._perform_cross_validation(X, y)
        
        self.logger.info("Regime forecasting models fitted successfully")
        return self
    
    def predict(self,
                features: pd.DataFrame,
                method: str = "ensemble") -> np.ndarray:
        """
        Predict future regimes.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix for prediction
        method : str
            Prediction method ("ensemble", "best", "naive", or specific model name)
        
        Returns:
        --------
        np.ndarray
            Predicted regime labels
        """
        if method == "naive":
            return self.naive_predictor.predict(features)
        
        # Prepare features
        X = self._prepare_prediction_features(features)
        if len(X) == 0:
            self.logger.warning("No valid features for prediction, using naive predictor")
            return self.naive_predictor.predict(features)
        
        X_scaled = self.scaler.transform(X)
        
        if method == "ensemble" and self.ensemble_model:
            return self.ensemble_model.predict(X_scaled)
        elif method == "best":
            best_model_name = self._get_best_model()
            return self.models[best_model_name].predict(X_scaled)
        elif method in self.models:
            return self.models[method].predict(X_scaled)
        else:
            self.logger.warning(f"Unknown prediction method: {method}, using naive predictor")
            return self.naive_predictor.predict(features)
    
    def predict_proba(self,
                     features: pd.DataFrame,
                     method: str = "ensemble") -> np.ndarray:
        """
        Predict regime probabilities.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix for prediction
        method : str
            Prediction method
        
        Returns:
        --------
        np.ndarray
            Predicted probabilities
        """
        if method == "naive":
            return self.naive_predictor.predict_proba(features)
        
        # Prepare features
        X = self._prepare_prediction_features(features)
        if len(X) == 0:
            self.logger.warning("No valid features for prediction, using naive predictor")
            return self.naive_predictor.predict_proba(features)
        
        X_scaled = self.scaler.transform(X)
        
        if method == "ensemble" and self.ensemble_model:
            return self.ensemble_model.predict_proba(X_scaled)
        elif method == "best":
            best_model_name = self._get_best_model()
            return self.models[best_model_name].predict_proba(X_scaled)
        elif method in self.models:
            return self.models[method].predict_proba(X_scaled)
        else:
            self.logger.warning(f"Unknown prediction method: {method}, using naive predictor")
            return self.naive_predictor.predict_proba(features)
    
    def forecast_regime(self,
                       current_date: datetime,
                       features: pd.DataFrame,
                       horizon: int = 1) -> Dict[str, Any]:
        """
        Forecast regime for next period(s) with point-in-time constraints.
        
        Parameters:
        -----------
        current_date : datetime
            Current decision date
        features : pd.DataFrame
            Feature data available at current_date
        horizon : int
            Forecasting horizon (number of periods ahead)
        
        Returns:
        --------
        Dict[str, Any]
            Forecast results with predictions and probabilities
        """
        self.logger.info(f"Forecasting regime for {current_date} (horizon: {horizon})")
        
        # Enforce point-in-time constraints
        available_features = enforce_point_in_time(features, current_date, lag_days=0)
        
        if available_features.empty:
            self.logger.warning("No features available for forecasting")
            return {
                'predicted_regime': 0,
                'probabilities': np.ones(self.config.regime.n_regimes) / self.config.regime.n_regimes,
                'confidence': 0.0,
                'method_used': 'naive',
                'forecast_date': current_date,
                'horizon': horizon
            }
        
        # Get the most recent feature vector
        latest_features = available_features.tail(1)
        
        # Multi-step forecasting for horizon > 1
        results = []
        current_features = latest_features.copy()
        
        for step in range(horizon):
            # Predict next regime
            predictions = self.predict(current_features, method="ensemble")
            probabilities = self.predict_proba(current_features, method="ensemble")
            
            predicted_regime = predictions[0]
            regime_probabilities = probabilities[0]
            confidence = np.max(regime_probabilities)
            
            results.append({
                'predicted_regime': int(predicted_regime),
                'probabilities': regime_probabilities,
                'confidence': float(confidence),
                'step': step + 1
            })
            
            # For multi-step forecasting, update features with prediction
            if step < horizon - 1:
                # Simple approach: assume features remain similar
                # In a more sophisticated approach, you'd model feature evolution
                pass
        
        # Return results for the final horizon
        final_result = results[-1]
        final_result.update({
            'method_used': 'ensemble' if self.ensemble_model else 'best',
            'forecast_date': current_date,
            'horizon': horizon,
            'all_steps': results if horizon > 1 else None
        })
        
        return final_result
    
    def _prepare_training_data(self,
                              features: pd.DataFrame,
                              regimes: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with proper time series structure.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        regimes : pd.Series
            Regime labels
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Features and target arrays
        """
        # Align features and regimes
        aligned_data = features.align(regimes, join='inner', axis=0)
        aligned_features = aligned_data[0]
        aligned_regimes = aligned_data[1]
        
        # Create lagged features (use current features to predict next regime)
        X = aligned_features.iloc[:-1].values  # Features at time t
        y = aligned_regimes.iloc[1:].values    # Regimes at time t+1
        
        # Remove rows with NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def _prepare_prediction_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        
        Returns:
        --------
        np.ndarray
            Prepared feature array
        """
        # Use the most recent complete feature vector
        X = features.dropna().tail(1).values
        
        return X
    
    def _create_ensemble_model(self):
        """Create ensemble model from fitted individual models."""
        if len(self.models) < 2:
            return
        
        try:
            if self.config.regime.ensemble_method == "voting":
                # Use weighted ensemble instead of sklearn VotingClassifier to avoid fitting issues
                self.ensemble_model = WeightedEnsemble(self.models, weights=None)
                
            elif self.config.regime.ensemble_method == "weighted_average":
                # Implement weighted average based on validation performance
                weights = []
                for name in self.models.keys():
                    accuracy = self.performance_metrics.get(name, {}).get('validation_accuracy', 0.5)
                    weights.append(accuracy)
                
                # Normalize weights
                if sum(weights) > 0:
                    weights = [w / sum(weights) for w in weights]
                else:
                    weights = None
                
                self.ensemble_model = WeightedEnsemble(self.models, weights=weights)
            
            self.logger.info(f"Created ensemble model using {self.config.regime.ensemble_method} method")
            
        except Exception as e:
            self.logger.error(f"Failed to create ensemble model: {e}")
            self.ensemble_model = None
    
    def _get_best_model(self) -> str:
        """Get the name of the best performing model."""
        best_model = "naive"
        best_accuracy = self.performance_metrics.get("naive", {}).get("validation_accuracy", 0)
        
        for model_name, metrics in self.performance_metrics.items():
            if model_name != "naive":
                accuracy = metrics.get("validation_accuracy", 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
        
        return best_model
    
    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray):
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name, model in self.models.items():
            try:
                cv_scores = cross_val_score(
                    model, X, y, cv=tscv, scoring='accuracy', n_jobs=1
                )
                
                self.performance_metrics[model_name]['cv_mean'] = cv_scores.mean()
                self.performance_metrics[model_name]['cv_std'] = cv_scores.std()
                
                self.logger.info(f"{model_name} CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
                
            except Exception as e:
                self.logger.error(f"Cross-validation failed for {model_name}: {e}")
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get performance summary of all models.
        
        Returns:
        --------
        pd.DataFrame
            Performance summary
        """
        summary_data = []
        
        for model_name, metrics in self.performance_metrics.items():
            summary_data.append({
                'model': model_name,
                'validation_accuracy': metrics.get('validation_accuracy', np.nan),
                'cv_mean': metrics.get('cv_mean', np.nan),
                'cv_std': metrics.get('cv_std', np.nan)
            })
        
        return pd.DataFrame(summary_data)
    
    def get_feature_importance_summary(self) -> Dict[str, pd.Series]:
        """
        Get feature importance summary for models that support it.
        
        Returns:
        --------
        Dict[str, pd.Series]
            Feature importance for each model
        """
        importance_summary = {}
        
        for model_name, importance in self.feature_importance.items():
            importance_summary[model_name] = pd.Series(importance)
        
        return importance_summary


class WeightedEnsemble:
    """
    Simple weighted ensemble classifier.
    """
    
    def __init__(self, models: Dict[str, Any], weights: Optional[List[float]] = None):
        """
        Initialize weighted ensemble.
        
        Parameters:
        -----------
        models : Dict[str, Any]
            Dictionary of fitted models
        weights : List[float], optional
            Weights for each model
        """
        self.models = models
        self.model_names = list(models.keys())
        
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()  # Normalize
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted ensemble."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using weighted ensemble."""
        weighted_probas = None
        
        for i, (name, model) in enumerate(self.models.items()):
            proba = model.predict_proba(X)
            
            if weighted_probas is None:
                weighted_probas = self.weights[i] * proba
            else:
                weighted_probas += self.weights[i] * proba
        
        return weighted_probas