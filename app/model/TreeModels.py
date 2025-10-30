import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import logging

logger = logging.getLogger('MarketPredictor')


class TreeBasedModels:
    """
    XGBoost and LightGBM models for stock prediction
    """

    def __init__(self, config, model_type='xgboost'):
        self.config = config
        self.model_type = model_type
        self.model = None
        self.feature_importance = None

        if model_type == 'xgboost':
            self.params = config['models']['xgboost']['params']
        elif model_type == 'lightgbm':
            self.params = config['models']['lightgbm']['params']
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None, task='regression'):
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            task: 'regression' or 'classification'
        """
        logger.info(f"Training {self.model_type} model for {task}...")
        logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")

        if self.model_type == 'xgboost':
            return self._train_xgboost(X_train, y_train, X_val, y_val, task)
        elif self.model_type == 'lightgbm':
            return self._train_lightgbm(X_train, y_train, X_val, y_val, task)

    def _train_xgboost(self, X_train, y_train, X_val, y_val, task):
        """Train XGBoost model"""

        params = self.params.copy()

        if task == 'regression':
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
        else:  # classification
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)

        eval_list = [(dtrain, 'train')]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_list.append((dval, 'val'))

        # Train
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=eval_list,
            early_stopping_rounds=20,
            verbose_eval=50
        )

        # Feature importance
        importance_dict = self.model.get_score(importance_type='gain')

        # Create DataFrame with all features (some may have 0 importance)
        feature_names = X_train.columns.tolist()
        importance_values = [importance_dict.get(f'f{i}', 0.0) for i in range(len(feature_names))]

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)

        logger.info(f"XGBoost training complete. Best iteration: {self.model.best_iteration}")

        return self.model

    def _train_lightgbm(self, X_train, y_train, X_val, y_val, task):
        """Train LightGBM model"""

        params = self.params.copy()

        if task == 'regression':
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
        else:  # classification
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('val')

        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=params['n_estimators'],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=50)
            ]
        )

        # Feature importance
        importance_values = self.model.feature_importance(importance_type='gain')

        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns.tolist(),
            'importance': importance_values
        }).sort_values('importance', ascending=False)

        logger.info(f"LightGBM training complete. Best iteration: {self.model.best_iteration}")

        return self.model

    def predict(self, X):
        """Make predictions"""

        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.model_type == 'xgboost':
            dtest = xgb.DMatrix(X)
            predictions = self.model.predict(dtest)
        else:  # lightgbm
            predictions = self.model.predict(X)

        return predictions

    def evaluate(self, X, y, task='regression'):
        """Evaluate model performance"""

        predictions = self.predict(X)

        if task == 'regression':
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)

            metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }

            logger.info(f"Evaluation - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.4f}")

        else:  # classification
            predictions_binary = (predictions > 0.5).astype(int)
            accuracy = accuracy_score(y, predictions_binary)

            # Calculate directional accuracy for trading
            correct_direction = np.sum((predictions > 0.5) == (y > 0))
            directional_accuracy = correct_direction / len(y)

            metrics = {
                'Accuracy': accuracy,
                'Directional_Accuracy': directional_accuracy
            }

            logger.info(f"Evaluation - Accuracy: {accuracy:.4f}, Directional: {directional_accuracy:.4f}")

        return metrics

    def get_feature_importance(self, top_n=20):
        """Get top N most important features"""

        if self.feature_importance is None:
            logger.warning("Feature importance not available")
            return None

        return self.feature_importance.head(top_n)

    def save_model(self, filepath):
        """Save model to disk"""

        if self.model is None:
            raise ValueError("No model to save")

        if self.model_type == 'xgboost':
            self.model.save_model(filepath)
        else:  # lightgbm
            self.model.save_model(filepath)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model from disk"""

        if self.model_type == 'xgboost':
            self.model = xgb.Booster()
            self.model.load_model(filepath)
        else:  # lightgbm
            self.model = lgb.Booster(model_file=filepath)

        logger.info(f"Model loaded from {filepath}")


class TimeSeriesCV:
    """
    Time series cross-validation for walk-forward analysis
    """

    def __init__(self, n_splits=5, test_size=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    def split(self, X, y=None, groups=None):
        """Generate train/test splits"""
        return self.tscv.split(X, y, groups)

    def evaluate_model(self, model, X, y, task='regression'):
        """
        Evaluate model using time series cross-validation

        Returns average metrics across all folds
        """
        logger.info(f"Running {self.n_splits}-fold time series cross-validation...")

        metrics_list = []

        for fold, (train_idx, test_idx) in enumerate(self.split(X), 1):
            logger.info(f"Fold {fold}/{self.n_splits}")

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_test_fold = y.iloc[test_idx]

            # Train on this fold
            model.train(X_train_fold, y_train_fold, task=task)

            # Evaluate
            fold_metrics = model.evaluate(X_test_fold, y_test_fold, task=task)
            fold_metrics['fold'] = fold
            metrics_list.append(fold_metrics)

        # Average metrics
        avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()

        logger.info("Cross-validation complete")
        logger.info(f"Average metrics: {avg_metrics}")

        return metrics_list, avg_metrics
