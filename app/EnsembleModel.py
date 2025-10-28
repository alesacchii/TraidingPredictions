import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
from configuration.Logger_config import setup_logger, logger


class EnsembleModel:
    """
    Meta-ensemble that combines predictions from multiple models
    Uses optimized weights to maximize performance
    """
    
    def __init__(self, config, models_dict=None):
        """
        Args:
            config: Configuration dictionary
            models_dict: Dictionary of trained models {name: model_object}
        """
        self.config = config
        self.models = models_dict or {}
        self.weights = None
        self.ensemble_config = config['models']['ensemble']
        self.method = self.ensemble_config['method']
        
    def add_model(self, name, model):
        """Add a model to the ensemble"""
        self.models[name] = model
        logger.info(f"Added {name} to ensemble")
    
    def collect_predictions(self, X, models_to_use=None):
        """
        Collect predictions from all models
        
        Args:
            X: Input features
            models_to_use: List of model names to use (None = all)
            
        Returns:
            DataFrame with predictions from each model
        """
        if models_to_use is None:
            models_to_use = list(self.models.keys())
        
        predictions = {}
        
        for name in models_to_use:
            if name not in self.models:
                logger.warning(f"Model {name} not found in ensemble")
                continue
            
            try:
                model = self.models[name]
                pred = model.predict(X)
                
                # Handle NaN values
                if pred is None or (isinstance(pred, np.ndarray) and np.all(np.isnan(pred))):
                    logger.warning(f"Model {name} returned invalid predictions")
                    continue
                
                predictions[name] = pred
                logger.info(f"Collected predictions from {name}")
                
            except Exception as e:
                logger.error(f"Error collecting predictions from {name}: {e}")
                continue
        
        if not predictions:
            logger.error("No valid predictions collected from any model")
            return None
        
        return pd.DataFrame(predictions)
    
    def optimize_weights(self, X_val, y_val, models_to_use=None):
        """
        Optimize ensemble weights on validation set
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            models_to_use: List of model names to use
            
        Returns:
            Optimized weights dictionary
        """
        logger.info("Optimizing ensemble weights...")
        
        # Collect predictions
        predictions_df = self.collect_predictions(X_val, models_to_use)
        
        if predictions_df is None or predictions_df.empty:
            logger.error("Cannot optimize weights - no predictions available")
            return None
        
        # Remove rows with any NaN
        valid_mask = ~predictions_df.isna().any(axis=1)
        predictions_clean = predictions_df[valid_mask].values
        y_clean = y_val[valid_mask].values if hasattr(y_val, 'values') else y_val[valid_mask]
        
        if len(predictions_clean) == 0:
            logger.error("No valid predictions for optimization")
            return None
        
        n_models = predictions_clean.shape[1]
        
        # Optimization objective: minimize RMSE
        def objective(weights):
            ensemble_pred = np.dot(predictions_clean, weights)
            rmse = np.sqrt(mean_squared_error(y_clean, ensemble_pred))
            return rmse
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_models) / n_models
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            self.weights = dict(zip(predictions_df.columns, optimal_weights))
            
            logger.info("Weight optimization successful:")
            for name, weight in self.weights.items():
                logger.info(f"  {name}: {weight:.4f}")
            
            # Evaluate with optimal weights
            ensemble_pred = np.dot(predictions_clean, optimal_weights)
            rmse = np.sqrt(mean_squared_error(y_clean, ensemble_pred))
            logger.info(f"Ensemble RMSE with optimized weights: {rmse:.6f}")
            
            return self.weights
        else:
            logger.warning("Weight optimization failed, using equal weights")
            self.weights = dict(zip(predictions_df.columns, initial_weights))
            return self.weights
    
    def predict(self, X, use_optimized_weights=True):
        """
        Make ensemble predictions
        
        Args:
            X: Input features
            use_optimized_weights: Use optimized weights if available
            
        Returns:
            Ensemble predictions
        """
        # Collect predictions from all models
        predictions_df = self.collect_predictions(X)
        
        if predictions_df is None or predictions_df.empty:
            logger.error("Cannot make predictions - no model outputs available")
            return None
        
        # Determine weights
        if use_optimized_weights and self.weights is not None:
            # Use optimized weights
            weights_array = np.array([
                self.weights.get(col, 1.0 / len(predictions_df.columns))
                for col in predictions_df.columns
            ])
            weights_array = weights_array / weights_array.sum()  # Normalize
        else:
            # Equal weights
            weights_array = np.ones(len(predictions_df.columns)) / len(predictions_df.columns)
        
        # Handle NaN values
        # For each row, compute weighted average of non-NaN predictions
        ensemble_predictions = []
        
        for idx in range(len(predictions_df)):
            row = predictions_df.iloc[idx].values
            mask = ~np.isnan(row)
            
            if mask.sum() == 0:
                # All predictions are NaN
                ensemble_predictions.append(np.nan)
            else:
                # Weighted average of available predictions
                available_weights = weights_array[mask]
                available_weights = available_weights / available_weights.sum()
                pred = np.dot(row[mask], available_weights)
                ensemble_predictions.append(pred)
        
        return np.array(ensemble_predictions)
    
    def predict_with_confidence(self, X, use_optimized_weights=True):
        """
        Make predictions with confidence intervals
        
        Returns:
            predictions: Point predictions
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
            std: Standard deviation of model predictions
        """
        predictions_df = self.collect_predictions(X)
        
        if predictions_df is None or predictions_df.empty:
            return None, None, None, None
        
        # Point prediction
        ensemble_pred = self.predict(X, use_optimized_weights)
        
        # Confidence from prediction variance
        pred_std = predictions_df.std(axis=1).values
        pred_mean = predictions_df.mean(axis=1).values
        
        # 95% confidence interval (assuming normal distribution)
        confidence_level = self.config.get('prediction', {}).get('confidence_level', 0.95)
        z_score = 1.96  # for 95% CI
        
        lower_bound = pred_mean - z_score * pred_std
        upper_bound = pred_mean + z_score * pred_std
        
        return ensemble_pred, lower_bound, upper_bound, pred_std
    
    def evaluate(self, X, y, task='regression', use_optimized_weights=True):
        """
        Evaluate ensemble performance
        """
        predictions = self.predict(X, use_optimized_weights)
        
        if predictions is None:
            logger.error("Cannot evaluate - predictions failed")
            return {}
        
        # Remove NaN values
        mask = ~np.isnan(predictions)
        predictions_clean = predictions[mask]
        y_clean = y[mask].values if hasattr(y, 'values') else y[mask]
        
        if len(predictions_clean) == 0:
            logger.warning("No valid predictions for evaluation")
            return {}
        
        if task == 'regression':
            mse = mean_squared_error(y_clean, predictions_clean)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_clean, predictions_clean)
            r2 = r2_score(y_clean, predictions_clean)
            
            metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            logger.info(f"Ensemble Evaluation - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.4f}")
            
        else:  # classification
            from sklearn.metrics import accuracy_score
            
            predictions_binary = (predictions_clean > 0.5).astype(int)
            accuracy = accuracy_score(y_clean, predictions_binary)
            
            correct_direction = np.sum((predictions_clean > 0.5) == (y_clean > 0))
            directional_accuracy = correct_direction / len(y_clean)
            
            metrics = {
                'Accuracy': accuracy,
                'Directional_Accuracy': directional_accuracy
            }
            
            logger.info(f"Ensemble Evaluation - Accuracy: {accuracy:.4f}, Directional: {directional_accuracy:.4f}")
        
        return metrics
    
    def get_model_contributions(self, X):
        """
        Analyze how much each model contributes to final predictions
        """
        predictions_df = self.collect_predictions(X)
        
        if predictions_df is None or self.weights is None:
            return None
        
        contributions = {}
        
        for model_name in predictions_df.columns:
            if model_name in self.weights:
                weight = self.weights[model_name]
                pred_contribution = predictions_df[model_name] * weight
                contributions[model_name] = {
                    'weight': weight,
                    'mean_prediction': predictions_df[model_name].mean(),
                    'mean_contribution': pred_contribution.mean()
                }
        
        return pd.DataFrame(contributions).T
    
    def compare_models(self, X, y):
        """
        Compare individual model performances
        """
        predictions_df = self.collect_predictions(X)
        
        if predictions_df is None:
            return None
        
        comparison = {}
        
        # Remove NaN for fair comparison
        valid_mask = ~predictions_df.isna().any(axis=1)
        y_clean = y[valid_mask].values if hasattr(y, 'values') else y[valid_mask]
        
        for model_name in predictions_df.columns:
            pred = predictions_df[model_name][valid_mask].values
            
            rmse = np.sqrt(mean_squared_error(y_clean, pred))
            mae = mean_absolute_error(y_clean, pred)
            r2 = r2_score(y_clean, pred)
            
            comparison[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
        
        # Add ensemble
        ensemble_pred = self.predict(X[valid_mask])
        if ensemble_pred is not None:
            valid_ensemble = ~np.isnan(ensemble_pred)
            if valid_ensemble.sum() > 0:
                rmse = np.sqrt(mean_squared_error(y_clean[valid_ensemble], ensemble_pred[valid_ensemble]))
                mae = mean_absolute_error(y_clean[valid_ensemble], ensemble_pred[valid_ensemble])
                r2 = r2_score(y_clean[valid_ensemble], ensemble_pred[valid_ensemble])
                
                comparison['Ensemble'] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                }
        
        df_comparison = pd.DataFrame(comparison).T
        df_comparison = df_comparison.sort_values('RMSE')
        
        logger.info("\nModel Comparison:")
        logger.info(f"\n{df_comparison}")
        
        return df_comparison
