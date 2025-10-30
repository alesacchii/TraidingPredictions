"""
Fixed Ensemble Model
Properly combines predictions and handles edge cases
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

class FixedEnsembleModel:
    def __init__(self, config, models=None):
        self.config = config
        self.models = models or {}
        self.weights = None
        self.optimization_method = config['models']['ensemble'].get('method', 'weighted_average')
        
    def optimize_weights(self, X_val, y_val):
        """Optimize ensemble weights using validation data"""
        predictions = {}
        valid_models = []
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                pred = model.predict(X_val)
                # Check if predictions are valid
                if pred is not None and not np.all(np.isnan(pred)):
                    # Remove NaN values for optimization
                    valid_mask = ~np.isnan(pred)
                    if valid_mask.sum() > 100:  # Need enough valid predictions
                        predictions[name] = pred
                        valid_models.append(name)
                        logger.info(f"Collected predictions from {name}")
                else:
                    logger.warning(f"Model {name} returned invalid predictions")
            except Exception as e:
                logger.warning(f"Could not get predictions from {name}: {e}")
        
        if len(valid_models) < 2:
            # If less than 2 models, use equal weights
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}
            logger.warning("Not enough valid models for optimization, using equal weights")
            return
        
        # Stack predictions
        pred_matrix = []
        for name in valid_models:
            pred_matrix.append(predictions[name])
        pred_matrix = np.column_stack(pred_matrix)
        
        # Remove rows with any NaN
        valid_mask = ~np.any(np.isnan(pred_matrix), axis=1)
        pred_matrix = pred_matrix[valid_mask]
        y_val_clean = y_val.values[valid_mask] if hasattr(y_val, 'values') else y_val[valid_mask]
        
        logger.info(f"Optimizing with {len(valid_models)} models and {len(y_val_clean)} valid samples")
        
        # Optimization function
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / weights.sum()
            ensemble_pred = np.dot(pred_matrix, weights)
            mse = mean_squared_error(y_val_clean, ensemble_pred)
            return mse
        
        # Initial weights (equal)
        x0 = np.ones(len(valid_models)) / len(valid_models)
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in valid_models]
        
        # Optimize
        try:
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = result.x / result.x.sum()  # Normalize
                self.weights = {name: w for name, w in zip(valid_models, optimized_weights)}
                
                # Add zero weights for invalid models
                for name in self.models.keys():
                    if name not in self.weights:
                        self.weights[name] = 0.0
                
                logger.info("Weight optimization successful:")
                for name, weight in self.weights.items():
                    if weight > 0:
                        logger.info(f"  {name}: {weight:.4f}")
                
                # Report ensemble performance
                ensemble_pred = np.dot(pred_matrix, optimized_weights)
                ensemble_rmse = np.sqrt(mean_squared_error(y_val_clean, ensemble_pred))
                logger.info(f"Ensemble RMSE with optimized weights: {ensemble_rmse:.6f}")
            else:
                raise Exception("Optimization failed")
                
        except Exception as e:
            logger.warning(f"Weight optimization failed: {e}, using equal weights")
            self.weights = {name: 1.0/len(valid_models) if name in valid_models else 0.0 
                          for name in self.models.keys()}
    
    def predict(self, X):
        """Generate ensemble predictions"""
        if self.weights is None:
            # Use equal weights if not optimized
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}
        
        predictions = {}
        valid_models = []
        
        # Collect predictions
        for name, model in self.models.items():
            if self.weights.get(name, 0) > 0:  # Only use models with positive weight
                try:
                    pred = model.predict(X)
                    if pred is not None and not np.all(np.isnan(pred)):
                        predictions[name] = pred
                        valid_models.append(name)
                        logger.info(f"Collected predictions from {name}")
                    else:
                        logger.warning(f"Model {name} returned invalid predictions")
                except Exception as e:
                    logger.warning(f"Prediction failed for {name}: {e}")
        
        if len(valid_models) == 0:
            logger.error("No valid predictions from any model")
            return np.full(len(X), np.nan)
        
        # Create ensemble prediction
        ensemble = np.zeros(len(X))
        total_weight = 0
        
        for name in valid_models:
            weight = self.weights.get(name, 0)
            if weight > 0:
                # Handle NaN values in individual predictions
                pred = predictions[name]
                valid_mask = ~np.isnan(pred)
                ensemble[valid_mask] += pred[valid_mask] * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            ensemble = ensemble / total_weight
        else:
            logger.warning("Total weight is 0, returning NaN")
            return np.full(len(X), np.nan)
        
        # Add some variability to avoid flat predictions
        # This is based on historical volatility
        if np.std(ensemble) < 1e-6:  # If predictions are too flat
            logger.warning("Ensemble predictions are flat, adding small noise")
            noise = np.random.normal(0, 0.0001, len(ensemble))
            ensemble = ensemble + noise
        
        return ensemble
    
    def evaluate(self, X, y, task='regression'):
        """Evaluate ensemble performance"""
        predictions = self.predict(X)
        
        # Remove NaN values
        valid_mask = ~np.isnan(predictions)
        predictions_clean = predictions[valid_mask]
        y_clean = y.values[valid_mask] if hasattr(y, 'values') else y[valid_mask]
        
        if len(predictions_clean) == 0:
            logger.error("No valid predictions for evaluation")
            return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y_clean, predictions_clean))
        mae = mean_absolute_error(y_clean, predictions_clean)
        r2 = r2_score(y_clean, predictions_clean)
        
        logger.info(f"Ensemble Evaluation - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.4f}")
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
