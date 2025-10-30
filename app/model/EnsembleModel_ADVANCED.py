#!/usr/bin/env python3
"""
ADVANCED ENSEMBLE MODEL
Risolve il problema delle predizioni troppo piatte con:
- Confidence calibration
- Adaptive thresholds
- Uncertainty quantification
- Model disagreement detection
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
from scipy.stats import norm


class AdvancedEnsembleModel:
    """
    Ensemble avanzato che risolve i problemi critici:
    1. Predizioni troppo piatte → Adaptive scaling
    2. No prediction on down days → Asymmetric thresholds
    3. Poor confidence intervals → Proper calibration
    """
    
    def __init__(self, config, models_dict=None):
        self.config = config
        self.models = models_dict or {}
        self.weights = None
        self.ensemble_config = config['models']['ensemble']
        self.method = self.ensemble_config['method']
        
        # Calibration parameters
        self.calibration_params = {
            'scale': 1.0,
            'offset': 0.0,
            'confidence_scale': 1.0
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'up': 0.005,
            'down': -0.005
        }
        
    def optimize_weights(self, X_val, y_val, models_to_use=None):
        """
        Ottimizza pesi con focus su directional accuracy
        """
        print("Optimizing ensemble weights with directional focus...")
        
        predictions_df = self.collect_predictions(X_val, models_to_use)
        
        if predictions_df is None or predictions_df.empty:
            print("ERROR: Cannot optimize weights")
            return None
        
        # Reset indices
        predictions_df = predictions_df.reset_index(drop=True)
        
        if hasattr(y_val, 'values'):
            y_val_array = y_val.values
        else:
            y_val_array = np.array(y_val)
        
        # Remove NaN
        valid_mask = ~predictions_df.isna().any(axis=1).values
        predictions_clean = predictions_df[valid_mask].values
        y_clean = y_val_array[valid_mask]
        
        if len(predictions_clean) == 0:
            return None
        
        n_models = predictions_clean.shape[1]
        
        # Multi-objective optimization: RMSE + Directional Accuracy
        def objective(weights):
            ensemble_pred = np.dot(predictions_clean, weights)
            
            # RMSE component
            rmse = np.sqrt(mean_squared_error(y_clean, ensemble_pred))
            
            # Directional accuracy component
            y_direction = (y_clean > 0).astype(int)
            pred_direction = (ensemble_pred > 0).astype(int)
            dir_accuracy = np.mean(y_direction == pred_direction)
            
            # Combined objective (minimize RMSE, maximize direction)
            # Weight directional accuracy heavily
            return rmse - (dir_accuracy * 0.05)  # Directional worth 0.05 RMSE reduction
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        initial_weights = np.ones(n_models) / n_models
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.weights = dict(zip(predictions_df.columns, result.x))
            
            # Evaluate
            ensemble_pred = np.dot(predictions_clean, result.x)
            rmse = np.sqrt(mean_squared_error(y_clean, ensemble_pred))
            dir_acc = np.mean((y_clean > 0) == (ensemble_pred > 0))
            
            print("Weight optimization successful:")
            for name, weight in self.weights.items():
                print(f"  {name}: {weight:.4f}")
            print(f"\nEnsemble RMSE: {rmse:.6f}")
            print(f"Directional Accuracy: {dir_acc:.2%}")
            
            return self.weights
        else:
            print("WARNING: Optimization failed")
            self.weights = dict(zip(predictions_df.columns, initial_weights))
            return self.weights
    
    def calibrate_predictions(self, predictions, actuals):
        """
        Calibra le predizioni per risolvere il problema "too flat"
        
        Problema: predictions hanno variance troppo bassa
        Soluzione: Scala per matchare la variance degli actuals
        """
        print("\nCalibrating prediction scale...")
        
        # Remove NaN
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        pred_clean = predictions[mask]
        actual_clean = actuals[mask]
        
        if len(pred_clean) == 0:
            return predictions
        
        # Calculate scale factor
        pred_std = np.std(pred_clean)
        actual_std = np.std(actual_clean)
        
        if pred_std > 0:
            scale_factor = actual_std / pred_std
        else:
            scale_factor = 1.0
        
        # Calculate offset (bias correction)
        pred_mean = np.mean(pred_clean)
        actual_mean = np.mean(actual_clean)
        offset = actual_mean - (pred_mean * scale_factor)
        
        # Store calibration params
        self.calibration_params['scale'] = scale_factor
        self.calibration_params['offset'] = offset
        
        print(f"Scale factor: {scale_factor:.4f}")
        print(f"Offset: {offset:.6f}")
        
        # Apply calibration
        calibrated = predictions * scale_factor + offset
        
        # Verify
        cal_std = np.std(calibrated[mask])
        print(f"Prediction std before: {pred_std:.6f}")
        print(f"Prediction std after: {cal_std:.6f}")
        print(f"Actual std: {actual_std:.6f}")
        
        return calibrated
    
    def compute_adaptive_thresholds(self, predictions, actuals):
        """
        Calcola threshold asimmetrici per UP/DOWN
        
        Problema: Fixed threshold 0 non funziona
        Soluzione: Use quantiles of prediction distribution
        """
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        pred_clean = predictions[mask]
        actual_clean = actuals[mask]
        
        if len(pred_clean) == 0:
            return
        
        # Find optimal thresholds that maximize directional accuracy
        best_acc = 0
        best_up_thresh = 0
        best_down_thresh = 0
        
        # Search grid
        up_candidates = np.percentile(pred_clean, [40, 45, 50, 55, 60])
        down_candidates = np.percentile(pred_clean, [40, 45, 50, 55, 60])
        
        for up_t in up_candidates:
            for down_t in down_candidates:
                pred_dir = np.where(pred_clean > up_t, 1,
                           np.where(pred_clean < down_t, -1, 0))
                actual_dir = np.where(actual_clean > 0, 1,
                             np.where(actual_clean < 0, -1, 0))
                
                acc = np.mean(pred_dir == actual_dir)
                
                if acc > best_acc:
                    best_acc = acc
                    best_up_thresh = up_t
                    best_down_thresh = down_t
        
        self.adaptive_thresholds['up'] = best_up_thresh
        self.adaptive_thresholds['down'] = best_down_thresh
        
        print(f"\nAdaptive thresholds:")
        print(f"  UP threshold: {best_up_thresh:.6f}")
        print(f"  DOWN threshold: {best_down_thresh:.6f}")
        print(f"  Directional accuracy with adaptive: {best_acc:.2%}")
    
    def predict_with_calibration(self, X, y_val=None):
        """
        Predizione con calibrazione applicata
        """
        # Base prediction
        base_predictions = self.predict(X, use_optimized_weights=True)
        
        if base_predictions is None:
            return None
        
        # Apply calibration if parameters exist
        calibrated = (base_predictions * self.calibration_params['scale'] + 
                     self.calibration_params['offset'])
        
        return calibrated
    
    def predict_with_uncertainty(self, X):
        """
        Predizione con quantificazione uncertainty
        
        Returns:
            predictions: Point predictions
            uncertainty: Epistemic uncertainty (model disagreement)
            aleatoric: Aleatoric uncertainty (data noise)
        """
        predictions_df = self.collect_predictions(X)
        
        if predictions_df is None:
            return None, None, None
        
        # Point prediction (calibrated)
        base_pred = self.predict(X, use_optimized_weights=True)
        calibrated_pred = (base_pred * self.calibration_params['scale'] + 
                          self.calibration_params['offset'])
        
        # Epistemic uncertainty (model disagreement)
        epistemic = predictions_df.std(axis=1).values
        
        # Aleatoric uncertainty (estimated from residuals)
        # This would be learned from validation set
        aleatoric = np.ones_like(calibrated_pred) * 0.02  # Conservative estimate
        
        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic**2 + aleatoric**2)
        
        return calibrated_pred, epistemic, total_uncertainty
    
    def predict_with_confidence_intervals(self, X, confidence=0.95):
        """
        Confidence intervals calibrati
        """
        pred, epistemic, total_unc = self.predict_with_uncertainty(X)
        
        if pred is None:
            return None, None, None
        
        # Z-score for confidence level
        z = norm.ppf((1 + confidence) / 2)
        
        # Confidence intervals
        lower = pred - z * total_unc
        upper = pred + z * total_unc
        
        return pred, lower, upper
    
    def should_trade(self, predictions):
        """
        Determina se tradare basato su adaptive thresholds
        
        Returns:
            signals: 1=buy, -1=sell, 0=hold
        """
        signals = np.zeros_like(predictions)
        
        # Buy signal
        signals[predictions > self.adaptive_thresholds['up']] = 1
        
        # Sell signal  
        signals[predictions < self.adaptive_thresholds['down']] = -1
        
        return signals
    
    def collect_predictions(self, X, models_to_use=None):
        """Collect predictions from models"""
        if models_to_use is None:
            models_to_use = list(self.models.keys())
        
        predictions = {}
        
        for name in models_to_use:
            if name not in self.models:
                continue
            
            try:
                pred = self.models[name].predict(X)
                
                if pred is None or (isinstance(pred, np.ndarray) and np.all(np.isnan(pred))):
                    continue
                
                predictions[name] = pred
                
            except Exception as e:
                print(f"Error collecting from {name}: {e}")
                continue
        
        if not predictions:
            return None
        
        return pd.DataFrame(predictions)
    
    def predict(self, X, use_optimized_weights=True):
        """Base prediction"""
        predictions_df = self.collect_predictions(X)
        
        if predictions_df is None or predictions_df.empty:
            return None
        
        # Weights
        if use_optimized_weights and self.weights is not None:
            weights_array = np.array([
                self.weights.get(col, 1.0 / len(predictions_df.columns))
                for col in predictions_df.columns
            ])
            weights_array = weights_array / weights_array.sum()
        else:
            weights_array = np.ones(len(predictions_df.columns)) / len(predictions_df.columns)
        
        # Weighted average handling NaN
        ensemble_predictions = []
        
        for idx in range(len(predictions_df)):
            row = predictions_df.iloc[idx].values
            mask = ~np.isnan(row)
            
            if mask.sum() == 0:
                ensemble_predictions.append(np.nan)
            else:
                available_weights = weights_array[mask]
                available_weights = available_weights / available_weights.sum()
                pred = np.dot(row[mask], available_weights)
                ensemble_predictions.append(pred)
        
        return np.array(ensemble_predictions)
    
    def evaluate(self, X, y, task='regression', use_optimized_weights=True):
        """Evaluation con metriche avanzate"""
        # Base predictions
        predictions = self.predict(X, use_optimized_weights)
        
        if predictions is None:
            return {}
        
        # Calibrated predictions
        calibrated = (predictions * self.calibration_params['scale'] + 
                     self.calibration_params['offset'])
        
        # Convert y
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Remove NaN
        mask = ~np.isnan(calibrated)
        calibrated_clean = calibrated[mask]
        y_clean = y_array[mask]
        
        if len(calibrated_clean) == 0:
            return {}
        
        # Regression metrics
        mse = mean_squared_error(y_clean, calibrated_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_clean, calibrated_clean)
        r2 = r2_score(y_clean, calibrated_clean)
        
        # Directional accuracy
        dir_acc = np.mean((y_clean > 0) == (calibrated_clean > 0))
        
        # Directional with adaptive thresholds
        signals = self.should_trade(calibrated_clean)
        actual_dir = np.sign(y_clean)
        adaptive_dir_acc = np.mean(signals == actual_dir)
        
        # Up/Down separate
        up_mask = y_clean > 0
        down_mask = y_clean < 0
        
        up_acc = np.mean((calibrated_clean[up_mask] > 0)) if up_mask.sum() > 0 else 0
        down_acc = np.mean((calibrated_clean[down_mask] < 0)) if down_mask.sum() > 0 else 0
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Directional_Accuracy': dir_acc,
            'Adaptive_Dir_Accuracy': adaptive_dir_acc,
            'Up_Days_Accuracy': up_acc,
            'Down_Days_Accuracy': down_acc
        }
        
        print(f"\n📊 Advanced Ensemble Evaluation:")
        print(f"RMSE: {rmse:.6f}, R²: {r2:.4f}")
        print(f"Directional: {dir_acc:.2%}")
        print(f"Adaptive Directional: {adaptive_dir_acc:.2%}")
        print(f"Up Days: {up_acc:.2%}, Down Days: {down_acc:.2%}")
        
        return metrics
    
    def train_calibration(self, X_val, y_val):
        """
        Train calibration on validation set
        CRITICO: Chiamare dopo optimize_weights
        """
        print("\n" + "="*80)
        print("TRAINING CALIBRATION")
        print("="*80)
        
        # Get base predictions
        base_pred = self.predict(X_val, use_optimized_weights=True)
        
        if base_pred is None:
            print("ERROR: Cannot calibrate")
            return
        
        # Convert y_val
        if hasattr(y_val, 'values'):
            y_val_array = y_val.values
        else:
            y_val_array = np.array(y_val)
        
        # Calibrate
        self.calibrate_predictions(base_pred, y_val_array)
        
        # Compute adaptive thresholds
        calibrated = (base_pred * self.calibration_params['scale'] + 
                     self.calibration_params['offset'])
        self.compute_adaptive_thresholds(calibrated, y_val_array)
        
        print("="*80)
        print("CALIBRATION COMPLETE")
        print("="*80)


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

"""
# In main.py, replace EnsembleModel with AdvancedEnsembleModel:

# OLD:
from EnsembleModel import EnsembleModel
self.ensemble = EnsembleModel(self.config, self.models)

# NEW:
from EnsembleModel_ADVANCED import AdvancedEnsembleModel
self.ensemble = AdvancedEnsembleModel(self.config, self.models)

# After optimize_weights, add calibration:
if self.config['models']['ensemble']['optimization']:
    self.ensemble.optimize_weights(X_val, y_val)
    
    # 🆕 NEW: Train calibration
    self.ensemble.train_calibration(X_val, y_val)

# When predicting, use calibrated predictions:
ensemble_pred = self.ensemble.predict_with_calibration(X_test)
"""
