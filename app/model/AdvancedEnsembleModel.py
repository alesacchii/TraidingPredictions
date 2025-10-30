#!/usr/bin/env python3
"""
FIXED Advanced Ensemble Model
Extends EnsembleModel with proper method implementations
NO BROKEN CALIBRATION
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
import sys
import os

# Import base EnsembleModel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from EnsembleModel import EnsembleModel

try:
    from app.configuration.Logger_config import setup_logger, logger
except:
    import logging
    logger = logging.getLogger(__name__)


class AdvancedEnsembleModel(EnsembleModel):
    """
    Advanced Ensemble that PROPERLY extends EnsembleModel
    
    Key fixes:
    1. Inherits all base methods from EnsembleModel
    2. NO BROKEN CALIBRATION by default
    3. Implements missing methods properly
    4. Compatible with main.py expectations
    """
    
    def __init__(self, config, models_dict=None):
        # Initialize base class
        super().__init__(config, models_dict)
        
        # Advanced features (DISABLED by default - they break everything)
        self.use_calibration = False
        self.calibration_scale = 1.0
        self.calibration_offset = 0.0
        self.adaptive_up_threshold = 0.01
        self.adaptive_down_threshold = 0.01
        
        logger.info("✓ Advanced Ensemble initialized (inherits from EnsembleModel)")
        logger.info("  - Calibration: DISABLED (causes harm)")
        logger.info("  - All base methods: AVAILABLE")
    
    def train_calibration(self, X_val, y_val):
        """
        Train calibration (OPTIONAL - usually makes things worse)
        Only enable if you know what you're doing
        """
        if not self.use_calibration:
            logger.info("Calibration disabled - skipping")
            return
        
        logger.info("\n--- Training Calibration (EXPERIMENTAL) ---")
        
        # Get base predictions
        predictions_df = self.collect_predictions(X_val)
        if predictions_df is None:
            logger.warning("Cannot calibrate - no predictions")
            return
        
        # Simple ensemble (equal weights for calibration)
        base_pred = predictions_df.mean(axis=1).values
        y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
        
        # Calculate scale factor
        pred_std = np.std(base_pred)
        actual_std = np.std(y_val_array)
        
        if pred_std > 1e-6:
            self.calibration_scale = actual_std / pred_std
        else:
            self.calibration_scale = 1.0
        
        # Calculate offset
        self.calibration_offset = np.mean(y_val_array) - np.mean(base_pred) * self.calibration_scale
        
        logger.info(f"Scale factor: {self.calibration_scale:.4f}")
        logger.info(f"Offset: {self.calibration_offset:.6f}")
        
        # Test calibrated predictions
        calibrated = base_pred * self.calibration_scale + self.calibration_offset
        rmse_before = np.sqrt(mean_squared_error(y_val_array, base_pred))
        rmse_after = np.sqrt(mean_squared_error(y_val_array, calibrated))
        
        if rmse_after < rmse_before:
            logger.info(f"✓ Calibration improves RMSE: {rmse_before:.6f} -> {rmse_after:.6f}")
            self.use_calibration = True
        else:
            logger.warning(f"✗ Calibration WORSENS RMSE: {rmse_before:.6f} -> {rmse_after:.6f}")
            logger.warning("  Calibration will be DISABLED")
            self.use_calibration = False
            self.calibration_scale = 1.0
            self.calibration_offset = 0.0
    
    def predict(self, X, use_optimized_weights=True):
        """
        Override predict to optionally apply calibration
        (Only if calibration actually helps - usually it doesn't)
        """
        # Get base predictions from parent class
        base_predictions = super().predict(X, use_optimized_weights)
        
        if base_predictions is None:
            return None
        
        # Apply calibration ONLY if enabled and helpful
        if self.use_calibration:
            calibrated = base_predictions * self.calibration_scale + self.calibration_offset
            return calibrated
        
        return base_predictions
    
    def predict_with_confidence(self, X, use_optimized_weights=True):
        """
        Implement missing method - predict with confidence intervals
        """
        predictions_df = self.collect_predictions(X)
        
        if predictions_df is None or predictions_df.empty:
            return None, None, None, None
        
        # Point prediction
        ensemble_pred = self.predict(X, use_optimized_weights)
        
        # Confidence from prediction variance
        pred_std = predictions_df.std(axis=1).values
        pred_mean = predictions_df.mean(axis=1).values
        
        # 95% confidence interval
        z_score = 1.96
        lower_bound = pred_mean - z_score * pred_std
        upper_bound = pred_mean + z_score * pred_std
        
        return ensemble_pred, lower_bound, upper_bound, pred_std
    
    def predict_with_confidence_intervals(self, X, use_optimized_weights=True):
        """
        Alias for predict_with_confidence (for compatibility)
        """
        return self.predict_with_confidence(X, use_optimized_weights)
    
    def evaluate_advanced(self, X, y, task='regression'):
        """
        Enhanced evaluation with directional accuracy
        """
        predictions = self.predict(X)
        
        if predictions is None:
            logger.error("Cannot evaluate - predictions failed")
            return {}
        
        # Remove NaN
        mask = ~np.isnan(predictions)
        predictions_clean = predictions[mask]
        y_clean = y[mask].values if hasattr(y, 'values') else y[mask]
        
        if len(predictions_clean) == 0:
            logger.warning("No valid predictions for evaluation")
            return {}
        
        # Basic metrics
        mse = mean_squared_error(y_clean, predictions_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_clean, predictions_clean)
        r2 = r2_score(y_clean, predictions_clean)
        
        # Directional accuracy
        correct_direction = np.sum((predictions_clean > 0) == (y_clean > 0))
        directional_accuracy = correct_direction / len(y_clean)
        
        # Separate up/down accuracy
        up_mask = y_clean > 0
        down_mask = y_clean < 0
        
        up_accuracy = np.mean((predictions_clean[up_mask] > 0)) if up_mask.sum() > 0 else 0
        down_accuracy = np.mean((predictions_clean[down_mask] < 0)) if down_mask.sum() > 0 else 0
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Directional': directional_accuracy,
            'Up_Days': up_accuracy,
            'Down_Days': down_accuracy
        }
        
        logger.info(f"\n📊 Advanced Ensemble Evaluation:")
        logger.info(f"RMSE: {rmse:.6f}, R²: {r2:.4f}")
        logger.info(f"Directional: {directional_accuracy:.2%}")
        logger.info(f"Up Days: {up_accuracy:.2%}, Down Days: {down_accuracy:.2%}")
        
        return metrics
    
    def compare_models(self, X, y):
        """
        Implement missing compare_models method
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
            
            # Directional accuracy
            correct_direction = np.sum((pred > 0) == (y_clean > 0))
            directional = correct_direction / len(y_clean)
            
            comparison[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Directional': directional
            }
        
        # Add ensemble
        ensemble_pred = self.predict(X[valid_mask])
        if ensemble_pred is not None:
            valid_ensemble = ~np.isnan(ensemble_pred)
            if valid_ensemble.sum() > 0:
                rmse = np.sqrt(mean_squared_error(y_clean[valid_ensemble], ensemble_pred[valid_ensemble]))
                mae = mean_absolute_error(y_clean[valid_ensemble], ensemble_pred[valid_ensemble])
                r2 = r2_score(y_clean[valid_ensemble], ensemble_pred[valid_ensemble])
                
                correct_direction = np.sum((ensemble_pred[valid_ensemble] > 0) == (y_clean[valid_ensemble] > 0))
                directional = correct_direction / valid_ensemble.sum()
                
                comparison['Ensemble'] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'Directional': directional
                }
        
        df_comparison = pd.DataFrame(comparison).T
        df_comparison = df_comparison.sort_values('RMSE')
        
        logger.info("\n--- Model Comparison ---")
        logger.info(f"\n{df_comparison}")
        
        return df_comparison


def create_advanced_ensemble(config, models):
    """
    Factory function to create advanced ensemble
    """
    ensemble = AdvancedEnsembleModel(config, models)
    logger.info("✓ Advanced ensemble created (extends EnsembleModel)")
    return ensemble


if __name__ == '__main__':
    print("✓ AdvancedEnsembleModel - Fixed version")
    print("  - Properly extends EnsembleModel")
    print("  - All methods implemented")
    print("  - Calibration disabled by default")
    print("  - Compatible with main.py")
