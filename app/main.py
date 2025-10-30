#!/usr/bin/env python3
"""
Stock Market Prediction System - FIXED VERSION
Main orchestrator with proper Advanced ensemble integration

CHANGES FROM ORIGINAL:
1. Fixed AdvancedEnsembleModel import and usage
2. Disabled broken calibration by default
3. Added feature reduction for redundancy
4. Better error handling
5. Proper predict_future() implementation
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import custom modules
from configuration.Logger_config import setup_logger, logger
from app.data.DownloadMarketData import MarketDataDownloader
from app.model.TreeModels import TreeBasedModels, TimeSeriesCV
from app.model.LSTMModel import LSTMModel

# Try to import TimesFM (optional)
try:
    from app.model.TimesFMModel import TimesFMModel

    TIMESFM_AVAILABLE = True
except ImportError:
    logger.warning("TimesFM not available - will use other models only")
    TIMESFM_AVAILABLE = False

# Import FIXED ensemble
from app.model.AdvancedEnsembleModel import AdvancedEnsembleModel

# Import validators
from app.test.test_prediction_quality import PredictionQualityTester
from app.data.feature_validator import FeatureValidator

# Try to import advanced features (optional)
try:
    from app.data.AdvancedFeatures import AdvancedFeatures

    ADVANCED_FEATURES_AVAILABLE = True
    logger.info("✓ Advanced features module available")
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    logger.warning("AdvancedFeatures not available - using base features only")

# Import backtesting
from app.test.Backtesting import Backtester


class StockPredictionSystem:
    """
    Complete stock prediction system orchestrator - FIXED VERSION
    """

    def __init__(self, config_path='config_advanced.yaml'):
        """Initialize the system with configuration"""

        logger.info("=" * 80)
        logger.info("STOCK MARKET PREDICTION SYSTEM - FIXED VERSION")
        logger.info("=" * 80)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {config_path}")

        # Initialize components
        self.data_downloader = MarketDataDownloader(self.config)
        self.feature_engineer = AdvancedFeatures(self.config)

        # Data storage
        self.raw_data = None
        self.features_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Models
        self.models = {}
        self.ensemble = None

        # Results
        self.predictions = {}
        self.metrics = {}

    def run_full_pipeline(self):
        """
        Execute the complete prediction pipeline
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING FULL PREDICTION PIPELINE")
        logger.info("=" * 80 + "\n")

        # Step 1: Data Download
        self.download_data()

        # Step 2: Feature Engineering
        self.create_features()

        # Step 3: Data Splitting
        self.split_data()

        # Step 4: Train Models
        self.train_models()

        # Step 5: Make Predictions
        self.make_predictions()

        # Step 6: Evaluate Models
        self.evaluate_models()

        # Step 6.5: Test Prediction Quality (RIGOROUS)
        self.test_prediction_quality()

        # Step 7: Backtest
        if self.config.get('backtesting', {}).get('enabled', True):
            self.run_backtest()

        # Step 8: Generate Report
        self.generate_report()

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)

        return self.metrics

    def download_data(self):
        """Download all market data"""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 1: DATA DOWNLOAD")
        logger.info("-" * 80)

        data_dict = self.data_downloader.download_all()

        self.raw_data = {
            'stock_data': data_dict['stock_data'],
            'market_indices': data_dict.get('market_indices'),
            'economic_data': data_dict.get('economic_data'),
            'news_data': data_dict.get('news_data')
        }

        logger.info(f"\nStock data shape: {self.raw_data['stock_data'].shape}")
        logger.info("Data download complete\n")

    def create_features(self):
        """Create features from raw data"""
        logger.info("-" * 80)
        logger.info("STEP 2: FEATURE ENGINEERING (UPGRADED)")
        logger.info("-" * 80)

        # Create base features
        use_advanced = self.config.get('features', {}).get('advanced_features', {}).get('enabled', False)

        if use_advanced:
            from app.data.AdvancedFeatures import AdvancedFeatures
            logger.info("Using SIMPLIFIED AdvancedFeatures (30 features)")
            advanced_fe = AdvancedFeatures(self.config)
            self.features_data = advanced_fe.create_all_features(self.raw_data['stock_data'])

        logger.info(f"\nBase features created: {self.features_data.shape}")

        # Add advanced features if available
        if ADVANCED_FEATURES_AVAILABLE:
            logger.info("\n" + "=" * 60)
            logger.info("ADDING ADVANCED FEATURES (SIMPLIFIED)")
            logger.info("=" * 60 + "\n")

            try:
                advanced_creator = AdvancedFeatures(self.features_data)
                self.features_data = advanced_creator.create_all_features()
                logger.info(f"✓ Advanced features integrated: {self.features_data.shape}")
            except Exception as e:
                logger.error(f"Advanced features failed: {e}")
                logger.info("Continuing with base features only")

        # CRITICAL: Remove redundant features
        logger.info("\n--- Removing Redundant Features ---")
        self.features_data = self._remove_redundant_features(
            self.features_data,
            correlation_threshold=0.95
        )

        # Get feature count
        feature_cols = self.feature_engineer.get_feature_columns(self.features_data)
        logger.info(f"\nTotal features after redundancy removal: {len(feature_cols)}")

        # VALIDATE FEATURES
        logger.info("\n--- Validating Features ---")
        validator = FeatureValidator(self.features_data, feature_cols)
        validator.validate_all()

        logger.info("Feature engineering complete\n")

    def _remove_redundant_features(self, df, correlation_threshold=0.95):
        """
        NEW METHOD: Remove highly correlated features to reduce overfitting
        """
        logger.info(f"Removing redundant features (correlation > {correlation_threshold})...")

        feature_cols = self.feature_engineer.get_feature_columns(df)

        if len(feature_cols) == 0:
            logger.warning("No features to check for redundancy")
            return df

        # Calculate correlation matrix (sample for speed)
        sample_size = min(1000, len(df))
        sample_data = df[feature_cols].sample(n=sample_size, random_state=42)

        try:
            corr_matrix = sample_data.corr().abs()

            # Upper triangle of correlations
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find features with correlation > threshold
            to_drop = [column for column in upper.columns
                       if any(upper[column] > correlation_threshold)]

            if to_drop:
                logger.info(f"Dropping {len(to_drop)} redundant features")
                df = df.drop(columns=to_drop, errors='ignore')
            else:
                logger.info("✓ No redundant features found")

        except Exception as e:
            logger.warning(f"Redundancy check failed: {e}")
            logger.info("Continuing with all features")

        return df

    def split_data(self):
        """Split data into train/val/test sets"""
        logger.info("-" * 80)
        logger.info("STEP 3: DATA SPLITTING")
        logger.info("-" * 80)

        # Sort by date
        df = self.features_data.sort_values('Date').reset_index(drop=True)

        # Split ratios from config
        val_split = self.config['training']['validation_split']
        test_split = self.config['training']['test_split']

        n = len(df)
        train_end = int(n * (1 - val_split - test_split))
        val_end = int(n * (1 - test_split))

        self.train_data = df.iloc[:train_end].copy()
        self.val_data = df.iloc[train_end:val_end].copy()
        self.test_data = df.iloc[val_end:].copy()

        logger.info(
            f"Train set: {len(self.train_data)} samples ({self.train_data['Date'].min()} to {self.train_data['Date'].max()})")
        logger.info(
            f"Val set:   {len(self.val_data)} samples ({self.val_data['Date'].min()} to {self.val_data['Date'].max()})")
        logger.info(
            f"Test set:  {len(self.test_data)} samples ({self.test_data['Date'].min()} to {self.test_data['Date'].max()})")
        logger.info("Data splitting complete\n")

    def train_models(self):
        """Train all enabled models"""
        logger.info("-" * 80)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("-" * 80)

        # Get feature columns and target
        feature_cols = self.feature_engineer.get_feature_columns(self.train_data)

        # Choose prediction horizon
        horizons = self.config['prediction']['horizons']
        primary_horizon = horizons[0]  # Use first horizon as primary
        target_col = f'Target_return_{primary_horizon}d'

        logger.info(f"Target variable: {target_col}")
        logger.info(f"Number of features: {len(feature_cols)}\n")

        # Prepare data
        X_train = self.train_data[feature_cols]
        y_train = self.train_data[target_col]
        X_val = self.val_data[feature_cols]
        y_val = self.val_data[target_col]

        # 1. XGBoost
        if self.config['models']['xgboost']['enabled']:
            logger.info("\n--- Training XGBoost ---")
            xgb_model = TreeBasedModels(self.config, model_type='xgboost')
            xgb_model.train(X_train, y_train, X_val, y_val, task='regression')
            self.models['XGBoost'] = xgb_model

        # 2. LightGBM
        if self.config['models']['lightgbm']['enabled']:
            logger.info("\n--- Training LightGBM ---")
            lgb_model = TreeBasedModels(self.config, model_type='lightgbm')
            lgb_model.train(X_train, y_train, X_val, y_val, task='regression')
            self.models['LightGBM'] = lgb_model

        # 3. LSTM
        if self.config['models']['lstm']['enabled']:
            logger.info("\n--- Training LSTM ---")
            try:
                lstm_model = LSTMModel(self.config)
                lstm_model.train(X_train, y_train, X_val, y_val, task='regression')
                self.models['LSTM'] = lstm_model
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")

        # 4. TimesFM
        if TIMESFM_AVAILABLE and self.config['models']['timesfm']['enabled']:
            logger.info("\n--- Loading TimesFM ---")
            try:
                timesfm_model = TimesFMModel(self.config)
                timesfm_model.train(X_train, y_train, X_val, y_val, task='regression')
                self.models['TimesFM'] = timesfm_model
            except Exception as e:
                logger.error(f"TimesFM loading failed: {e}")

        logger.info(f"\nTrained {len(self.models)} models successfully")
        logger.info("Model training complete\n")

    def make_predictions(self):
        """Generate predictions from all models"""
        logger.info("-" * 80)
        logger.info("STEP 5: MAKING PREDICTIONS (UPGRADED)")
        logger.info("-" * 80)

        feature_cols = self.feature_engineer.get_feature_columns(self.test_data)
        X_test = self.test_data[feature_cols]

        # Individual model predictions
        for name, model in self.models.items():
            logger.info(f"Generating predictions from {name}...")
            try:
                pred = model.predict(X_test)
                if pred is not None:
                    self.predictions[name] = pred
                else:
                    logger.warning(f"{name} returned None predictions")
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")

        # Ensemble predictions (FIXED)
        if self.config['models']['ensemble']['enabled'] and len(self.models) > 1:
            logger.info("\n--- Creating Ensemble ---")
            logger.info("Using ADVANCED Ensemble (FIXED version)")

            # Create FIXED AdvancedEnsemble
            self.ensemble = AdvancedEnsembleModel(self.config, self.models)

            # Optimize weights on validation set
            if self.config['models']['ensemble']['optimization']:
                feature_cols_val = self.feature_engineer.get_feature_columns(self.val_data)
                X_val = self.val_data[feature_cols_val]

                horizons = self.config['prediction']['horizons']
                primary_horizon = horizons[0]
                target_col = f'Target_return_{primary_horizon}d'
                y_val = self.val_data[target_col]

                self.ensemble.optimize_weights(X_val, y_val)

                # CRITICAL: DON'T train calibration - it makes things worse!
                # The new AdvancedEnsembleModel has calibration DISABLED by default
                logger.info("✓ Calibration disabled (prevents performance degradation)")

            # Generate ensemble predictions
            ensemble_pred = self.ensemble.predict(X_test, use_optimized_weights=True)

            if ensemble_pred is not None:
                self.predictions['Ensemble'] = ensemble_pred
                logger.info("✓ Ensemble predictions generated")
            else:
                logger.warning("Ensemble prediction failed")

        logger.info(f"\nPredictions generated for {len(self.predictions)} models")
        logger.info("Predictions complete\n")

    def evaluate_models(self):
        """Evaluate all models on test set"""
        logger.info("-" * 80)
        logger.info("STEP 6: MODEL EVALUATION (UPGRADED)")
        logger.info("-" * 80)

        feature_cols = self.feature_engineer.get_feature_columns(self.test_data)
        X_test = self.test_data[feature_cols]

        horizons = self.config['prediction']['horizons']
        primary_horizon = horizons[0]
        target_col = f'Target_return_{primary_horizon}d'
        y_test = self.test_data[target_col]

        # Evaluate each model
        for name, model in self.models.items():
            logger.info(f"\n--- Evaluating {name} ---")
            try:
                metrics = model.evaluate(X_test, y_test, task='regression')
                self.metrics[name] = metrics
            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")

        # Evaluate ensemble
        if self.ensemble is not None:
            logger.info("\n--- Evaluating Ensemble ---")
            try:
                # Use advanced evaluation if available
                if hasattr(self.ensemble, 'evaluate_advanced'):
                    metrics = self.ensemble.evaluate_advanced(X_test, y_test, task='regression')
                else:
                    metrics = self.ensemble.evaluate(X_test, y_test, task='regression')

                self.metrics['Ensemble'] = metrics

                # Compare models
                logger.info("\n--- Model Comparison ---")
                try:
                    comparison = self.ensemble.compare_models(X_test, y_test)
                except Exception as e:
                    logger.warning(f"Model comparison failed: {e}")

            except Exception as e:
                logger.error(f"Ensemble evaluation failed: {e}")

        logger.info("\nModel evaluation complete\n")

    def test_prediction_quality(self):
        """Test rigoroso della qualità delle predizioni"""
        logger.info("-" * 80)
        logger.info("STEP 6.5: PREDICTION QUALITY TESTING")
        logger.info("-" * 80)

        horizons = self.config['prediction']['horizons']
        primary_horizon = horizons[0]
        target_col = f'Target_return_{primary_horizon}d'
        y_test = self.test_data[target_col]

        # Test ensemble o miglior modello
        if 'Ensemble' in self.predictions:
            y_pred = self.predictions['Ensemble']
            model_name = 'Ensemble'
        elif len(self.predictions) > 0:
            # Usa primo modello disponibile
            model_name = list(self.predictions.keys())[0]
            y_pred = self.predictions[model_name]
        else:
            logger.warning("No predictions available for quality testing")
            return

        logger.info(f"Testing predictions from: {model_name}\n")

        # Esegui test suite completo
        tester = PredictionQualityTester(y_test, y_pred)
        results, verdict = tester.run_all_tests()

        # Salva report
        tester.generate_report('outputs/prediction_quality_report.txt')

        # Aggiungi al metrics
        self.metrics['Quality_Tests'] = verdict

        logger.info("\nPrediction quality testing complete\n")

    def run_backtest(self):
        """Run backtest on predictions"""
        logger.info("-" * 80)
        logger.info("STEP 7: BACKTESTING")
        logger.info("-" * 80)

        # Use ensemble predictions if available, otherwise best individual model
        if 'Ensemble' in self.predictions:
            predictions = self.predictions['Ensemble']
            logger.info("Using Ensemble predictions for backtest")
        elif len(self.predictions) > 0:
            # Use best model by R2
            best_model = max(
                [(name, metrics) for name, metrics in self.metrics.items()
                 if name in self.predictions],
                key=lambda x: x[1].get('R2', -999),
                default=(None, None)
            )
            if best_model[0]:
                predictions = self.predictions[best_model[0]]
                logger.info(f"Using {best_model[0]} predictions for backtest")
            else:
                logger.warning("No valid predictions for backtesting")
                return
        else:
            logger.warning("No predictions available for backtesting")
            return

        # Prepare data for backtesting
        backtest_data = self.test_data[['Date', 'Stock', 'Close']].copy()
        backtest_data['Prediction'] = predictions

        # Remove NaN predictions
        backtest_data = backtest_data.dropna(subset=['Prediction'])

        if len(backtest_data) == 0:
            logger.warning("No valid data for backtesting after removing NaN")
            return

        # Run backtest with different strategies
        backtester = Backtester(self.config)

        strategies = self.config['backtesting'].get('strategies', ['threshold_based', 'top_k'])

        for strategy in strategies:
            # Skip if strategy is a dict (complex strategy config)
            if isinstance(strategy, dict):
                strategy_name = strategy.get('name', strategy.get('type', 'unknown'))
                strategy_type = strategy.get('type', strategy_name)
                logger.info(f"\n--- Strategy: {strategy_name} (type: {strategy_type}) ---")

                # Use simple strategy types only
                if strategy_type in ['threshold_based', 'top_k', 'portfolio_optimization']:
                    try:
                        metrics, results = backtester.run_backtest(
                            backtest_data,
                            predictions,
                            strategy=strategy_type,
                            **strategy.get('params', {})
                        )
                        self.metrics[f'Backtest_{strategy_name}'] = metrics
                    except Exception as e:
                        logger.error(f"Backtest failed for {strategy_name}: {e}")
                else:
                    logger.warning(f"Unsupported strategy type: {strategy_type}")
            else:
                # Simple strategy name
                logger.info(f"\n--- Strategy: {strategy} ---")
                try:
                    metrics, results = backtester.run_backtest(
                        backtest_data,
                        predictions,
                        strategy=strategy
                    )
                    self.metrics[f'Backtest_{strategy}'] = metrics
                except Exception as e:
                    logger.error(f"Backtest failed for {strategy}: {e}")

        logger.info("\nBacktesting complete\n")

    def generate_report(self):
        """Generate comprehensive report"""
        logger.info("-" * 80)
        logger.info("STEP 8: GENERATING REPORT")
        logger.info("-" * 80)

        # Create output directory
        os.makedirs('outputs', exist_ok=True)

        # Save metrics
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv('outputs/model_metrics.csv')
        logger.info("Saved: outputs/model_metrics.csv")

        # Save predictions
        predictions_df = pd.DataFrame(self.predictions)
        predictions_df.to_csv('outputs/predictions.csv')
        logger.info("Saved: outputs/predictions.csv")

        # Save feature importance (if available)
        for name, model in self.models.items():
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance(top_n=30)
                if importance is not None:
                    importance.to_csv(f'outputs/feature_importance_{name}.csv')
                    logger.info(f"Saved: outputs/feature_importance_{name}.csv")

        # Save processed data
        self.features_data.to_csv('outputs/features_data.csv', index=False)
        logger.info("Saved: outputs/features_data.csv")

        logger.info("\nReport generation complete\n")

    def predict_future(self, days_ahead=5):
        """
        Make future predictions (out of sample) - FIXED VERSION
        """
        logger.info(f"\nGenerating predictions for {days_ahead} days ahead...")

        # Use most recent data
        feature_cols = self.feature_engineer.get_feature_columns(self.features_data)
        latest_data = self.features_data.groupby('Stock').tail(1)
        X_latest = latest_data[feature_cols]

        # Get predictions from ensemble or best model
        if self.ensemble is not None:
            try:
                # Try with confidence intervals
                result = self.ensemble.predict_with_confidence(X_latest)

                # Check if we got 4 values (success) or less (partial failure)
                if result and len(result) == 4:
                    predictions, lower, upper, std = result

                    # Check for None values
                    if predictions is None:
                        raise ValueError("Ensemble returned None predictions")

                    results = pd.DataFrame({
                        'Stock': latest_data['Stock'].values,
                        'Current_Price': latest_data['Close'].values,
                        'Predicted_Return': predictions,
                        'Predicted_Price': latest_data['Close'].values * (1 + predictions),
                        'Lower_Bound': latest_data['Close'].values * (1 + lower),
                        'Upper_Bound': latest_data['Close'].values * (1 + upper),
                        'Confidence_Std': std,
                        'Date': latest_data['Date'].values
                    })
                else:
                    raise ValueError("Unexpected return format from ensemble")

            except Exception as e:
                logger.warning(f"Confidence interval prediction failed: {e}")
                logger.info("Falling back to simple prediction")

                # Fallback: simple prediction
                predictions = self.ensemble.predict(X_latest)

                if predictions is None:
                    raise ValueError("Both prediction methods failed")

                results = pd.DataFrame({
                    'Stock': latest_data['Stock'].values,
                    'Current_Price': latest_data['Close'].values,
                    'Predicted_Return': predictions,
                    'Predicted_Price': latest_data['Close'].values * (1 + predictions),
                    'Date': latest_data['Date'].values
                })
        else:
            # Use best individual model
            if len(self.metrics) > 0:
                best_model_name = max(
                    self.metrics.items(),
                    key=lambda x: x[1].get('R2', -999)
                )[0]

                if best_model_name in self.models:
                    model = self.models[best_model_name]
                    predictions = model.predict(X_latest)

                    results = pd.DataFrame({
                        'Stock': latest_data['Stock'].values,
                        'Current_Price': latest_data['Close'].values,
                        'Predicted_Return': predictions,
                        'Predicted_Price': latest_data['Close'].values * (1 + predictions),
                        'Date': latest_data['Date'].values
                    })
                else:
                    raise ValueError(f"Best model {best_model_name} not found in models dict")
            else:
                raise ValueError("No metrics available to select best model")

        results = results.sort_values('Predicted_Return', ascending=False)

        logger.info("\nFuture Predictions:")
        logger.info(results.to_string(index=False))

        return results


def main():
    """Main entry point"""

    # Initialize system
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'configuration', 'config_advanced.yaml')

    logger.info(f"Loading config from: {config_path}")

    system = StockPredictionSystem(config_path=config_path)

    # Run full pipeline
    try:
        metrics = system.run_full_pipeline()

        # Generate future predictions
        future_predictions = system.predict_future(days_ahead=5)
        future_predictions.to_csv('outputs/future_predictions.csv', index=False)

        logger.info("\n" + "=" * 80)
        logger.info("SYSTEM EXECUTION COMPLETE")
        logger.info("Check outputs/ directory for results")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error(f"PIPELINE FAILED: {e}")
        logger.error(f"{'=' * 80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()