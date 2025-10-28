#!/usr/bin/env python3
"""
Quick Demo - Stock Market Prediction System
Minimal example to test the system quickly
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..configuration.Logger_config import logger
from ..DownloadMarketData import MarketDataDownloader
from ..FeatureEngineering import FeatureEngineer
from ..TreeModels import TreeBasedModels
from ..EnsembleModel import EnsembleModel


def quick_demo():
    """
    Quick demonstration of the system with minimal configuration
    """
    logger.info("="*80)
    logger.info("QUICK DEMO - STOCK MARKET PREDICTION")
    logger.info("="*80)
    
    # Simple configuration
    config = {
        'data_download': {
            'stocks_list': ['AAPL', 'GOOGL'],  # Just 2 stocks for speed
            'start_date': '2022-01-01',
            'end_date': None
        },
        'features': {
            'technical_indicators': [],
            'sentiment': {'enabled': False},
            'macro_economic': {'enabled': False}
        },
        'models': {
            'xgboost': {
                'enabled': True,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'reg_alpha': 0,
                    'reg_lambda': 1,
                    'random_state': 42
                }
            },
            'lightgbm': {
                'enabled': True,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0,
                    'reg_lambda': 1,
                    'random_state': 42
                }
            },
            'lstm': {'enabled': False},
            'timesfm': {'enabled': False},
            'ensemble': {
                'enabled': True,
                'method': 'weighted_average',
                'optimization': True
            }
        },
        'prediction': {
            'horizons': [1, 5],
            'confidence_level': 0.95
        },
        'training': {
            'validation_split': 0.2,
            'test_split': 0.1,
            'walk_forward': True,
            'early_stopping': {
                'enabled': True,
                'patience': 20,
                'monitor': 'val_loss'
            }
        },
        'logger': {
            'log_level': 'INFO'
        }
    }
    
    # Step 1: Download Data
    logger.info("\n--- STEP 1: Downloading Data ---")
    downloader = MarketDataDownloader(config)
    stock_data = downloader.download_stock_data()
    
    # Step 2: Create Features
    logger.info("\n--- STEP 2: Creating Features ---")
    engineer = FeatureEngineer(config)
    features_data = engineer.create_all_features(stock_data)
    
    # Step 3: Split Data
    logger.info("\n--- STEP 3: Splitting Data ---")
    df = features_data.sort_values('Date').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Step 4: Prepare Features and Target
    feature_cols = engineer.get_feature_columns(train_data)
    target_col = 'Target_return_1d'
    
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_val = val_data[feature_cols]
    y_val = val_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    logger.info(f"Features: {len(feature_cols)}, Target: {target_col}")
    
    # Step 5: Train Models
    logger.info("\n--- STEP 4: Training Models ---")
    
    models = {}
    
    # XGBoost
    logger.info("\nTraining XGBoost...")
    xgb_model = TreeBasedModels(config, model_type='xgboost')
    xgb_model.train(X_train, y_train, X_val, y_val, task='regression')
    models['XGBoost'] = xgb_model
    
    # LightGBM
    logger.info("\nTraining LightGBM...")
    lgb_model = TreeBasedModels(config, model_type='lightgbm')
    lgb_model.train(X_train, y_train, X_val, y_val, task='regression')
    models['LightGBM'] = lgb_model
    
    # Step 6: Create Ensemble
    logger.info("\n--- STEP 5: Creating Ensemble ---")
    ensemble = EnsembleModel(config, models)
    ensemble.optimize_weights(X_val, y_val)
    
    # Step 7: Evaluate on Test Set
    logger.info("\n--- STEP 6: Evaluation on Test Set ---")
    
    for name, model in models.items():
        logger.info(f"\n{name}:")
        model.evaluate(X_test, y_test, task='regression')
    
    logger.info("\nEnsemble:")
    ensemble.evaluate(X_test, y_test, task='regression')
    
    # Step 8: Compare Models
    logger.info("\n--- STEP 7: Model Comparison ---")
    comparison = ensemble.compare_models(X_test, y_test)
    print("\n", comparison)
    
    # Step 9: Feature Importance
    logger.info("\n--- STEP 8: Top 10 Most Important Features ---")
    importance = xgb_model.get_feature_importance(top_n=10)
    print("\n", importance)
    
    # Step 10: Make Predictions for Latest Data
    logger.info("\n--- STEP 9: Future Predictions ---")
    
    latest_data = features_data.groupby('Stock').tail(1)
    X_latest = latest_data[feature_cols]
    
    predictions, lower, upper, std = ensemble.predict_with_confidence(X_latest)
    
    results = pd.DataFrame({
        'Stock': latest_data['Stock'].values,
        'Current_Price': latest_data['Close'].values,
        'Predicted_Return_%': predictions * 100,
        'Lower_Bound_%': lower * 100,
        'Upper_Bound_%': upper * 100,
        'Confidence_Std': std
    })
    
    print("\n", results.to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETE!")
    logger.info("="*80)
    
    return {
        'models': models,
        'ensemble': ensemble,
        'comparison': comparison,
        'predictions': results
    }


if __name__ == '__main__':
    try:
        results = quick_demo()
        print("\n✓ Demo completed successfully!")
    except Exception as e:
        logger.error(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
