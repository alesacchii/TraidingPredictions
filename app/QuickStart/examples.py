"""
EXAMPLES - Stock Market Prediction System
Practical usage examples for different scenarios
"""

import yaml
import pandas as pd
from ..main import StockPredictionSystem
from ..DownloadMarketData import MarketDataDownloader
from ..FeatureEngineering import FeatureEngineer
from ..TreeModels import TreeBasedModels
from ..EnsembleModel import EnsembleModel
from ..Backtesting import Backtester


# ============================================================================
# EXAMPLE 1: Basic Usage - Single Stock Prediction
# ============================================================================

def example_1_single_stock():
    """
    Predict future returns for a single stock (e.g., Apple)
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Stock Prediction (AAPL)")
    print("="*80)
    
    # Custom config for single stock
    config = {
        'data_download': {
            'stocks_list': ['AAPL'],
            'start_date': '2020-01-01',
            'end_date': None
        },
        'features': {
            'sentiment': {'enabled': False},
            'macro_economic': {'enabled': False}
        },
        'models': {
            'xgboost': {'enabled': True, 'params': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42}},
            'lightgbm': {'enabled': True, 'params': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42}},
            'lstm': {'enabled': False},
            'timesfm': {'enabled': False},
            'ensemble': {'enabled': True, 'method': 'weighted_average', 'optimization': True}
        },
        'prediction': {'horizons': [1, 5, 20]},
        'training': {'validation_split': 0.2, 'test_split': 0.1},
        'backtesting': {'enabled': False},
        'logger': {'log_level': 'INFO'}
    }
    
    # Save temporary config
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Run system
    system = StockPredictionSystem('temp_config.yaml')
    system.run_full_pipeline()
    
    # Get predictions
    predictions = system.predict_future(days_ahead=5)
    print("\nPredictions for next 5 days:")
    print(predictions)
    
    return predictions


# ============================================================================
# EXAMPLE 2: Portfolio Analysis - Multiple Stocks
# ============================================================================

def example_2_portfolio():
    """
    Analyze a portfolio of tech stocks
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Tech Portfolio Analysis")
    print("="*80)
    
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX']
    
    config = {
        'data_download': {
            'stocks_list': tech_stocks,
            'start_date': '2019-01-01',
            'end_date': None
        },
        'models': {
            'xgboost': {'enabled': True, 'params': {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'random_state': 42}},
            'lightgbm': {'enabled': True, 'params': {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'random_state': 42}},
            'lstm': {'enabled': False},
            'timesfm': {'enabled': False},
            'ensemble': {'enabled': True, 'method': 'weighted_average', 'optimization': True}
        },
        'prediction': {'horizons': [1, 5, 20]},
        'training': {'validation_split': 0.2, 'test_split': 0.1},
        'backtesting': {'enabled': True, 'initial_capital': 100000, 'strategies': ['top_k']},
        'features': {'sentiment': {'enabled': False}, 'macro_economic': {'enabled': False}},
        'logger': {'log_level': 'INFO'}
    }
    
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    system = StockPredictionSystem('temp_config.yaml')
    metrics = system.run_full_pipeline()
    
    # Get best performing stocks
    predictions = system.predict_future(days_ahead=5)
    top_5 = predictions.nlargest(5, 'Predicted_Return')
    
    print("\n📈 TOP 5 STOCKS TO BUY:")
    print(top_5[['Stock', 'Current_Price', 'Predicted_Return_%']])
    
    return top_5


# ============================================================================
# EXAMPLE 3: Quick Daily Predictions
# ============================================================================

def example_3_daily_predictions():
    """
    Get quick predictions for today - minimal setup
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Quick Daily Predictions")
    print("="*80)
    
    # Fast config - only XGBoost for speed
    config = {
        'data_download': {
            'stocks_list': ['AAPL', 'GOOGL', 'TSLA', 'NVDA', 'META'],
            'start_date': '2023-01-01',
            'end_date': None
        },
        'models': {
            'xgboost': {'enabled': True, 'params': {'n_estimators': 100, 'max_depth': 4, 'random_state': 42}},
            'lightgbm': {'enabled': False},
            'lstm': {'enabled': False},
            'timesfm': {'enabled': False},
            'ensemble': {'enabled': False}
        },
        'prediction': {'horizons': [1]},
        'training': {'validation_split': 0.15, 'test_split': 0.05},
        'backtesting': {'enabled': False},
        'features': {'sentiment': {'enabled': False}, 'macro_economic': {'enabled': False}},
        'logger': {'log_level': 'WARNING'}  # Less verbose
    }
    
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    system = StockPredictionSystem('temp_config.yaml')
    system.run_full_pipeline()
    
    predictions = system.predict_future(days_ahead=1)
    
    print("\n🎯 TODAY'S PREDICTIONS:")
    for _, row in predictions.iterrows():
        direction = "📈 BUY" if row['Predicted_Return_%'] > 0 else "📉 SELL"
        print(f"{direction}  {row['Stock']:6s} | Current: ${row['Current_Price']:.2f} | Expected: {row['Predicted_Return_%']:+.2f}%")
    
    return predictions


# ============================================================================
# EXAMPLE 4: Feature Importance Analysis
# ============================================================================

def example_4_feature_analysis():
    """
    Analyze which features are most important for predictions
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Feature Importance Analysis")
    print("="*80)
    
    # Load config
    with open('config_advanced.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Simplify for demo
    config['data_download']['stocks_list'] = ['AAPL', 'GOOGL']
    config['models']['lstm']['enabled'] = False
    config['models']['timesfm']['enabled'] = False
    config['backtesting']['enabled'] = False
    
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    system = StockPredictionSystem('temp_config.yaml')
    system.download_data()
    system.create_features()
    system.split_data()
    system.train_models()
    
    # Get feature importance from XGBoost
    xgb_model = system.models['XGBoost']
    importance = xgb_model.get_feature_importance(top_n=20)
    
    print("\n🔍 TOP 20 MOST IMPORTANT FEATURES:")
    print(importance.to_string(index=False))
    
    # Visualize if possible
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        plt.barh(importance['feature'][:20], importance['importance'][:20])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=150)
        print("\n📊 Chart saved: outputs/feature_importance.png")
    except:
        pass
    
    return importance


# ============================================================================
# EXAMPLE 5: Custom Backtesting Strategy
# ============================================================================

def example_5_custom_backtest():
    """
    Test a custom trading strategy
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Backtesting Strategy")
    print("="*80)
    
    # Full system run first
    with open('config_advanced.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Simplify
    config['data_download']['stocks_list'] = ['AAPL', 'GOOGL', 'MSFT']
    config['data_download']['start_date'] = '2022-01-01'
    config['models']['lstm']['enabled'] = False
    config['models']['timesfm']['enabled'] = False
    
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    system = StockPredictionSystem('temp_config.yaml')
    system.run_full_pipeline()
    
    # Get predictions
    predictions = system.predictions.get('Ensemble', system.predictions.get('XGBoost'))
    
    # Custom backtest with aggressive strategy
    test_data = system.test_data[['Date', 'Stock', 'Close']].copy()
    test_data['Prediction'] = predictions
    
    backtester = Backtester(system.config)
    
    # Strategy 1: High threshold (only very confident predictions)
    print("\n📊 Strategy 1: High Confidence (threshold=2%)")
    metrics1, _ = backtester.run_backtest(
        test_data, predictions,
        strategy='threshold_based',
        threshold=0.02,
        hold_days=5
    )
    
    # Strategy 2: Top 2 stocks
    print("\n📊 Strategy 2: Top 2 Stocks (rebalance every 10 days)")
    metrics2, _ = backtester.run_backtest(
        test_data, predictions,
        strategy='top_k',
        k=2,
        rebalance_days=10
    )
    
    # Compare
    comparison = pd.DataFrame({
        'High Confidence': metrics1,
        'Top 2 Stocks': metrics2
    }).T
    
    print("\n📈 STRATEGY COMPARISON:")
    print(comparison[['Total_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate']])
    
    return comparison


# ============================================================================
# EXAMPLE 6: Compare Different Models
# ============================================================================

def example_6_model_comparison():
    """
    Compare performance of different models
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Model Comparison")
    print("="*80)
    
    results = {}
    
    # Test each model individually
    models_to_test = ['xgboost', 'lightgbm']
    
    for model_name in models_to_test:
        print(f"\n--- Testing {model_name.upper()} ---")
        
        config = {
            'data_download': {'stocks_list': ['AAPL'], 'start_date': '2021-01-01'},
            'models': {
                'xgboost': {'enabled': model_name == 'xgboost', 'params': {'n_estimators': 150, 'random_state': 42}},
                'lightgbm': {'enabled': model_name == 'lightgbm', 'params': {'n_estimators': 150, 'random_state': 42}},
                'lstm': {'enabled': False},
                'timesfm': {'enabled': False},
                'ensemble': {'enabled': False}
            },
            'prediction': {'horizons': [1]},
            'training': {'validation_split': 0.2, 'test_split': 0.1},
            'backtesting': {'enabled': False},
            'features': {'sentiment': {'enabled': False}, 'macro_economic': {'enabled': False}},
            'logger': {'log_level': 'WARNING'}
        }
        
        with open('temp_config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        system = StockPredictionSystem('temp_config.yaml')
        system.run_full_pipeline()
        
        results[model_name.upper()] = system.metrics.get(model_name.upper(), {})
    
    # Compare
    comparison = pd.DataFrame(results).T
    
    print("\n🏆 MODEL COMPARISON:")
    print(comparison)
    
    return comparison


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == '__main__':
    import os
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "="*80)
    print("STOCK PREDICTION SYSTEM - PRACTICAL EXAMPLES")
    print("="*80)
    print("\nSelect an example to run:")
    print("1. Single Stock Prediction (AAPL)")
    print("2. Tech Portfolio Analysis")
    print("3. Quick Daily Predictions")
    print("4. Feature Importance Analysis")
    print("5. Custom Backtesting Strategy")
    print("6. Model Comparison")
    print("0. Run all examples")
    
    choice = input("\nEnter choice (0-6): ").strip()
    
    examples = {
        '1': example_1_single_stock,
        '2': example_2_portfolio,
        '3': example_3_daily_predictions,
        '4': example_4_feature_analysis,
        '5': example_5_custom_backtest,
        '6': example_6_model_comparison
    }
    
    if choice == '0':
        # Run all
        for name, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"\n❌ Example {name} failed: {e}")
    elif choice in examples:
        try:
            examples[choice]()
            print("\n✅ Example completed successfully!")
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice")
    
    # Cleanup
    if os.path.exists('temp_config.yaml'):
        os.remove('temp_config.yaml')
