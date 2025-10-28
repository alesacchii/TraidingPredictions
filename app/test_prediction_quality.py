#!/usr/bin/env python3
"""
PREDICTION QUALITY TESTING SUITE
Test rigoroso della bontà delle previsioni - SENZA pietà
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from configuration.Logger_config import setup_logger, logger
import warnings
warnings.filterwarnings('ignore')


class PredictionQualityTester:
    """
    Test suite completo per validare la bontà delle predizioni
    Approccio cinico: se i test falliscono, il modello è inutile
    """
    
    def __init__(self, y_true, y_pred, prices=None):
        """
        Args:
            y_true: Actual returns/values
            y_pred: Predicted returns/values  
            prices: Optional actual prices for additional tests
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.prices = np.array(prices) if prices is not None else None
        
        # Remove NaN
        mask = ~(np.isnan(self.y_true) | np.isnan(self.y_pred))
        self.y_true = self.y_true[mask]
        self.y_pred = self.y_pred[mask]
        if self.prices is not None:
            self.prices = self.prices[mask]
        
        self.results = {}
    
    def run_all_tests(self):
        """
        Esegui TUTTI i test di qualità
        Returns: dict con risultati e verdict finale
        """
        logger.info("="*80)
        logger.info("PREDICTION QUALITY TEST SUITE - STARTING")
        logger.info("="*80)
        
        # Test 1: Basic Metrics
        self.test_basic_metrics()
        
        # Test 2: Better than Naive
        self.test_better_than_naive()
        
        # Test 3: Directional Accuracy
        self.test_directional_accuracy()
        
        # Test 4: Statistical Significance
        self.test_statistical_significance()
        
        # Test 5: Residual Analysis
        self.test_residual_analysis()
        
        # Test 6: Prediction Stability
        self.test_prediction_stability()
        
        # Test 7: Outlier Handling
        self.test_outlier_handling()
        
        # Test 8: Economic Significance
        self.test_economic_significance()
        
        # Final Verdict
        verdict = self.generate_verdict()
        
        return self.results, verdict
    
    def test_basic_metrics(self):
        """Test 1: Metriche base - devono essere accettabili"""
        logger.info("\n--- TEST 1: Basic Metrics ---")
        
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        mae = mean_absolute_error(self.y_true, self.y_pred)
        r2 = r2_score(self.y_true, self.y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((self.y_true - self.y_pred) / (self.y_true + 1e-8))) * 100
        
        # Correlation
        correlation = np.corrcoef(self.y_true, self.y_pred)[0, 1]
        
        self.results['basic_metrics'] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Correlation': correlation
        }
        
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info(f"Correlation: {correlation:.4f}")
        
        # Verdetto
        passed = (r2 > 0.10 and correlation > 0.30)
        self.results['basic_metrics']['passed'] = passed
        
        if passed:
            logger.info("✓ PASS: Metrics are acceptable")
        else:
            logger.warning("✗ FAIL: Metrics are too poor (R² < 0.10 or Correlation < 0.30)")
    
    def test_better_than_naive(self):
        """Test 2: Deve battere benchmark naive (persistence model)"""
        logger.info("\n--- TEST 2: Better than Naive Baseline ---")
        
        # Naive prediction: assume domani = oggi
        naive_pred = np.roll(self.y_true, 1)
        naive_pred[0] = 0  # First value
        
        rmse_model = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        rmse_naive = np.sqrt(mean_squared_error(self.y_true, naive_pred))
        
        improvement = (rmse_naive - rmse_model) / rmse_naive * 100
        
        self.results['naive_comparison'] = {
            'RMSE_model': rmse_model,
            'RMSE_naive': rmse_naive,
            'Improvement_%': improvement
        }
        
        logger.info(f"Model RMSE: {rmse_model:.6f}")
        logger.info(f"Naive RMSE: {rmse_naive:.6f}")
        logger.info(f"Improvement: {improvement:+.2f}%")
        
        # Verdetto
        passed = improvement > 5  # Almeno 5% meglio
        self.results['naive_comparison']['passed'] = passed
        
        if passed:
            logger.info(f"✓ PASS: Model is {improvement:.1f}% better than naive")
        else:
            logger.warning(f"✗ FAIL: Model is only {improvement:.1f}% better (need >5%)")
    
    def test_directional_accuracy(self):
        """Test 3: Accuratezza direzionale - predice su/giù correttamente?"""
        logger.info("\n--- TEST 3: Directional Accuracy ---")
        
        # Direzione vera vs predetta
        true_direction = (self.y_true > 0).astype(int)
        pred_direction = (self.y_pred > 0).astype(int)
        
        directional_accuracy = np.mean(true_direction == pred_direction)
        
        # Random baseline = 50%
        baseline = 0.50
        improvement_vs_random = (directional_accuracy - baseline) / baseline * 100
        
        # Separate accuracy for up and down
        up_mask = true_direction == 1
        down_mask = true_direction == 0
        
        up_accuracy = np.mean(pred_direction[up_mask] == 1) if up_mask.sum() > 0 else 0
        down_accuracy = np.mean(pred_direction[down_mask] == 0) if down_mask.sum() > 0 else 0
        
        self.results['directional_accuracy'] = {
            'Overall': directional_accuracy,
            'Up_Days': up_accuracy,
            'Down_Days': down_accuracy,
            'Improvement_vs_Random_%': improvement_vs_random
        }
        
        logger.info(f"Overall Directional Accuracy: {directional_accuracy:.2%}")
        logger.info(f"Up Days Accuracy: {up_accuracy:.2%}")
        logger.info(f"Down Days Accuracy: {down_accuracy:.2%}")
        logger.info(f"vs Random (50%): {improvement_vs_random:+.2f}%")
        
        # Verdetto - almeno 52% per essere utile nel trading
        passed = directional_accuracy > 0.52
        self.results['directional_accuracy']['passed'] = passed
        
        if passed:
            logger.info(f"✓ PASS: Directional accuracy {directional_accuracy:.1%} > 52%")
        else:
            logger.warning(f"✗ FAIL: Directional accuracy {directional_accuracy:.1%} ≤ 52%")
    
    def test_statistical_significance(self):
        """Test 4: Le predizioni sono statisticamente significative?"""
        logger.info("\n--- TEST 4: Statistical Significance ---")
        
        # T-test: predictions vs zero (random)
        t_stat, p_value = stats.ttest_1samp(self.y_pred - self.y_true, 0)
        
        # Correlation significance
        correlation = np.corrcoef(self.y_true, self.y_pred)[0, 1]
        n = len(self.y_true)
        # t-statistic for correlation
        t_corr = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
        p_corr = 2 * (1 - stats.t.cdf(abs(t_corr), n - 2))
        
        self.results['statistical_significance'] = {
            'T_statistic': t_stat,
            'P_value': p_value,
            'Correlation_P_value': p_corr,
            'Significant_at_5%': p_corr < 0.05,
            'Significant_at_1%': p_corr < 0.01
        }
        
        logger.info(f"T-statistic: {t_stat:.4f}")
        logger.info(f"P-value: {p_value:.4f}")
        logger.info(f"Correlation P-value: {p_corr:.4f}")
        
        # Verdetto
        passed = p_corr < 0.05
        self.results['statistical_significance']['passed'] = passed
        
        if passed:
            logger.info("✓ PASS: Predictions are statistically significant (p < 0.05)")
        else:
            logger.warning("✗ FAIL: Predictions are NOT statistically significant")
    
    def test_residual_analysis(self):
        """Test 5: Analisi residui - devono essere random"""
        logger.info("\n--- TEST 5: Residual Analysis ---")
        
        residuals = self.y_true - self.y_pred
        
        # Test normalità (Shapiro-Wilk)
        if len(residuals) >= 3:
            _, p_normality = stats.shapiro(residuals[:5000])  # max 5000 samples
        else:
            p_normality = 0
        
        # Autocorrelazione residui (Durbin-Watson)
        residuals_diff = np.diff(residuals)
        dw_stat = np.sum(residuals_diff**2) / np.sum(residuals**2)
        
        # Mean e std residui
        mean_residuals = np.mean(residuals)
        std_residuals = np.std(residuals)
        
        self.results['residual_analysis'] = {
            'Mean': mean_residuals,
            'Std': std_residuals,
            'Normality_P_value': p_normality,
            'Durbin_Watson': dw_stat,
            'Residuals_Normal': p_normality > 0.05,
            'No_Autocorrelation': 1.5 < dw_stat < 2.5
        }
        
        logger.info(f"Residuals Mean: {mean_residuals:.6f}")
        logger.info(f"Residuals Std: {std_residuals:.6f}")
        logger.info(f"Normality P-value: {p_normality:.4f}")
        logger.info(f"Durbin-Watson: {dw_stat:.4f}")
        
        # Verdetto
        passed = abs(mean_residuals) < 0.001 and 1.5 < dw_stat < 2.5
        self.results['residual_analysis']['passed'] = passed
        
        if passed:
            logger.info("✓ PASS: Residuals are well-behaved")
        else:
            logger.warning("✗ WARNING: Residuals show patterns (potential bias)")
    
    def test_prediction_stability(self):
        """Test 6: Stabilità predizioni - non deve essere erratico"""
        logger.info("\n--- TEST 6: Prediction Stability ---")
        
        # Volatility of predictions vs actuals
        pred_volatility = np.std(self.y_pred)
        true_volatility = np.std(self.y_true)
        volatility_ratio = pred_volatility / true_volatility
        
        # Change in predictions (smoothness)
        pred_changes = np.abs(np.diff(self.y_pred))
        true_changes = np.abs(np.diff(self.y_true))
        
        avg_pred_change = np.mean(pred_changes)
        avg_true_change = np.mean(true_changes)
        
        self.results['prediction_stability'] = {
            'Pred_Volatility': pred_volatility,
            'True_Volatility': true_volatility,
            'Volatility_Ratio': volatility_ratio,
            'Avg_Pred_Change': avg_pred_change,
            'Avg_True_Change': avg_true_change
        }
        
        logger.info(f"Prediction Volatility: {pred_volatility:.6f}")
        logger.info(f"True Volatility: {true_volatility:.6f}")
        logger.info(f"Volatility Ratio: {volatility_ratio:.4f}")
        logger.info(f"Avg Prediction Change: {avg_pred_change:.6f}")
        
        # Verdetto - predizioni non devono essere troppo smooth o troppo erratiche
        passed = 0.5 < volatility_ratio < 2.0
        self.results['prediction_stability']['passed'] = passed
        
        if passed:
            logger.info(f"✓ PASS: Predictions have reasonable volatility ({volatility_ratio:.2f})")
        else:
            logger.warning(f"✗ WARNING: Predictions volatility unusual ({volatility_ratio:.2f})")
    
    def test_outlier_handling(self):
        """Test 7: Come gestisce gli outlier?"""
        logger.info("\n--- TEST 7: Outlier Handling ---")
        
        # Identifica outlier (>3 std)
        residuals = self.y_true - self.y_pred
        std_res = np.std(residuals)
        outliers = np.abs(residuals) > 3 * std_res
        
        n_outliers = np.sum(outliers)
        outlier_pct = n_outliers / len(residuals) * 100
        
        # Performance su non-outlier vs outlier
        normal_mae = mean_absolute_error(self.y_true[~outliers], self.y_pred[~outliers])
        outlier_mae = mean_absolute_error(self.y_true[outliers], self.y_pred[outliers]) if n_outliers > 0 else 0
        
        self.results['outlier_handling'] = {
            'N_Outliers': n_outliers,
            'Outlier_%': outlier_pct,
            'Normal_MAE': normal_mae,
            'Outlier_MAE': outlier_mae
        }
        
        logger.info(f"Outliers detected: {n_outliers} ({outlier_pct:.2f}%)")
        logger.info(f"MAE on normal points: {normal_mae:.6f}")
        logger.info(f"MAE on outliers: {outlier_mae:.6f}")
        
        # Verdetto - outlier dovrebbero essere < 5%
        passed = outlier_pct < 5.0
        self.results['outlier_handling']['passed'] = passed
        
        if passed:
            logger.info(f"✓ PASS: Outlier percentage acceptable ({outlier_pct:.1f}%)")
        else:
            logger.warning(f"✗ WARNING: Too many outliers ({outlier_pct:.1f}%)")
    
    def test_economic_significance(self):
        """Test 8: Significatività economica - genera profitto?"""
        logger.info("\n--- TEST 8: Economic Significance ---")
        
        # Simula trading strategy: buy se pred > threshold
        thresholds = [0.001, 0.005, 0.01]
        results_by_threshold = {}
        
        for threshold in thresholds:
            signals = (self.y_pred > threshold).astype(int)
            returns = self.y_true * signals
            
            total_return = np.sum(returns)
            avg_return = np.mean(returns[signals == 1]) if signals.sum() > 0 else 0
            n_trades = signals.sum()
            win_rate = np.sum((returns > 0) & (signals == 1)) / n_trades if n_trades > 0 else 0
            
            results_by_threshold[threshold] = {
                'Total_Return': total_return,
                'Avg_Return': avg_return,
                'N_Trades': n_trades,
                'Win_Rate': win_rate
            }
            
            logger.info(f"Threshold {threshold:.3f}: Total Return={total_return:.4f}, Trades={n_trades}, WinRate={win_rate:.2%}")
        
        self.results['economic_significance'] = results_by_threshold
        
        # Verdetto - almeno una strategia deve essere profittevole
        best_return = max(r['Total_Return'] for r in results_by_threshold.values())
        passed = best_return > 0
        self.results['economic_significance']['passed'] = passed
        
        if passed:
            logger.info(f"✓ PASS: Predictions can generate positive returns ({best_return:.4f})")
        else:
            logger.warning("✗ FAIL: No threshold produces positive returns")
    
    def generate_verdict(self):
        """Genera verdetto finale - BRUTALMENTE ONESTO"""
        logger.info("\n" + "="*80)
        logger.info("FINAL VERDICT")
        logger.info("="*80)
        
        # Conta test passati
        test_names = [
            'basic_metrics', 'naive_comparison', 'directional_accuracy',
            'statistical_significance', 'residual_analysis', 'prediction_stability',
            'outlier_handling', 'economic_significance'
        ]
        
        passed_tests = []
        failed_tests = []
        
        for test_name in test_names:
            if test_name in self.results:
                if self.results[test_name].get('passed', False):
                    passed_tests.append(test_name)
                else:
                    failed_tests.append(test_name)
        
        total_tests = len(test_names)
        passed_count = len(passed_tests)
        pass_rate = passed_count / total_tests
        
        logger.info(f"Tests Passed: {passed_count}/{total_tests} ({pass_rate:.0%})")
        logger.info(f"Passed: {', '.join(passed_tests)}")
        if failed_tests:
            logger.info(f"Failed: {', '.join(failed_tests)}")
        
        # Verdetto finale
        if pass_rate >= 0.875:  # 7/8
            verdict = "EXCELLENT - Model predictions are high quality"
            logger.info(f"✓✓✓ {verdict}")
        elif pass_rate >= 0.75:  # 6/8
            verdict = "GOOD - Model predictions are acceptable"
            logger.info(f"✓✓ {verdict}")
        elif pass_rate >= 0.625:  # 5/8
            verdict = "ACCEPTABLE - Model predictions are marginally useful"
            logger.info(f"✓ {verdict}")
        elif pass_rate >= 0.5:  # 4/8
            verdict = "POOR - Model predictions are weak"
            logger.warning(f"⚠ {verdict}")
        else:
            verdict = "USELESS - Model predictions are not reliable"
            logger.error(f"✗ {verdict}")
        
        logger.info("="*80)
        
        return {
            'verdict': verdict,
            'pass_rate': pass_rate,
            'passed_count': passed_count,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests
        }
    
    def generate_report(self, filepath='outputs/prediction_quality_report.txt'):
        """Genera report dettagliato"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PREDICTION QUALITY TEST REPORT\n")
            f.write("="*80 + "\n\n")
            
            for test_name, test_results in self.results.items():
                if test_name == 'economic_significance':
                    continue
                f.write(f"\n{test_name.upper().replace('_', ' ')}\n")
                f.write("-"*40 + "\n")
                for key, value in test_results.items():
                    f.write(f"{key}: {value}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Report saved to {filepath}")


def test_model_predictions(system):
    """
    Testa le predizioni di un sistema già addestrato
    
    Args:
        system: StockPredictionSystem object con predizioni già generate
    """
    # Get test data and predictions
    y_true = system.test_data['Target_return_1d'].values
    
    # Use ensemble or best model
    if 'Ensemble' in system.predictions:
        y_pred = system.predictions['Ensemble']
        model_name = 'Ensemble'
    else:
        # Use first available model
        model_name = list(system.predictions.keys())[0]
        y_pred = system.predictions[model_name]
    
    logger.info(f"\nTesting predictions from: {model_name}")
    
    # Run tests
    tester = PredictionQualityTester(y_true, y_pred)
    results, verdict = tester.run_all_tests()
    
    # Generate report
    tester.generate_report()
    
    return results, verdict


if __name__ == '__main__':
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Simulate returns
    n = 1000
    true_returns = np.random.randn(n) * 0.02
    
    # Simulate predictions with some signal
    predictions = true_returns + np.random.randn(n) * 0.015
    
    print("\n" + "="*80)
    print("TESTING WITH SYNTHETIC DATA")
    print("="*80)
    
    tester = PredictionQualityTester(true_returns, predictions)
    results, verdict = tester.run_all_tests()
    
    print(f"\n\nFINAL VERDICT: {verdict['verdict']}")
    print(f"Pass Rate: {verdict['pass_rate']:.0%}")
