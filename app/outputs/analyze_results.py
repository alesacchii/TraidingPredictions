#!/usr/bin/env python3
"""
PROFESSIONAL RESULTS ANALYZER
Dashboard interattivo per analisi completa delle predizioni
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import json


class PredictionAnalyzer:
    """Analyzer professionale con grafici interattivi"""
    
    def __init__(self, results_dir='outputs'):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.load_all_data()
        
    def load_all_data(self):
        """Carica tutti i file di output"""
        print("📊 Loading results...")
        
        # Predictions
        pred_file = self.results_dir / 'predictions.csv'
        if pred_file.exists():
            self.data['predictions'] = pd.read_csv(pred_file)
            print(f"✓ Loaded predictions: {len(self.data['predictions'])} samples")
        
        # Future predictions
        future_file = self.results_dir / 'future_predictions.csv'
        if future_file.exists():
            self.data['future'] = pd.read_csv(future_file)
            print(f"✓ Loaded future predictions: {len(self.data['future'])} stocks")
        
        # Model metrics
        metrics_file = self.results_dir / 'model_metrics.csv'
        if metrics_file.exists():
            self.data['metrics'] = pd.read_csv(metrics_file, index_col=0)
            print(f"✓ Loaded metrics: {len(self.data['metrics'])} models")
        
        # Feature importance
        for model in ['XGBoost', 'LightGBM']:
            feat_file = self.results_dir / f'feature_importance_{model}.csv'
            if feat_file.exists():
                self.data[f'features_{model}'] = pd.read_csv(feat_file)
                print(f"✓ Loaded {model} features: {len(self.data[f'features_{model}'])} features")
        
        # Features data (sample per velocità)
        features_file = self.results_dir / 'features_data.csv'
        if features_file.exists():
            # Leggi solo prime 10000 righe per velocità
            self.data['features_full'] = pd.read_csv(features_file, nrows=10000)
            print(f"✓ Loaded features data (sample): {len(self.data['features_full'])} rows")
        
        # Quality report
        report_file = self.results_dir / 'prediction_quality_report.txt'
        if report_file.exists():
            with open(report_file, 'r') as f:
                self.data['quality_report'] = f.read()
            print(f"✓ Loaded quality report")
        
        print(f"\n✅ Total files loaded: {len(self.data)}")
    
    def create_overview_dashboard(self):
        """Dashboard principale con overview completo"""
        print("\n📈 Creating overview dashboard...")
        
        if 'metrics' not in self.data:
            print("❌ No metrics data available")
            return None
        
        metrics = self.data['metrics']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance (RMSE)', 
                          'R² Score Comparison',
                          'MAE vs RMSE Trade-off',
                          'Model Metrics Summary'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'table'}]]
        )
        
        # 1. RMSE Comparison
        models = metrics.index.tolist()
        rmse_values = metrics['RMSE'].values
        
        colors = ['red' if r > 0.03 else 'orange' if r > 0.02 else 'green' 
                  for r in rmse_values]
        
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, 
                   marker_color=colors,
                   name='RMSE',
                   text=[f'{v:.6f}' for v in rmse_values],
                   textposition='outside'),
            row=1, col=1
        )
        
        # 2. R² Score
        r2_values = metrics['R2'].values
        colors_r2 = ['green' if r > 0.2 else 'orange' if r > 0.1 else 'red' 
                     for r in r2_values]
        
        fig.add_trace(
            go.Bar(x=models, y=r2_values,
                   marker_color=colors_r2,
                   name='R²',
                   text=[f'{v:.4f}' for v in r2_values],
                   textposition='outside'),
            row=1, col=2
        )
        
        # 3. MAE vs RMSE scatter
        mae_values = metrics['MAE'].values
        
        fig.add_trace(
            go.Scatter(x=mae_values, y=rmse_values,
                      mode='markers+text',
                      marker=dict(size=15, color=colors),
                      text=models,
                      textposition='top center',
                      name='Models'),
            row=2, col=1
        )
        
        # 4. Metrics table
        table_data = metrics.round(6).reset_index()
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(table_data.columns),
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[table_data[col] for col in table_data.columns],
                          fill_color='lavender',
                          align='left')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="🎯 Model Performance Dashboard",
            showlegend=False,
            height=1000,
            title_font_size=20
        )
        
        # Add reference lines using shapes instead of add_hline (avoid subplot issues)
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(models)-0.5, y0=0.02, y1=0.02,
            line=dict(color="green", width=2, dash="dash"),
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(models)-0.5, y0=0.2, y1=0.2,
            line=dict(color="green", width=2, dash="dash"),
            row=1, col=2
        )
        
        return fig
    
    def create_predictions_analysis(self):
        """Analisi dettagliata delle predizioni"""
        print("\n🔍 Creating predictions analysis...")
        
        if 'predictions' not in self.data or 'features_full' not in self.data:
            print("❌ Missing data for predictions analysis")
            return None
        
        preds = self.data['predictions']
        features = self.data['features_full']
        
        # Merge per avere actual values
        if 'Target_return_1d' in features.columns:
            df = features[['Date', 'Stock', 'Target_return_1d']].copy()
            df = df.tail(len(preds))  # Prendi solo test set
            df['Predicted'] = preds.iloc[:, 0].values[:len(df)]  # Prima colonna predizioni
            df['Error'] = df['Target_return_1d'] - df['Predicted']
        else:
            print("⚠️ No target column found, using predictions only")
            df = pd.DataFrame({
                'Predicted': preds.iloc[:, 0].values
            })
        
        # Create figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Predictions vs Actual Returns',
                          'Prediction Error Distribution',
                          'Cumulative Prediction Error',
                          'Prediction Scatter Plot',
                          'Error Over Time',
                          'Prediction Histogram'),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'histogram'}]]
        )
        
        if 'Target_return_1d' in df.columns:
            # 1. Time series comparison
            sample_size = min(200, len(df))
            sample_df = df.tail(sample_size)
            
            fig.add_trace(
                go.Scatter(x=list(range(len(sample_df))), 
                          y=sample_df['Target_return_1d'].values,
                          name='Actual', mode='lines',
                          line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=list(range(len(sample_df))),
                          y=sample_df['Predicted'].values,
                          name='Predicted', mode='lines',
                          line=dict(color='red', dash='dash')),
                row=1, col=1
            )
            
            # 2. Error distribution
            fig.add_trace(
                go.Histogram(x=df['Error'].values, nbinsx=50,
                            name='Error Distribution',
                            marker_color='orange'),
                row=1, col=2
            )
            
            # 3. Cumulative error
            cumulative_error = df['Error'].cumsum().values
            fig.add_trace(
                go.Scatter(x=list(range(len(cumulative_error))),
                          y=cumulative_error,
                          name='Cumulative Error',
                          mode='lines',
                          line=dict(color='purple')),
                row=2, col=1
            )
            
            # 4. Scatter plot
            fig.add_trace(
                go.Scatter(x=df['Target_return_1d'].values,
                          y=df['Predicted'].values,
                          mode='markers',
                          marker=dict(size=5, color='green', opacity=0.5),
                          name='Predictions'),
                row=2, col=2
            )
            
            # Perfect prediction line
            min_val = min(df['Target_return_1d'].min(), df['Predicted'].min())
            max_val = max(df['Target_return_1d'].max(), df['Predicted'].max())
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          mode='lines',
                          line=dict(color='red', dash='dash'),
                          name='Perfect Prediction'),
                row=2, col=2
            )
            
            # 5. Error over time
            fig.add_trace(
                go.Scatter(x=list(range(len(df))),
                          y=df['Error'].abs().values,
                          mode='markers',
                          marker=dict(size=3, color='red', opacity=0.3),
                          name='Absolute Error'),
                row=3, col=1
            )
        
        # 6. Prediction histogram
        fig.add_trace(
            go.Histogram(x=df['Predicted'].values, nbinsx=50,
                        name='Predictions',
                        marker_color='blue'),
            row=3, col=2
        )
        
        fig.update_layout(
            title_text="📊 Detailed Predictions Analysis",
            showlegend=True,
            height=1200,
            title_font_size=20
        )
        
        return fig
    
    def create_feature_importance_chart(self):
        """Grafici feature importance"""
        print("\n🔬 Creating feature importance charts...")
        
        # Check available models
        available_models = []
        for model in ['XGBoost', 'LightGBM']:
            if f'features_{model}' in self.data:
                available_models.append(model)
        
        if not available_models:
            print("❌ No feature importance data available")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=len(available_models), cols=1,
            subplot_titles=[f'{model} - Top 20 Features' for model in available_models]
        )
        
        for idx, model in enumerate(available_models, 1):
            features_df = self.data[f'features_{model}'].head(20)
            
            # Reverse order for better visualization
            features_df = features_df.iloc[::-1]
            
            fig.add_trace(
                go.Bar(
                    x=features_df['importance'].values,
                    y=features_df['feature'].values,
                    orientation='h',
                    name=model,
                    marker_color='lightblue' if model == 'XGBoost' else 'lightgreen',
                    text=features_df['importance'].round(2),
                    textposition='outside'
                ),
                row=idx, col=1
            )
        
        fig.update_layout(
            title_text="🎯 Feature Importance Analysis",
            showlegend=True,
            height=400 * len(available_models),
            title_font_size=20
        )
        
        return fig
    
    def create_future_predictions_chart(self):
        """Grafici predizioni future"""
        print("\n🔮 Creating future predictions chart...")
        
        if 'future' not in self.data:
            print("❌ No future predictions available")
            return None
        
        future = self.data['future'].sort_values('Predicted_Return', ascending=False)
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('📈 Top Stocks by Predicted Return',
                          '💰 Expected Price Changes'),
            specs=[[{'type': 'bar'}], [{'type': 'bar'}]]
        )
        
        # Color based on return
        colors = ['green' if r > 0 else 'red' for r in future['Predicted_Return'].values]
        
        # 1. Returns
        fig.add_trace(
            go.Bar(
                x=future['Stock'].values,
                y=(future['Predicted_Return'].values * 100),
                marker_color=colors,
                name='Predicted Return %',
                text=[f'{v:.2f}%' for v in future['Predicted_Return'].values * 100],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Price changes
        price_change = future['Predicted_Price'] - future['Current_Price']
        colors_price = ['green' if p > 0 else 'red' for p in price_change.values]
        
        fig.add_trace(
            go.Bar(
                x=future['Stock'].values,
                y=price_change.values,
                marker_color=colors_price,
                name='Price Change ($)',
                text=[f'${v:.2f}' for v in price_change.values],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title_text="🔮 Future Predictions Dashboard",
            showlegend=False,
            height=800,
            title_font_size=20
        )
        
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Price Change ($)", row=2, col=1)
        
        return fig
    
    def create_quality_report_visual(self):
        """Visualizzazione grafica del quality report"""
        print("\n📋 Creating quality report visualization...")
        
        if 'quality_report' not in self.data:
            print("❌ No quality report available")
            return None
        
        # Parse report
        report_text = self.data['quality_report']
        
        # Extract metrics manually
        metrics_dict = {}
        current_test = None
        
        for line in report_text.split('\n'):
            if line.strip() and not line.startswith('=') and not line.startswith('-'):
                if ':' in line and 'TEST' not in line.upper():
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert to number
                    try:
                        if value == 'True':
                            value = 1.0
                        elif value == 'False':
                            value = 0.0
                        else:
                            value = float(value)
                        
                        if current_test:
                            metrics_dict[f"{current_test}_{key}"] = value
                    except:
                        pass
                elif 'TEST' in line.upper() and ':' in line:
                    current_test = line.split(':')[0].strip().replace('TEST ', '').replace(':', '')
        
        # Create radar chart for test results
        tests = ['BASIC METRICS', 'NAIVE COMPARISON', 'DIRECTIONAL ACCURACY',
                'STATISTICAL SIGNIFICANCE', 'RESIDUAL ANALYSIS', 'PREDICTION STABILITY',
                'OUTLIER HANDLING']
        
        passed_tests = []
        for test in tests:
            key = f"{test}_passed"
            if key in metrics_dict:
                passed_tests.append(metrics_dict[key])
            else:
                # Try to find in report
                if 'passed: True' in report_text and test in report_text:
                    passed_tests.append(1.0)
                else:
                    passed_tests.append(0.0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=passed_tests,
            theta=tests,
            fill='toself',
            name='Test Results',
            line_color='blue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="✅ Quality Tests Results (1=Pass, 0=Fail)",
            height=600,
            title_font_size=20
        )
        
        return fig
    
    def generate_full_report(self, output_file='prediction_analysis.html'):
        """Genera report HTML completo"""
        print("\n" + "="*80)
        print("📊 GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        # Create all figures
        figures = {}
        
        fig1 = self.create_overview_dashboard()
        if fig1:
            figures['overview'] = fig1
        
        fig2 = self.create_predictions_analysis()
        if fig2:
            figures['predictions'] = fig2
        
        fig3 = self.create_feature_importance_chart()
        if fig3:
            figures['features'] = fig3
        
        fig4 = self.create_future_predictions_chart()
        if fig4:
            figures['future'] = fig4
        
        fig5 = self.create_quality_report_visual()
        if fig5:
            figures['quality'] = fig5
        
        if not figures:
            print("❌ No figures created - check data files")
            return None
        
        # Create HTML report
        html_parts = ['''
<!DOCTYPE html>
<html>
<head>
    <title>Stock Prediction Analysis Report</title>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        h1 {
            color: #667eea;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        h2 {
            color: #764ba2;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        .summary {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #667eea;
        }
        .metric {
            display: inline-block;
            background: white;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            min-width: 200px;
        }
        .metric-label {
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        .good { color: #28a745; }
        .bad { color: #dc3545; }
        .warning { color: #ffc107; }
        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Stock Market Prediction Analysis</h1>
        <p style="text-align: center; color: #666; font-size: 1.2em;">
            Comprehensive Performance Report
        </p>
''']
        
        # Add summary section
        if 'metrics' in self.data:
            metrics = self.data['metrics']
            best_model = metrics['R2'].idxmax()
            best_r2 = metrics['R2'].max()
            best_rmse = metrics['RMSE'].min()
            
            html_parts.append(f'''
        <div class="summary">
            <h2>📈 Executive Summary</h2>
            <div style="text-align: center;">
                <div class="metric">
                    <div class="metric-label">Best Model</div>
                    <div class="metric-value">{best_model}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best R²</div>
                    <div class="metric-value {'good' if best_r2 > 0.2 else 'bad'}">{best_r2:.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best RMSE</div>
                    <div class="metric-value {'good' if best_rmse < 0.02 else 'bad'}">{best_rmse:.6f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Models Trained</div>
                    <div class="metric-value">{len(metrics)}</div>
                </div>
            </div>
        </div>
''')
        
        # Add charts
        for name, fig in figures.items():
            html_parts.append(f'<div class="chart-container">')
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            html_parts.append('</div>')
        
        # Add footer
        html_parts.append('''
        <div class="footer">
            <p>Generated by Professional Stock Prediction Analyzer</p>
            <p style="font-size: 0.9em; color: #999;">
                ⚠️ Disclaimer: Past performance does not guarantee future results. 
                This is for educational purposes only.
            </p>
        </div>
    </div>
</body>
</html>
''')
        
        # Save HTML
        output_path = self.results_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))
        
        print(f"\n✅ Report generated: {output_path}")
        print(f"📄 Open in browser to view interactive dashboard")
        
        return output_path
    
    def print_summary(self):
        """Stampa summary testuale"""
        print("\n" + "="*80)
        print("📊 RESULTS SUMMARY")
        print("="*80)
        
        if 'metrics' in self.data:
            print("\n🎯 MODEL PERFORMANCE:")
            print(self.data['metrics'].round(6).to_string())
        
        if 'future' in self.data:
            print("\n🔮 TOP 5 FUTURE PREDICTIONS:")
            future = self.data['future'].sort_values('Predicted_Return', ascending=False).head(5)
            print(future[['Stock', 'Current_Price', 'Predicted_Price', 'Predicted_Return']].to_string(index=False))
        
        print("\n" + "="*80)


def main():
    """Main execution"""
    import sys
    
    # Get results directory
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'outputs'
    
    print(f"\n🚀 Starting Professional Results Analyzer")
    print(f"📁 Results directory: {results_dir}")
    
    # Create analyzer
    analyzer = PredictionAnalyzer(results_dir)
    
    # Print summary
    analyzer.print_summary()
    
    # Generate full report
    report_path = analyzer.generate_full_report()
    
    if report_path:
        print(f"\n✅ SUCCESS!")
        print(f"📊 Interactive report: {report_path}")
        print(f"\n💡 Open this file in your web browser for interactive charts")
    else:
        print(f"\n❌ Failed to generate report")


if __name__ == '__main__':
    main()
