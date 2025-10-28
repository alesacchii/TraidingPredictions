# Stock Market Prediction System

## Sistema Avanzato di Predizione Multi-Modello per Mercati Azionari

Sistema end-to-end che combina **4 modelli di machine learning** (XGBoost, LightGBM, LSTM, TimesFM) in un ensemble ottimizzato per prevedere l'andamento futuro dei mercati azionari.

---

## 🎯 Caratteristiche Principali

### Modelli Implementati
1. **XGBoost** - Pattern non-lineari e feature importance
2. **LightGBM** - Velocità e robustezza 
3. **LSTM** - Dipendenze temporali profonde
4. **TimesFM** - Foundation model di Google per time series (zero-shot forecasting)
5. **Meta-Ensemble** - Combina tutti i modelli con pesi ottimizzati

### Feature Engineering Avanzato
- **50+ indicatori tecnici**: SMA, EMA, MACD, RSI, Bollinger Bands, ATR, Stochastic, Williams %R, CCI, ROC, MFI, VWAP, OBV
- **Analisi volume**: Volume ratios, spikes, money flow
- **Volatilità**: Multiple timeframes, regimi
- **Pattern recognition**: Support/resistance, Fibonacci retracements
- **Cross-asset correlation**: SPY, VIX, market indices
- **Sentiment analysis**: News headlines (opzionale)
- **Dati macro-economici**: Fed funds, Treasury yields, Oil prices (opzionale)

### Backtesting Rigoroso
- Walk-forward analysis
- Multiple strategie: threshold-based, top-K, portfolio optimization
- Metriche complete: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor
- Transaction costs (commission + slippage)

---

## 📦 Installazione

### Requisiti
- Python 3.8+
- pip

### Setup Rapido

```bash
# 1. Installa le dipendenze
./setup.sh
# oppure
pip install -r requirements.txt

# 2. (Opzionale) Configura API keys per dati aggiuntivi
export NEWS_API_KEY='4bc2d29abd264882838ba008ed99ba33'
export FRED_API_KEY='eb51fe0cd83954a2364b69ebd5363581'

# 3. Esegui il sistema
python3 main.py
```

---

## 🚀 Quick Start

### Esecuzione Base

```python
from main import StockPredictionSystem

# Inizializza il sistema
system = StockPredictionSystem(config_path='config_advanced.yaml')

# Esegui pipeline completa
metrics = system.run_full_pipeline()

# Predizioni future
predictions = system.predict_future(days_ahead=5)
```

### Configurazione Custom

Modifica `config_advanced.yaml` per:
- Cambiare stocks (`stocks_list`)
- Aggiungere/rimuovere modelli (`models.*.enabled`)
- Modificare parametri di training
- Configurare strategie di backtesting

---

## 📊 Architettura del Sistema

```
┌─────────────────┐
│  Data Download  │ ← yfinance, FRED API, News API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feature Engineer │ ← 50+ technical indicators
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Model Training             │
│  ┌──────────┐  ┌──────────┐   │
│  │ XGBoost  │  │ LightGBM │   │
│  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐   │
│  │   LSTM   │  │ TimesFM  │   │
│  └──────────┘  └──────────┘   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Meta-Ensemble   │ ← Optimized weights
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Predictions    │ ← With confidence intervals
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Backtesting    │ ← Multiple strategies
└─────────────────┘
```

---

## 📈 Output e Risultati

Il sistema genera:

### File Output (cartella `outputs/`)
- `model_metrics.csv` - Performance di ogni modello
- `predictions.csv` - Predizioni complete
- `future_predictions.csv` - Predizioni out-of-sample
- `feature_importance_*.csv` - Importanza delle features
- `features_data.csv` - Dataset completo con features

### Metriche Valutazione
- **Regression**: RMSE, MAE, R²
- **Trading**: Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, Profit Factor
- **Ensemble**: Model contributions, optimized weights

### Log Dettagliato
Tutte le operazioni sono loggiate in `logs/prediction_system.log`

---

## 🔧 Configurazione Avanzata

### Modelli

```yaml
models:
  xgboost:
    enabled: true
    params:
      n_estimators: 500
      max_depth: 7
      learning_rate: 0.05
  
  timesfm:
    enabled: true
    model_name: "google/timesfm-2.5-200m-pytorch"
    params:
      context_length: 512
      horizon_length: 30
```

### Feature Engineering

```yaml
features:
  technical_indicators: [...]
  sentiment:
    enabled: true
  macro_economic:
    enabled: true
```

### Backtesting

```yaml
backtesting:
  enabled: true
  initial_capital: 100000
  commission: 0.001  # 0.1%
  strategies:
    - threshold_based
    - top_k
```

---

## 📝 Esempi d'Uso

### 1. Predizione Singola

```python
# Predizioni per i prossimi 5 giorni
future = system.predict_future(days_ahead=5)
print(future[['Stock', 'Current_Price', 'Predicted_Price', 'Predicted_Return']])
```

### 2. Analisi Feature Importance

```python
# Top 20 features più importanti
xgb_model = system.models['XGBoost']
importance = xgb_model.get_feature_importance(top_n=20)
print(importance)
```

### 3. Confronto Modelli

```python
# Confronta performance di tutti i modelli
comparison = system.ensemble.compare_models(X_test, y_test)
print(comparison.sort_values('RMSE'))
```

### 4. Backtesting Custom

```python
from Backtesting import Backtester

backtester = Backtester(system.config)
metrics, results = backtester.run_backtest(
    data=test_data,
    predictions=predictions,
    strategy='threshold_based',
    threshold=0.02,  # 2% minimo
    hold_days=3
)
```

---

## 🧪 Performance Attese

### Metriche Tipiche (variano per dataset)
- **R²**: 0.15 - 0.35 (mercati sono difficili!)
- **Sharpe Ratio**: 1.0 - 2.5
- **Win Rate**: 52% - 58%
- **Max Drawdown**: -15% - -25%

**Nota**: I mercati finanziari sono intrinsecamente difficili da predire. Un R² di 0.20+ è considerato molto buono nel contesto del trading.

---

## 🔍 Troubleshooting

### TimesFM non carica
```bash
# Installa il package ufficiale se disponibile
pip install timesfm

# Oppure disabilita in config
models:
  timesfm:
    enabled: false
```

### Out of Memory con LSTM
```yaml
# Riduci sequence length e batch size
models:
  lstm:
    params:
      sequence_length: 30  # invece di 60
      batch_size: 16       # invece di 32
```

### API Keys mancanti
Le API keys sono opzionali. Il sistema funziona senza, ma con features ridotte:
- Senza NEWS_API_KEY: no sentiment analysis
- Senza FRED_API_KEY: no macro-economic indicators

---

## 📚 Struttura File

```
.
├── config_advanced.yaml      # Configurazione principale
├── requirements.txt          # Dipendenze Python
├── setup.sh                  # Script di installazione
│
├── main.py                   # Orchestratore principale
├── Logger_config.py          # Sistema di logging
├── DownloadMarketData.py     # Download dati
├── FeatureEngineering.py     # Creazione features
├── TreeModels.py             # XGBoost & LightGBM
├── LSTMModel.py              # LSTM neural network
├── TimesFMModel.py           # Google TimesFM integration
├── EnsembleModel.py          # Meta-ensemble
├── Backtesting.py            # Sistema di backtesting
│
└── outputs/                  # Risultati generati
    ├── model_metrics.csv
    ├── predictions.csv
    ├── future_predictions.csv
    └── ...
```

---

## ⚠️ Disclaimer

**IMPORTANTE**: Questo sistema è per scopi educativi e di ricerca. 

- NON è un consiglio finanziario
- Le performance passate NON garantiscono risultati futuri
- I mercati sono imprevedibili e comportano rischi significativi
- Consulta sempre un consulente finanziario professionale
- Non investire denaro che non puoi permetterti di perdere

---

## 🎓 Risorse e Link

### Modelli
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [Google TimesFM](https://huggingface.co/google/timesfm-2.5-200m-pytorch)

### Data Sources
- [yfinance](https://github.com/ranaroussi/yfinance)
- [FRED API](https://fred.stlouisfed.org/docs/api/)
- [News API](https://newsapi.org/)

### Indicatori Tecnici
- [TA-Lib](https://github.com/mrjbq7/ta-lib)
- [pandas-ta](https://github.com/twopirllc/pandas-ta)

---

## 🤝 Contributi

Sistema progettato per essere modulare ed estensibile.

Per aggiungere un nuovo modello:
1. Crea una classe in un nuovo file (es. `MyModel.py`)
2. Implementa i metodi: `train()`, `predict()`, `evaluate()`
3. Aggiungi la configurazione in `config_advanced.yaml`
4. Registra il modello in `main.py`

---

## 📜 Licenza

Open source - Usa a tuo rischio e pericolo. Vedi disclaimer sopra.

---

## 🔮 Roadmap Future

- [ ] Additional models: Transformer, GRU, Prophet
- [ ] Real-time predictions con streaming data
- [ ] Dashboard interattiva (Streamlit/Plotly)
- [ ] Portfolio optimization avanzata (Kelly Criterion, Black-Litterman)
- [ ] Alternative data sources (social media, options flow)
- [ ] AutoML per hyperparameter tuning
- [ ] Deployment su cloud (AWS, GCP, Azure)

---

**Developed with focus on accuracy, robustness, and real-world applicability.**

*"The stock market is a device for transferring money from the impatient to the patient." - Warren Buffett*
