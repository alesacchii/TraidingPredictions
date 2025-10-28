# QUICK START GUIDE
# Stock Market Prediction System

## 🚀 Avvio Rapido in 3 Passi

### Passo 1: Installazione
```bash
# Clona/scarica il progetto
cd stock-prediction-system

# Installa dipendenze
./setup.sh
# OPPURE
pip install -r requirements.txt
```

### Passo 2: Configurazione Base
```bash
# Il sistema è pronto! 
# La configurazione di default in config_advanced.yaml è già ottimizzata

# OPZIONALE: Personalizza gli stock
# Modifica config_advanced.yaml, sezione data_download.stocks_list:
# stocks_list:
#   - AAPL
#   - GOOGL
#   - TUOI_STOCK_PREFERITI
```

### Passo 3: Esecuzione
```bash
# Demo rapida (2 minuti)
python3 demo.py

# Sistema completo
python3 main.py

# Esempi pratici
python3 examples.py
```

---

## 📊 Primo Uso - Scenario Tipico

### Vuoi predire AAPL per domani?

```python
from main import StockPredictionSystem

# 1. Setup
system = StockPredictionSystem('config_advanced.yaml')

# 2. Esegui pipeline
system.run_full_pipeline()

# 3. Ottieni predizioni
predictions = system.predict_future(days_ahead=1)
print(predictions)

# Output:
# Stock  Current_Price  Predicted_Return_%  Predicted_Price
# AAPL   175.43        +1.23               177.59
```

---

## ⚙️ Personalizzazione Rapida

### Cambiare Stock
```yaml
# config_advanced.yaml
data_download:
  stocks_list:
    - AAPL
    - TSLA
    - NVDA
```

### Disabilitare Modelli Lenti
```yaml
# config_advanced.yaml
models:
  lstm:
    enabled: false  # Disabilita LSTM (più lento)
  timesfm:
    enabled: false  # Disabilita TimesFM (richiede GPU)
```

### Solo Predizioni Veloci (no backtest)
```yaml
# config_advanced.yaml
backtesting:
  enabled: false
```

---

## 🎯 Cosa Aspettarsi

### Output Files (cartella `outputs/`)
1. `model_metrics.csv` - Performance modelli
2. `predictions.csv` - Predizioni complete
3. `future_predictions.csv` - Predizioni out-of-sample
4. `feature_importance_*.csv` - Features importanti

### Tempo di Esecuzione (stima)
- Demo (2 stock, 2 anni): **~2 minuti**
- Full system (5 stock, 7 anni): **~10-15 minuti**
- Con LSTM: **+5-10 minuti**
- Con TimesFM: **+10-20 minuti** (GPU) / **+30+ minuti** (CPU)

### Performance Attese
- **R²**: 0.15 - 0.35 (i mercati sono difficili!)
- **Sharpe Ratio**: 1.0 - 2.5
- **Win Rate**: 52% - 58%

---

## 🔧 Troubleshooting Rapido

### Errore: "Module not found"
```bash
# Reinstalla dipendenze
pip install -r requirements.txt --upgrade
```

### Errore: "Out of memory" con LSTM
```yaml
# Riduci batch size e sequence length
models:
  lstm:
    params:
      sequence_length: 30  # era 60
      batch_size: 16       # era 32
```

### TimesFM non carica
```yaml
# Disabilita TimesFM
models:
  timesfm:
    enabled: false
```

### Download dati fallisce
```python
# Controlla connessione internet
# yfinance richiede connessione per scaricare dati storici
# Se offline: usa dati pre-scaricati (vedi esempio sotto)
```

---

## 📱 API Keys (Opzionali)

### News Sentiment (opzionale)
```bash
# 1. Registrati su https://newsapi.org/
# 2. Ottieni API key gratuita
# 3. Esporta variabile ambiente:
export NEWS_API_KEY='your_api_key_here'
```

### Dati Macro-Economici (opzionale)
```bash
# 1. Registrati su https://fred.stlouisfed.org/
# 2. Richiedi API key
# 3. Esporta variabile ambiente:
export FRED_API_KEY='your_fred_key_here'
```

**NOTA**: Il sistema funziona SENZA API keys, ma con features ridotte.

---

## 💡 Tips Pro

### 1. Salva Modelli Addestrati
```python
# Salva dopo training
system.models['XGBoost'].save_model('my_xgb_model.json')

# Ricarica successivamente
model = TreeBasedModels(config, 'xgboost')
model.load_model('my_xgb_model.json')
```

### 2. Predizioni Veloci Giornaliere
```bash
# Usa solo XGBoost per velocità
# Disabilita backtesting
# Usa dati recenti (start_date: '2023-01-01')
python3 examples.py  # Scegli opzione 3
```

### 3. Analisi Features Importanti
```python
# Scopri quali indicatori tecnici funzionano meglio
xgb_model = system.models['XGBoost']
importance = xgb_model.get_feature_importance(top_n=30)
print(importance)
```

### 4. Backtesting Custom
```python
from Backtesting import Backtester

backtester = Backtester(config)
metrics, results = backtester.run_backtest(
    data=your_data,
    predictions=your_predictions,
    strategy='threshold_based',
    threshold=0.015,  # Compra solo con +1.5% atteso
    hold_days=3       # Tieni per 3 giorni
)
```

---

## 📚 Prossimi Passi

### Dopo il Primo Run
1. ✅ Controlla `outputs/model_metrics.csv` per performance
2. ✅ Analizza `feature_importance_*.csv` per capire cosa conta
3. ✅ Testa diverse strategie di backtesting
4. ✅ Ottimizza hyperparameters se necessario

### Approfondimenti
- Leggi `README.md` per dettagli completi
- Esplora `examples.py` per casi d'uso avanzati
- Modifica `config_advanced.yaml` per personalizzazione

---

## ⚠️ DISCLAIMER

**IMPORTANTE**: 
- Questo è un sistema educativo/di ricerca
- NON è consiglio finanziario
- I mercati sono imprevedibili
- Non investire denaro che non puoi perdere
- Consulta sempre un professionista

---

## 🆘 Supporto

### Problemi Comuni

**Q: Il sistema è lento**
A: Disabilita LSTM e TimesFM, usa solo XGBoost/LightGBM

**Q: Predizioni poco accurate**
A: Normale! R² 0.20+ è già ottimo per i mercati azionari

**Q: Errori durante download**
A: Controlla connessione internet, yfinance richiede accesso web

**Q: Out of memory**
A: Riduci sequence_length LSTM, usa meno stock, o più RAM

---

## 🎓 Risorse

- [yfinance Docs](https://github.com/ranaroussi/yfinance)
- [XGBoost Guide](https://xgboost.readthedocs.io/)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)

---

**Buon Trading! 📈**

*"The market is a device for transferring money from the impatient to the patient." - Warren Buffett*
