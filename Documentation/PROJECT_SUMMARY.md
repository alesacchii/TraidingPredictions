# SISTEMA DI PREDIZIONE MERCATI AZIONARI
## Multi-Model Stock Market Prediction System

---

## 📦 CONTENUTO PACCHETTO

Hai ricevuto un sistema completo di predizione per mercati azionari con:

### File Core (16 files)
```
✓ main.py                    - Orchestratore principale
✓ config_advanced.yaml       - Configurazione sistema
✓ requirements.txt           - Dipendenze Python

✓ Logger_config.py           - Sistema logging
✓ DownloadMarketData.py      - Download dati (yfinance + context data)
✓ FeatureEngineering.py      - 50+ indicatori tecnici
✓ TreeModels.py              - XGBoost & LightGBM
✓ LSTMModel.py               - LSTM neural network
✓ TimesFMModel.py            - Google TimesFM integration
✓ EnsembleModel.py           - Meta-ensemble optimizer
✓ Backtesting.py             - Sistema backtesting completo

✓ demo.py                    - Demo veloce (2 min)
✓ examples.py                - 6 esempi pratici

✓ setup.sh                   - Script installazione
✓ README.md                  - Documentazione completa
✓ QUICKSTART.md              - Guida rapida
```

---

## 🎯 COSA FA IL SISTEMA

### Input
- Lista di stock symbols (es. AAPL, GOOGL, TSLA)
- Periodo storico (es. 2018-oggi)
- Configurazione modelli

### Processing
1. **Download dati** da yfinance (+ opzionali: news, macro-economia)
2. **Feature engineering**: 50+ indicatori tecnici automatici
3. **Training**: 4 modelli ML (XGBoost, LightGBM, LSTM, TimesFM)
4. **Ensemble**: Ottimizzazione pesi per combinare modelli
5. **Backtesting**: Valutazione su dati storici con metriche trading

### Output
- **Predizioni future** (1, 5, 20 giorni ahead)
- **Confidence intervals** per ogni predizione
- **Performance metrics** (R², RMSE, Sharpe, Win Rate, etc.)
- **Feature importance** (quali indicatori contano di più)
- **Backtest results** (quanto avrebbe guadagnato la strategia)

---

## 🚀 COME INIZIARE

### Metodo 1: Demo Veloce (2 minuti)
```bash
./setup.sh
python3 demo.py
```

### Metodo 2: Sistema Completo
```bash
./setup.sh
python3 main.py
```

### Metodo 3: Esempi Pratici
```bash
./setup.sh
python3 examples.py
# Scegli uno dei 6 esempi
```

---

## 💪 PUNTI DI FORZA

### 1. Multi-Model Ensemble
- Non un singolo modello, ma **4 modelli combinati**
- Pesi ottimizzati automaticamente
- Riduce overfitting, aumenta robustezza

### 2. Feature Engineering Professionale
- 50+ indicatori tecnici
- Volume analysis
- Volatility regimes
- Support/resistance detection
- Cross-asset correlations

### 3. Backtesting Rigoroso
- Walk-forward analysis (no look-ahead bias)
- Transaction costs realistici
- Multiple strategie testabili
- Metriche complete (Sharpe, Sortino, Drawdown, Win Rate)

### 4. Production-Ready
- Modulare e estensibile
- Logging completo
- Error handling robusto
- Configurazione YAML
- Documentazione estesa

### 5. TimesFM Integration
- Foundation model di Google
- Zero-shot forecasting
- State-of-the-art per time series

---

## 📊 STRUTTURA MODULARE

```
Sistema
├── Data Layer
│   ├── MarketDataDownloader  (yfinance)
│   ├── News API              (sentiment)
│   └── FRED API              (macro-economics)
│
├── Feature Layer
│   └── FeatureEngineer       (50+ indicators)
│
├── Model Layer
│   ├── XGBoost               (gradient boosting)
│   ├── LightGBM              (fast gradient boosting)
│   ├── LSTM                  (deep learning)
│   └── TimesFM               (foundation model)
│
├── Ensemble Layer
│   └── EnsembleModel         (optimized weights)
│
├── Evaluation Layer
│   └── Backtester            (trading simulation)
│
└── Orchestration
    └── StockPredictionSystem (main pipeline)
```

---

## 🎓 CUSTOMIZZAZIONE

### Cambiare Stock
```yaml
# config_advanced.yaml
data_download:
  stocks_list:
    - AAPL
    - TUOI_STOCK
```

### Disabilitare Modelli Lenti
```yaml
models:
  lstm:
    enabled: false
  timesfm:
    enabled: false
```

### Modificare Horizon
```yaml
prediction:
  horizons: [1, 5, 20]  # giorni in futuro
```

### Strategia di Trading
```yaml
backtesting:
  strategies:
    - threshold_based
    - top_k
    - portfolio_optimization
```

---

## 📈 PERFORMANCE REALISTICHE

### Aspettative
- **R² Score**: 0.15 - 0.35 (mercati sono difficili!)
- **Sharpe Ratio**: 1.0 - 2.5
- **Win Rate**: 52% - 58%
- **Annual Return**: 10% - 25%

### Nota Importante
I mercati finanziari sono **intrinsecamente difficili** da predire:
- R² di 0.20+ è considerato **eccellente**
- Nessun sistema garantisce profitti
- Past performance ≠ future results
- Usa sempre risk management

---

## 🛠️ REQUISITI TECNICI

### Software
- Python 3.8+
- 4GB+ RAM (8GB+ raccomandato per LSTM)
- Internet (per download dati)

### Hardware (opzionale)
- GPU: Accelera LSTM e TimesFM
- CPU: Funziona anche solo CPU (più lento)

### API Keys (opzionali)
- NEWS_API_KEY (sentiment da news)
- FRED_API_KEY (dati macro-economici)

**Il sistema funziona SENZA API keys**, ma con features ridotte.

---

## 📚 FILE DA LEGGERE

### Priorità 1 (Inizia qui)
1. **QUICKSTART.md** - Setup in 5 minuti
2. **README.md** - Documentazione completa

### Priorità 2 (Approfondimenti)
3. **demo.py** - Codice semplice per capire il flow
4. **examples.py** - 6 casi d'uso pratici

### Priorità 3 (Customizzazione)
5. **config_advanced.yaml** - Tutte le configurazioni
6. Codice sorgente (*.py) - Per modifiche avanzate

---

## 🔥 QUICK WINS

### Win #1: Predizioni Immediate (2 min)
```bash
python3 demo.py
```

### Win #2: Analizza Features Importanti
```python
from examples import example_4_feature_analysis
example_4_feature_analysis()
```

### Win #3: Compara Modelli
```python
from examples import example_6_model_comparison
example_6_model_comparison()
```

---

## ⚠️ DISCLAIMER LEGALE

**QUESTO SISTEMA È PER SCOPI EDUCATIVI/DI RICERCA**

- ❌ NON è consiglio finanziario
- ❌ NON garantisce profitti
- ❌ I mercati sono imprevedibili
- ✅ Usa per imparare e ricercare
- ✅ Testa con capitale virtuale prima
- ✅ Consulta sempre professionisti

**Non investire denaro che non puoi permetterti di perdere.**

---

## 🎯 PROSSIMI PASSI

### Oggi
1. ✅ Esegui `./setup.sh`
2. ✅ Prova `python3 demo.py`
3. ✅ Leggi QUICKSTART.md

### Questa Settimana
4. ✅ Esegui sistema completo: `python3 main.py`
5. ✅ Analizza output in `outputs/`
6. ✅ Prova esempi: `python3 examples.py`

### Questo Mese
7. ✅ Customizza configurazione
8. ✅ Testa diverse strategie
9. ✅ Ottimizza per i tuoi stock preferiti
10. ✅ Backtest su periodi diversi

---

## 💡 TIPS PRO

### Tip #1: Inizia Semplice
- Prima: solo 1-2 stock, solo XGBoost
- Poi: aggiungi stock e modelli gradualmente

### Tip #2: Analizza Feature Importance
- Scopri QUALI indicatori funzionano meglio
- Elimina features inutili per velocità

### Tip #3: Backtest su Periodi Diversi
- Bull market (2020-2021)
- Bear market (2022)
- Sideways market (2015-2016)

### Tip #4: Usa Ensemble
- Sempre meglio di singoli modelli
- Più robusto e stabile

### Tip #5: Transaction Costs
- Non ignorarli nei backtest!
- Commission + slippage cambiano tutto

---

## 🆘 SUPPORTO

### Problemi Comuni

**Sistema lento?**
→ Disabilita LSTM e TimesFM

**Predizioni inaccurate?**
→ Normale! R² 0.20+ è già ottimo

**Out of memory?**
→ Riduci sequence_length, usa meno stock

**Download fallisce?**
→ Controlla internet, yfinance serve web

---

## 📦 PACKAGE SUMMARY

```
Total Files:       16
Total Size:        ~150KB
Lines of Code:     ~3,500
Models:            4 (XGBoost, LightGBM, LSTM, TimesFM)
Features:          50+ technical indicators
Strategies:        3 backtesting strategies
Documentation:     Complete (README + QUICKSTART)
Examples:          7 (demo + 6 practical examples)
```

---

## 🏆 CARATTERISTICHE UNICHE

✅ **Multi-model ensemble** (non singolo modello)
✅ **TimesFM integration** (Google foundation model)
✅ **50+ technical indicators** automatici
✅ **Backtesting rigoroso** con transaction costs
✅ **Confidence intervals** per ogni predizione
✅ **Production-ready** code quality
✅ **Completamente documentato**
✅ **Modulare ed estensibile**

---

## 🎓 PER CHI È QUESTO SISTEMA?

### ✅ Perfetto per:
- Quant traders che vogliono automazione
- Data scientists interessati a finance
- Studenti/ricercatori in ML/Finance
- Chi vuole capire feature engineering finanziario
- Developer che costruiscono trading systems

### ⚠️ Non per:
- Chi cerca "soldi facili" garantiti
- Chi non ha esperienza programmazione
- Chi non capisce i rischi del trading
- Chi non ha capitale per testare (usa virtuale!)

---

## 🔮 POSSIBILI ESTENSIONI

Il sistema è progettato per essere esteso:

- [ ] Altri modelli (Transformer, Prophet, GRU)
- [ ] Real-time predictions (streaming data)
- [ ] Dashboard web (Streamlit/Plotly)
- [ ] Options data integration
- [ ] Social media sentiment
- [ ] Portfolio optimization avanzata
- [ ] AutoML hyperparameter tuning
- [ ] Cloud deployment (AWS/GCP/Azure)

---

## ✨ HAI RICEVUTO

Un sistema **professionale, completo, documentato** per predizione mercati azionari.

**Non un tutorial**, ma un **sistema production-ready** che puoi:
- Usare subito
- Studiare per imparare
- Estendere per i tuoi bisogni
- Integrare in progetti più grandi

---

## 🚀 INIZIA ORA

```bash
# 3 comandi per iniziare:
./setup.sh
python3 demo.py
python3 main.py
```

**Buon trading e buona ricerca! 📈**

---

*"In investing, what is comfortable is rarely profitable." - Robert Arnott*
