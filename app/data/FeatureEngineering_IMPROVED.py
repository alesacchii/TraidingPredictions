"""
Improved Feature Engineering Module
Fixes scaling issues and adds more predictive features
"""

import pandas as pd
import numpy as np
import talib as ta
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class ImprovedFeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.feature_names = None
        
    def create_base_features(self, df):
        """Create robust base features"""
        features = pd.DataFrame(index=df.index)
        
        for stock in df['Stock'].unique():
            mask = df['Stock'] == stock
            stock_data = df[mask].copy()
            
            # Price-based features with proper scaling
            features.loc[mask, f'{stock}_returns_1d'] = stock_data['Close'].pct_change()
            features.loc[mask, f'{stock}_returns_2d'] = stock_data['Close'].pct_change(2)
            features.loc[mask, f'{stock}_returns_5d'] = stock_data['Close'].pct_change(5)
            features.loc[mask, f'{stock}_returns_10d'] = stock_data['Close'].pct_change(10)
            features.loc[mask, f'{stock}_returns_20d'] = stock_data['Close'].pct_change(20)
            
            # Log returns (more stable)
            features.loc[mask, f'{stock}_log_return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
            
            # Volatility features
            features.loc[mask, f'{stock}_volatility_5d'] = features.loc[mask, f'{stock}_returns_1d'].rolling(5).std()
            features.loc[mask, f'{stock}_volatility_20d'] = features.loc[mask, f'{stock}_returns_1d'].rolling(20).std()
            
            # Price relative to moving averages
            sma_20 = stock_data['Close'].rolling(20).mean()
            sma_50 = stock_data['Close'].rolling(50).mean()
            features.loc[mask, f'{stock}_price_sma20_ratio'] = (stock_data['Close'] - sma_20) / sma_20
            features.loc[mask, f'{stock}_price_sma50_ratio'] = (stock_data['Close'] - sma_50) / sma_50
            
            # Volume features (normalized)
            features.loc[mask, f'{stock}_volume_ratio'] = stock_data['Volume'] / stock_data['Volume'].rolling(20).mean()
            features.loc[mask, f'{stock}_volume_change'] = stock_data['Volume'].pct_change()
            
            # Price patterns
            features.loc[mask, f'{stock}_high_low_ratio'] = (stock_data['High'] - stock_data['Low']) / stock_data['Close']
            features.loc[mask, f'{stock}_close_open_ratio'] = (stock_data['Close'] - stock_data['Open']) / stock_data['Open']
            
        return features

    def create_technical_indicators(self, df):
        """Create technical indicators with proper normalization"""
        features = pd.DataFrame(index=df.index)

        for stock in df['Stock'].unique():
            mask = df['Stock'] == stock
            stock_data = df[mask].copy()

            # Convert to float64 (fix for TA-Lib input)
            high = stock_data['High'].astype(float).values
            low = stock_data['Low'].astype(float).values
            close = stock_data['Close'].astype(float).values
            volume = stock_data['Volume'].astype(float).values

            # RSI (already normalized 0-100)
            rsi = ta.RSI(close, timeperiod=14)
            features.loc[mask, f'{stock}_rsi'] = (rsi - 50) / 50  # Center around 0

            # MACD
            macd, signal, hist = ta.MACD(close)
            features.loc[mask, f'{stock}_macd_hist'] = hist / close  # Normalize by price

            # Bollinger Bands
            upper, middle, lower = ta.BBANDS(close)
            features.loc[mask, f'{stock}_bb_position'] = (close - middle) / (upper - middle)
            features.loc[mask, f'{stock}_bb_width'] = (upper - lower) / middle

            # Stochastic
            slowk, slowd = ta.STOCH(high, low, close)
            features.loc[mask, f'{stock}_stoch_k'] = (slowk - 50) / 50

            # ATR
            atr = ta.ATR(high, low, close, timeperiod=14)
            features.loc[mask, f'{stock}_atr_ratio'] = atr / close

            # Money Flow Index (MFI)
            mfi = ta.MFI(high, low, close, volume, timeperiod=14)
            features.loc[mask, f'{stock}_mfi'] = (mfi - 50) / 50

            # Williams %R
            willr = ta.WILLR(high, low, close, timeperiod=14)
            features.loc[mask, f'{stock}_williams_r'] = willr / 50  # Normalize to -2 to 0

        return features

    def create_market_features(self, df, market_data):
        """Create market-relative features"""
        features = pd.DataFrame(index=df.index)
        
        if market_data is None:
            return features
            
        for stock in df['Stock'].unique():
            mask = df['Stock'] == stock
            stock_data = df[mask].copy()
            
            # Market correlation features
            if 'sp500' in market_data:
                sp500_returns = market_data['sp500'].pct_change()
                stock_returns = stock_data['Close'].pct_change()
                
                # Rolling correlation
                features.loc[mask, f'{stock}_sp500_corr_20d'] = \
                    stock_returns.rolling(20).corr(sp500_returns)
                
                # Relative strength
                features.loc[mask, f'{stock}_relative_strength'] = \
                    stock_returns.rolling(20).mean() - sp500_returns.rolling(20).mean()
            
            # VIX features
            if 'vix' in market_data:
                vix_norm = (market_data['vix'] - market_data['vix'].rolling(20).mean()) / \
                           market_data['vix'].rolling(20).std()
                features.loc[mask, f'{stock}_vix_norm'] = vix_norm
                
        return features
    
    def create_advanced_features(self, df, features):
        """Create interaction and non-linear features"""
        advanced = pd.DataFrame(index=df.index)
        
        for stock in df['Stock'].unique():
            mask = df['Stock'] == stock
            
            # Momentum features
            if f'{stock}_returns_5d' in features.columns and f'{stock}_returns_20d' in features.columns:
                advanced.loc[mask, f'{stock}_momentum_ratio'] = \
                    features.loc[mask, f'{stock}_returns_5d'] / (features.loc[mask, f'{stock}_returns_20d'] + 0.0001)
            
            # Volatility regime
            if f'{stock}_volatility_5d' in features.columns and f'{stock}_volatility_20d' in features.columns:
                advanced.loc[mask, f'{stock}_vol_regime'] = \
                    features.loc[mask, f'{stock}_volatility_5d'] / (features.loc[mask, f'{stock}_volatility_20d'] + 0.0001)
            
            # Technical indicator combinations
            if f'{stock}_rsi' in features.columns and f'{stock}_stoch_k' in features.columns:
                advanced.loc[mask, f'{stock}_rsi_stoch_avg'] = \
                    (features.loc[mask, f'{stock}_rsi'] + features.loc[mask, f'{stock}_stoch_k']) / 2
            
        return advanced
    
    def scale_features(self, features, fit=False):
        """Apply robust scaling to features"""
        scaled_features = features.copy()
        
        for col in features.columns:
            if col not in self.scalers:
                if fit:
                    self.scalers[col] = RobustScaler()
                    # Fit on non-NaN values
                    valid_data = features[col].dropna()
                    if len(valid_data) > 0:
                        self.scalers[col].fit(valid_data.values.reshape(-1, 1))
                else:
                    continue  # Skip if scaler doesn't exist
            
            # Transform
            valid_mask = ~features[col].isna()
            if valid_mask.any() and col in self.scalers:
                scaled_values = self.scalers[col].transform(
                    features.loc[valid_mask, col].values.reshape(-1, 1)
                )
                scaled_features.loc[valid_mask, col] = scaled_values.flatten()
        
        return scaled_features

    def engineer_features(self, df, market_data=None, economic_data=None, sentiment_data=None, fit_scalers=False):
        """Main feature engineering pipeline estesa"""
        # 1. Feature base e tecniche
        base_features = self.create_base_features(df)
        tech_features = self.create_technical_indicators(df)
        market_features = self.create_market_features(df, market_data)
        econ_features = self.create_economic_features(df, economic_data)
        sentiment_features = self.create_sentiment_features(df, sentiment_data)

        # 2. Combina tutto
        all_features = pd.concat(
            [base_features, tech_features, market_features, econ_features, sentiment_features],
            axis=1
        )

        # Add advanced features
        advanced_features = self.create_advanced_features(df, all_features)
        all_features = pd.concat([all_features, advanced_features], axis=1)

        # --- FIX 1: more tolerant NaN filtering ---
        nan_threshold = 0.85  # allow more NaNs to survive
        nan_ratio = all_features.isna().sum() / len(all_features)
        valid_columns = nan_ratio[nan_ratio < nan_threshold].index
        all_features = all_features[valid_columns]

        # --- FIX 2: safety check ---
        if all_features.empty:
            print("⚠️ Warning: No valid features after filtering. Check rolling windows or NaN handling.")
            # fallback: keep unfiltered base features
            all_features = base_features.fillna(method='ffill').fillna(method='bfill')

        # Fill remaining NaNs
        all_features = all_features.fillna(method='ffill').fillna(method='bfill')

        # Scale features
        scaled_features = self.scale_features(all_features, fit=fit_scalers)

        # Store feature names
        self.feature_names = scaled_features.columns.tolist()

        # Add back identifiers
        scaled_features['Date'] = df['Date']
        scaled_features['Stock'] = df['Stock']

        # Add targets
        for stock in df['Stock'].unique():
            mask = df['Stock'] == stock
            stock_data = df[mask]

            for horizon in [1, 5, 20]:
                scaled_features.loc[mask, f'Target_return_{horizon}d'] = \
                    stock_data['Close'].pct_change(horizon).shift(-horizon)

        # --- FIX 3: log columns count ---
        print(f"✅ Engineered features: {len(self.feature_names)} columns retained.")

        return scaled_features

    def get_feature_columns(self, df):
        """Get list of feature columns"""
        exclude = ['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']
        exclude += [col for col in df.columns if 'Target' in col]
        return [col for col in df.columns if col not in exclude]

    def create_economic_features(self, df, economic_data):
        """Crea feature macroeconomiche e collineate con le date dei prezzi"""
        features = pd.DataFrame(index=df.index)
        if economic_data is None or len(economic_data) == 0:
            return features

        # Allinea le date economiche al dataset principale
        econ_df = economic_data.copy()
        econ_df['Date'] = pd.to_datetime(econ_df['Date'])
        econ_df = econ_df.sort_values('Date')

        for stock in df['Stock'].unique():
            mask = df['Stock'] == stock
            stock_data = df[mask].copy()

            # Merge temporale per data più vicina
            merged = pd.merge_asof(
                stock_data.sort_values('Date'),
                econ_df.sort_values('Date'),
                on='Date',
                direction='backward'
            )

            for col in econ_df.columns:
                if col != 'Date':
                    features.loc[mask, f'{stock}_econ_{col}'] = merged[col]

        return features


    def create_sentiment_features(self, df, sentiment_data):
        """Crea feature di sentiment derivate da dati testuali/news"""
        features = pd.DataFrame(index=df.index)
        if sentiment_data is None or len(sentiment_data) == 0:
            return features

        sent_df = sentiment_data.copy()
        sent_df['Date'] = pd.to_datetime(sent_df['Date'])
        sent_df = sent_df.sort_values('Date')

        # Normalizza eventuali colonne note di sentiment
        sentiment_cols = [c for c in sent_df.columns if c.lower() in ['sentiment', 'sentiment_score', 'compound']]
        if not sentiment_cols:
            return features  # nessuna colonna utile

        sentiment_col = sentiment_cols[0]

        for stock in df['Stock'].unique():
            mask = df['Stock'] == stock
            stock_data = df[mask].copy()

            merged = pd.merge_asof(
                stock_data.sort_values('Date'),
                sent_df.sort_values('Date'),
                on='Date',
                direction='backward'
            )

            # Feature di base
            features.loc[mask, f'{stock}_sentiment'] = merged[sentiment_col]

            # Media mobile per stabilità
            features.loc[mask, f'{stock}_sentiment_ma3'] = merged[sentiment_col].rolling(3).mean()
            features.loc[mask, f'{stock}_sentiment_ma7'] = merged[sentiment_col].rolling(7).mean()

        return features
