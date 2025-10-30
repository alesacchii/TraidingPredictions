#!/usr/bin/env python3
"""
ADVANCED FEATURE ENGINEERING - COMPLETO
Crea TUTTE le features necessarie in un unico posto
Include: base features + technical indicators + advanced patterns
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('MarketPredictor')


class AdvancedFeatureEngineer:
    """
    Feature engineering completo - crea TUTTE le features necessarie
    """

    def __init__(self, config):
        self.config = config
        self.feature_config = config.get('features', {})

    def create_all_features(self, data, market_data=None, economic_data=None, sentiment_data=None):
        """
        Crea TUTTE le features da zero
        """
        logger.info("Starting ADVANCED feature engineering...")

        df = data.copy()
        features_list = []

        for stock in df['Stock'].unique():
            logger.info(f"Creating features for {stock}...")
            stock_data = df[df['Stock'] == stock].copy()
            stock_data = stock_data.sort_values('Date').reset_index(drop=True)

            # FASE 1: BASE FEATURES
            stock_data = self._create_price_features(stock_data)
            stock_data = self._create_volume_features(stock_data)

            # FASE 2: TECHNICAL INDICATORS
            stock_data = self._create_moving_averages(stock_data)
            stock_data = self._create_momentum_indicators(stock_data)
            stock_data = self._create_volatility_indicators(stock_data)
            stock_data = self._create_trend_indicators(stock_data)

            # FASE 3: ADVANCED FEATURES
            stock_data = self._create_statistical_features(stock_data)
            stock_data = self._create_pattern_features(stock_data)
            stock_data = self._create_regime_features(stock_data)

            # Add market context if available
            if market_data is not None:
                stock_data = self._add_market_context(stock_data, market_data)

            features_list.append(stock_data)

        # Combine all stocks
        result = pd.concat(features_list, ignore_index=True)

        # Add economic data if available
        if economic_data is not None:
            result = self._merge_economic_data(result, economic_data)

        # Add sentiment if available
        if sentiment_data is not None:
            result = self._merge_sentiment_data(result, sentiment_data)

        # Create target variables
        result = self._create_targets(result)

        # Drop NaN
        initial_rows = len(result)
        result = result.dropna()
        dropped_rows = initial_rows - len(result)

        logger.info(f"Feature engineering complete. Created {len(result.columns)} columns")
        logger.info(f"Dropped {dropped_rows} rows with NaN values")
        logger.info(f"Final dataset: {len(result)} rows")

        return result

    # ============================================================================
    # FASE 1: BASE FEATURES
    # ============================================================================

    def _create_price_features(self, df):
        """Price-based features"""

        # Returns
        df['Returns_1d'] = df['Close'].pct_change(1)
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_20d'] = df['Close'].pct_change(20)
        df['Returns_60d'] = df['Close'].pct_change(60)

        # Log returns
        df['Log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Price momentum
        df['Price_momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Price_momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        df['Price_momentum_60'] = df['Close'] / df['Close'].shift(60) - 1

        # Price ratios
        df['High_Low_ratio'] = df['High'] / df['Low']
        df['Close_Open_ratio'] = df['Close'] / df['Open']

        # Gap
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Gap_percentage'] = df['Gap'] * 100

        return df

    def _create_volume_features(self, df):
        """Volume-based features"""

        # Volume moving averages
        for period in [5, 20, 60]:
            df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
            df[f'Volume_ratio_{period}'] = df['Volume'] / df[f'Volume_SMA_{period}']

        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_SMA_20'] = df['OBV'].rolling(window=20).mean()

        # Volume spike detection
        df['Volume_spike'] = (df['Volume'] > df['Volume_SMA_20'] * 2).astype(int)

        # Money Flow Index
        period = 14
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        df['MFI_14'] = mfi

        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['Close_to_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']

        return df

    # ============================================================================
    # FASE 2: TECHNICAL INDICATORS
    # ============================================================================

    def _create_moving_averages(self, df):
        """Moving averages"""

        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'Close_to_SMA_{period}'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']

        # Exponential Moving Averages
        for period in [12, 26, 50]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Golden Cross / Death Cross
        df['SMA_50_200_cross'] = np.sign(df['SMA_50'] - df['SMA_200'])

        return df

    def _create_momentum_indicators(self, df):
        """Momentum indicators"""

        # RSI
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        period = 14
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        df['STOCH_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['STOCH_d'] = df['STOCH_k'].rolling(window=3).mean()

        # Williams %R
        df['Williams_R'] = -100 * (high_max - df['Close']) / (high_max - low_min)

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100

        # Commodity Channel Index (CCI)
        period = 20
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI_20'] = (tp - sma_tp) / (0.015 * mad)

        return df

    def _create_volatility_indicators(self, df):
        """Volatility indicators"""

        # Bollinger Bands
        for period in [20, 50]:
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df[f'BB_upper_{period}'] = sma + (std * 2)
            df[f'BB_middle_{period}'] = sma
            df[f'BB_lower_{period}'] = sma - (std * 2)
            df[f'BB_width_{period}'] = (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}']) / df[f'BB_middle_{period}']
            df[f'BB_position_{period}'] = (df['Close'] - df[f'BB_lower_{period}']) / (
                    df[f'BB_upper_{period}'] - df[f'BB_lower_{period}'])

        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()
        df['ATR_ratio'] = df['ATR_14'] / df['Close']

        # Historical Volatility
        for period in [10, 20, 60]:
            df[f'Volatility_{period}'] = df['Returns_1d'].rolling(window=period).std() * np.sqrt(252)

        return df

    def _create_trend_indicators(self, df):
        """Trend indicators"""

        # Trend strength
        df['Trend_strength'] = np.abs(df['Close'] - df['SMA_20']) / (df['ATR_14'] + 1e-8)

        # Support and resistance levels
        window = 20
        df['Support_level'] = df['Low'].rolling(window=window).min()
        df['Resistance_level'] = df['High'].rolling(window=window).max()
        df['Price_to_support'] = (df['Close'] - df['Support_level']) / (df['Support_level'] + 1e-8)
        df['Price_to_resistance'] = (df['Resistance_level'] - df['Close']) / (df['Close'] + 1e-8)

        return df

    # ============================================================================
    # FASE 3: ADVANCED FEATURES
    # ============================================================================

    def _create_statistical_features(self, df):
        """Statistical features"""

        # Z-score of price
        for period in [20, 60]:
            mean = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df[f'Price_zscore_{period}'] = (df['Close'] - mean) / std

        # Skewness and Kurtosis
        for period in [20, 60]:
            df[f'Returns_skew_{period}'] = df['Returns_1d'].rolling(window=period).skew()
            df[f'Returns_kurt_{period}'] = df['Returns_1d'].rolling(window=period).kurt()

        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'Returns_autocorr_{lag}'] = df['Returns_1d'].rolling(window=20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )

        return df

    def _create_pattern_features(self, df):
        """Pattern recognition features"""

        # Fibonacci retracement levels
        df['Fib_0.382'] = df['Support_level'] + 0.382 * (df['Resistance_level'] - df['Support_level'])
        df['Fib_0.5'] = df['Support_level'] + 0.5 * (df['Resistance_level'] - df['Support_level'])
        df['Fib_0.618'] = df['Support_level'] + 0.618 * (df['Resistance_level'] - df['Support_level'])

        return df

    def _create_regime_features(self, df):
        """Market regime features"""

        # Volatility regime (low/medium/high)
        if 'Volatility_20' in df.columns:
            vol_20 = df['Volatility_20']
            vol_q33 = vol_20.rolling(window=252, min_periods=60).quantile(0.33)
            vol_q67 = vol_20.rolling(window=252, min_periods=60).quantile(0.67)

            df['Volatility_regime'] = 1  # medium by default
            df.loc[vol_20 < vol_q33, 'Volatility_regime'] = 0  # low
            df.loc[vol_20 > vol_q67, 'Volatility_regime'] = 2  # high
        else:
            df['Volatility_regime'] = 1

        # Trend regime (bull/bear/sideways) - ADESSO SMA_50 e SMA_200 ESISTONO
        df['Trend_Regime'] = np.sign(df['SMA_50'] - df['SMA_200'])

        # Volume regime
        df['Volume_Regime'] = np.where(df['Volume'] > df['Volume_SMA_20'], 1, 0)

        return df

    # ============================================================================
    # CONTEXT DATA
    # ============================================================================

    def _add_market_context(self, df, market_data):
        """Add market indices"""

        market_data = market_data.copy()

        if isinstance(market_data.columns, pd.MultiIndex):
            market_data.columns = [col[0] if isinstance(col, tuple) else col for col in market_data.columns]

        if isinstance(market_data.index, pd.MultiIndex):
            market_data = market_data.reset_index()

        if 'Date' not in market_data.columns:
            if 'index' in market_data.columns:
                market_data = market_data.rename(columns={'index': 'Date'})
            elif market_data.index.name == 'Date' or (
                    market_data.index.name and 'date' in str(market_data.index.name).lower()):
                market_data = market_data.reset_index()
            else:
                logger.warning("Market data has no Date column, skipping")
                return df

        df['Date'] = pd.to_datetime(df['Date'])
        market_data['Date'] = pd.to_datetime(market_data['Date'])

        market_data_cols = [col for col in market_data.columns if col != 'Date']
        if len(market_data_cols) == 0:
            logger.warning("Market data has no data columns, skipping")
            return df

        df = df.merge(market_data[['Date'] + market_data_cols], on='Date', how='left')

        if 'S&P 500' in df.columns:
            df['SPY_returns'] = df['S&P 500'].pct_change()
            df['Correlation_SPY'] = df['Returns_1d'].rolling(window=60, min_periods=30).corr(df['SPY_returns'])

        if 'VIX' in df.columns:
            df['VIX_change'] = df['VIX'].pct_change()

        return df

    def _merge_economic_data(self, df, economic_data):
        """Merge economic indicators"""

        if isinstance(economic_data.index, pd.MultiIndex):
            economic_data = economic_data.reset_index()

        if 'Date' not in economic_data.columns:
            if economic_data.index.name == 'Date' or 'date' in str(economic_data.index.name).lower():
                economic_data = economic_data.reset_index()
            else:
                logger.warning("Economic data has no Date column, skipping")
                return df

        df['Date'] = pd.to_datetime(df['Date'])
        economic_data['Date'] = pd.to_datetime(economic_data['Date'])

        df = df.merge(economic_data, on='Date', how='left')

        econ_cols = [col for col in economic_data.columns if col != 'Date']
        if econ_cols:
            df[econ_cols] = df[econ_cols].ffill()

        return df

    def _merge_sentiment_data(self, df, sentiment_data):
        """Add sentiment from news"""

        try:
            from textblob import TextBlob

            sentiment_data['Sentiment'] = sentiment_data.apply(
                lambda row: TextBlob(str(row['Title']) + ' ' + str(row['Description'])).sentiment.polarity,
                axis=1
            )

            daily_sentiment = sentiment_data.groupby(['Stock', 'Date']).agg({
                'Sentiment': ['mean', 'std', 'count']
            }).reset_index()

            daily_sentiment.columns = ['Stock', 'Date', 'Sentiment_mean', 'Sentiment_std', 'News_count']

            df['Date'] = pd.to_datetime(df['Date'])
            daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

            df = df.merge(daily_sentiment, on=['Stock', 'Date'], how='left', suffixes=('', '_news'))

            if 'Sentiment_mean' in df.columns:
                df['Sentiment_mean'] = df['Sentiment_mean'].fillna(0)
            if 'Sentiment_std' in df.columns:
                df['Sentiment_std'] = df['Sentiment_std'].fillna(0)
            if 'News_count' in df.columns:
                df['News_count'] = df['News_count'].fillna(0)

            logger.info("Sentiment features added")

        except ImportError:
            logger.warning("textblob not available, skipping sentiment")
        except Exception as e:
            logger.warning(f"Error processing sentiment: {e}")

        return df

    def _create_targets(self, df):
        """Create target variables"""

        horizons = self.config.get('prediction', {}).get('horizons', [1, 5, 20])

        for horizon in horizons:
            df[f'Target_return_{horizon}d'] = df.groupby('Stock')['Close'].shift(-horizon) / df['Close'] - 1
            df[f'Target_price_{horizon}d'] = df.groupby('Stock')['Close'].shift(-horizon)
            df[f'Target_direction_{horizon}d'] = (df[f'Target_return_{horizon}d'] > 0).astype(int)
            df[f'Target_sharpe_{horizon}d'] = df[f'Target_return_{horizon}d'] / df['Volatility_20']

        return df

    def get_feature_columns(self, df):
        """Get list of feature columns (exclude targets and metadata)"""

        exclude_patterns = ['Target_', 'Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']

        feature_cols = [col for col in df.columns
                        if not any(pattern in col for pattern in exclude_patterns)]

        return feature_cols