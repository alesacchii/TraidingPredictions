#!/usr/bin/env python3
"""
ADVANCED FEATURE ENGINEERING
Migliora drasticamente la capacità predittiva con features avanzate
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler


class AdvancedFeatureEngineer:
    """
    Feature engineer avanzato con:
    - Market regime detection
    - Volatility clustering
    - Microstructure features
    - Non-linear transformations
    """

    def __init__(self, config):
        self.config = config
        self.scaler = RobustScaler()  # Più robusto agli outliers

    def create_market_regime_features(self, df):
        """
        CRITICO: Detecta bull/bear/sideways market
        Il modello deve comportarsi diversamente in regimi diversi
        """
        print("Creating market regime features...")

        # Trend strength (ADX-like)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()

        df['ADX'] = adx
        df['Trend_Strength'] = np.where(adx > 25, 1, 0)  # 1 = trending

        # Regime classification
        sma_20 = df['Close'].rolling(20).mean()
        sma_50 = df['Close'].rolling(50).mean()
        sma_200 = df['Close'].rolling(200).mean()

        # Bull: price > all SMAs and SMAs in order
        bull_condition = (df['Close'] > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200)
        # Bear: price < all SMAs
        bear_condition = (df['Close'] < sma_20) & (sma_20 < sma_50) & (sma_50 < sma_200)

        df['Market_Regime'] = 0  # Sideways
        df.loc[bull_condition, 'Market_Regime'] = 1  # Bull
        df.loc[bear_condition, 'Market_Regime'] = -1  # Bear

        # Regime change detection
        df['Regime_Change'] = df['Market_Regime'].diff().abs()

        return df

    def create_volatility_clustering_features(self, df):
        """
        CRITICO: Volatilità tende a clusterizzare
        High vol → High vol, Low vol → Low vol
        """
        print("Creating volatility clustering features...")

        returns = df['Close'].pct_change()

        # Multiple volatility windows
        for window in [5, 10, 20, 60]:
            # Realized volatility
            df[f'RealizedVol_{window}'] = returns.rolling(window).std() * np.sqrt(252)

            # Parkinson volatility (usa High/Low)
            hl_ratio = np.log(df['High'] / df['Low'])
            df[f'ParkinsonVol_{window}'] = (hl_ratio.rolling(window).var() / (4 * np.log(2))) ** 0.5 * np.sqrt(252)

            # Volatility of volatility
            df[f'VolOfVol_{window}'] = df[f'RealizedVol_{window}'].rolling(window).std()

        # Volatility regime (persistent high/low vol)
        vol_20 = df['RealizedVol_20']
        vol_q33 = vol_20.rolling(252).quantile(0.33)
        vol_q67 = vol_20.rolling(252).quantile(0.67)

        df['Vol_Regime_Low'] = (vol_20 < vol_q33).astype(int)
        df['Vol_Regime_High'] = (vol_20 > vol_q67).astype(int)

        # Volatility spike detection
        vol_ma = vol_20.rolling(20).mean()
        vol_std = vol_20.rolling(20).std()
        df['Vol_Spike'] = ((vol_20 - vol_ma) / vol_std > 2).astype(int)

        return df

    def create_microstructure_features(self, df):
        """
        Features da microstructura del mercato
        Cattura price action intraday
        """
        print("Creating microstructure features...")

        # Price range
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']

        # Open-Close relationship
        df['Body_Size'] = np.abs(df['Close'] - df['Open']) / df['Close']
        df['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']

        # Candlestick patterns (simplified)
        body = df['Close'] - df['Open']
        df['Doji'] = (np.abs(body) / df['High_Low_Range'] < 0.1).astype(int)
        df['Hammer'] = ((df['Lower_Shadow'] > 2 * df['Body_Size']) & (body > 0)).astype(int)
        df['Shooting_Star'] = ((df['Upper_Shadow'] > 2 * df['Body_Size']) & (body < 0)).astype(int)

        # Price action momentum
        df['Momentum_Quality'] = df['Close'].diff() / df['High_Low_Range']

        # Tick direction (proxy)
        df['Tick_Direction'] = np.sign(df['Close'].diff())
        df['Tick_Persistence'] = df['Tick_Direction'].rolling(10).sum() / 10

        return df

    def create_non_linear_features(self, df):
        """
        Trasformazioni non-lineari per catturare pattern complessi
        """
        print("Creating non-linear features...")

        returns = df['Close'].pct_change()

        # Returns powers (cattura skewness)
        df['Returns_Squared'] = returns ** 2
        df['Returns_Cubed'] = returns ** 3

        # Log transformations
        df['Log_Volume'] = np.log1p(df['Volume'])
        df['Log_High_Low'] = np.log(df['High'] / df['Low'])

        # Interaction features
        if 'RSI_14' in df.columns and 'MACD' in df.columns:
            df['RSI_MACD_Interaction'] = df['RSI_14'] * df['MACD']

        if 'BB_position_20' in df.columns and 'RealizedVol_20' in df.columns:
            df['BB_Vol_Interaction'] = df['BB_position_20'] * df['RealizedVol_20']

        # Cyclical encoding for time features
        if 'Date' in df.columns:
            df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
            df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

            df['DayOfMonth'] = pd.to_datetime(df['Date']).dt.day
            df['DayOfMonth_Sin'] = np.sin(2 * np.pi * df['DayOfMonth'] / 31)
            df['DayOfMonth_Cos'] = np.cos(2 * np.pi * df['DayOfMonth'] / 31)

        return df

    def create_statistical_moments_features(self, df):
        """
        Momenti statistici higher-order
        """
        print("Creating statistical moments features...")

        returns = df['Close'].pct_change()

        for window in [20, 60]:
            # Skewness (asymmetry)
            df[f'Skewness_{window}'] = returns.rolling(window).skew()

            # Kurtosis (tail risk)
            df[f'Kurtosis_{window}'] = returns.rolling(window).kurt()

            # Quantiles
            df[f'Quantile_25_{window}'] = returns.rolling(window).quantile(0.25)
            df[f'Quantile_75_{window}'] = returns.rolling(window).quantile(0.75)
            df[f'IQR_{window}'] = df[f'Quantile_75_{window}'] - df[f'Quantile_25_{window}']

        return df

    def create_momentum_quality_features(self, df):
        """
        Qualità del momentum - non solo direzione
        """
        print("Creating momentum quality features...")

        returns = df['Close'].pct_change()

        # Consecutive moves
        df['Consecutive_Ups'] = (returns > 0).astype(int)
        df['Consecutive_Ups_Count'] = df['Consecutive_Ups'].groupby(
            (df['Consecutive_Ups'] != df['Consecutive_Ups'].shift()).cumsum()
        ).cumsum()

        df['Consecutive_Downs'] = (returns < 0).astype(int)
        df['Consecutive_Downs_Count'] = df['Consecutive_Downs'].groupby(
            (df['Consecutive_Downs'] != df['Consecutive_Downs'].shift()).cumsum()
        ).cumsum()

        # Momentum acceleration
        for period in [5, 20]:
            momentum = df['Close'].pct_change(period)
            df[f'Momentum_Acceleration_{period}'] = momentum.diff()

        # Price distance from extremes
        df['Distance_From_52w_High'] = (df['High'].rolling(252).max() - df['Close']) / df['Close']
        df['Distance_From_52w_Low'] = (df['Close'] - df['Low'].rolling(252).min()) / df['Close']

        return df

    def create_volume_profile_features(self, df):
        """
        Analisi del profilo volume - dove si concentra il trading
        """
        print("Creating volume profile features...")

        # Volume-weighted features
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3

        for window in [20, 60]:
            # VWAP
            df[f'VWAP_{window}'] = (typical_price * df['Volume']).rolling(window).sum() / df['Volume'].rolling(
                window).sum()
            df[f'Price_To_VWAP_{window}'] = (df['Close'] - df[f'VWAP_{window}']) / df[f'VWAP_{window}']

            # Volume profile (accumulation/distribution)
            price_change = df['Close'].diff()
            volume_signed = df['Volume'] * np.sign(price_change)
            df[f'Volume_Profile_{window}'] = volume_signed.rolling(window).sum()

        # Relative volume
        df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Volume_Breakout'] = (df['Relative_Volume'] > 2).astype(int)

        return df

    def create_all_advanced_features(self, df):
        """
        Crea TUTTE le features avanzate
        """
        print("\n" + "=" * 80)
        print("CREATING ADVANCED FEATURES")
        print("=" * 80)

        df = df.copy()

        # Apply all feature creation methods
        df = self.create_market_regime_features(df)
        df = self.create_volatility_clustering_features(df)
        df = self.create_microstructure_features(df)
        df = self.create_non_linear_features(df)
        df = self.create_statistical_moments_features(df)
        df = self.create_momentum_quality_features(df)
        df = self.create_volume_profile_features(df)

        # Count new features
        new_features = [col for col in df.columns if col not in
                        ['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']]

        print(f"\n✅ Created {len(new_features)} advanced features")
        print("=" * 80)

        return df


def integrate_with_existing_feature_engineer(original_df, config):
    """
    Funzione helper per integrare con FeatureEngineer esistente

    Usage:
        # Nel tuo main.py o FeatureEngineering.py:
        from FeatureEngineering_IMPROVED import integrate_with_existing_feature_engineer

        # Dopo aver creato features base:
        features_data = self.feature_engineer.create_all_features(stock_data, ...)

        # Aggiungi features avanzate:
        features_data = integrate_with_existing_feature_engineer(features_data, config)
    """
    advanced_engineer = AdvancedFeatureEngineer(config)

    # Process each stock separately
    enhanced_dfs = []

    for stock in original_df['Stock'].unique():
        stock_df = original_df[original_df['Stock'] == stock].copy()
        stock_df = stock_df.sort_values('Date').reset_index(drop=True)

        # Add advanced features
        enhanced_df = advanced_engineer.create_all_advanced_features(stock_df)
        enhanced_dfs.append(enhanced_df)

    # Combine
    result = pd.concat(enhanced_dfs, ignore_index=True)

    return result


# ============================================================================
# ESEMPIO DI INTEGRAZIONE NEL MAIN.PY
# ============================================================================

"""
# Nel tuo main.py, modifica il metodo create_features:

def create_features(self):
    logger.info("-" * 80)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("-" * 80)

    # Features base (esistenti)
    self.features_data = self.feature_engineer.create_all_features(
        self.raw_data['stock_data'],
        market_data=self.raw_data['market_indices'],
        economic_data=self.raw_data['economic_data'],
        sentiment_data=self.raw_data['news_data']
    )

    # 🆕 NUOVO: Aggiungi features avanzate
    logger.info("\\n--- Adding Advanced Features ---")
    from FeatureEngineering_IMPROVED import integrate_with_existing_feature_engineer

    self.features_data = integrate_with_existing_feature_engineer(
        self.features_data, 
        self.config
    )

    logger.info(f"\\nTotal features: {self.features_data.shape[1]}")
    logger.info("Feature engineering complete\\n")
"""

if __name__ == '__main__':
    print("Advanced Feature Engineering Module")
    print("=" * 80)
    print("This module adds critical features that improve prediction accuracy:")
    print("- Market regime detection")
    print("- Volatility clustering")
    print("- Microstructure features")
    print("- Non-linear transformations")
    print("- Statistical moments")
    print("- Momentum quality")
    print("- Volume profiles")
    print("=" * 80)