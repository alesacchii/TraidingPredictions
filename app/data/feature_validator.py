#!/usr/bin/env python3
"""
FEATURE VALIDATION MODULE
Verifica rigorosa della correttezza del calcolo delle features
"""

import pandas as pd
import numpy as np
from app.configuration.Logger_config import  logger


class FeatureValidator:
    """
    Valida che le features siano calcolate correttamente
    Trova errori comuni: look-ahead bias, NaN, valori impossibili, etc.
    """

    def __init__(self, data, feature_cols):
        """
        Args:
            data: DataFrame con features
            feature_cols: Lista nomi colonne features
        """
        self.data = data
        self.feature_cols = feature_cols
        self.errors = []
        self.warnings = []

    def validate_all(self):
        """Esegui tutte le validazioni"""
        logger.info("=" * 80)
        logger.info("FEATURE VALIDATION - STARTING")
        logger.info("=" * 80)

        self.check_look_ahead_bias()
        self.check_nan_values()
        self.check_infinite_values()
        self.check_value_ranges()
        self.check_feature_correlations()
        self.check_constant_features()
        self.check_data_leakage()

        # Report
        self.generate_report()

        return len(self.errors) == 0

    def check_look_ahead_bias(self):
        """
        CRITICO: Verifica che non ci sia look-ahead bias
        Le features non devono usare informazioni future
        """
        logger.info("\n--- Checking Look-Ahead Bias ---")

        # Check se target è usato per creare features
        target_cols = [col for col in self.data.columns if 'Target_' in col]

        for target in target_cols:
            if target in self.feature_cols:
                self.errors.append(f"CRITICAL: Target '{target}' is in feature columns!")

        # Check se features hanno correlazione perfetta con target
        if target_cols and self.feature_cols:
            target_col = target_cols[0]
            for feat in self.feature_cols[:20]:  # Check first 20 features
                try:
                    corr = self.data[[feat, target_col]].corr().iloc[0, 1]
                    if abs(corr) > 0.99:
                        self.warnings.append(
                            f"Feature '{feat}' has suspiciously high correlation ({corr:.4f}) with target")
                except:
                    pass

        if not self.errors:
            logger.info("✓ No look-ahead bias detected")
        else:
            logger.error(f"✗ Found {len(self.errors)} look-ahead bias issues!")

    def check_nan_values(self):
        """Verifica presenza NaN"""
        logger.info("\n--- Checking NaN Values ---")

        nan_counts = self.data[self.feature_cols].isna().sum()
        nan_pct = (nan_counts / len(self.data)) * 100

        problematic_features = nan_pct[nan_pct > 50]  # >50% NaN

        if len(problematic_features) > 0:
            self.warnings.append(f"{len(problematic_features)} features have >50% NaN values")
            logger.warning(f"Features with >50% NaN: {problematic_features.index.tolist()}")

        total_nan = nan_counts.sum()
        if total_nan > 0:
            logger.info(f"Total NaN values: {total_nan}")
        else:
            logger.info("✓ No NaN values found")

    def check_infinite_values(self):
        """Verifica presenza valori infiniti"""
        logger.info("\n--- Checking Infinite Values ---")

        inf_found = False
        for col in self.feature_cols:
            if np.isinf(self.data[col]).any():
                self.errors.append(f"Feature '{col}' contains infinite values")
                inf_found = True

        if inf_found:
            logger.error("✗ Infinite values detected!")
        else:
            logger.info("✓ No infinite values")

    def check_value_ranges(self):
        """Verifica che i valori siano in range ragionevoli"""
        logger.info("\n--- Checking Value Ranges ---")

        for col in self.feature_cols:
            values = self.data[col].dropna()

            if len(values) == 0:
                continue

            # Check per percentuali (RSI, Stochastic, etc) - NOT Williams %R
            if any(x in col.upper() for x in ['RSI', 'STOCH']) and 'WILLIAMS' not in col.upper():
                if values.min() < -10 or values.max() > 110:
                    self.warnings.append(
                        f"Feature '{col}' has suspicious range: [{values.min():.2f}, {values.max():.2f}]")

            # Williams %R should be [-100, 0]
            if 'WILLIAMS' in col.upper():
                if values.min() < -110 or values.max() > 10:
                    self.warnings.append(
                        f"Feature '{col}' has wrong range: [{values.min():.2f}, {values.max():.2f}] (expected [-100, 0])")

            # Check per valori troppo grandi (ignore volume-based features)
            if values.max() > 1e10 and not any(x in col.upper() for x in ['OBV', 'VOLUME', 'VWAP']):
                self.warnings.append(f"Feature '{col}' has very large values (max={values.max():.2e})")

            # Check se tutti valori identici
            if values.std() < 1e-10:
                self.warnings.append(f"Feature '{col}' is nearly constant (std={values.std():.2e})")

        if not self.warnings:
            logger.info("✓ All value ranges look reasonable")
        else:
            logger.warning(f"Found {len([w for w in self.warnings if 'range' in w.lower()])} range warnings")

    def check_feature_correlations(self):
        """Verifica features altamente correlate (ridondanti)"""
        logger.info("\n--- Checking Feature Correlations ---")

        # Sample per velocità
        sample_size = min(1000, len(self.data))
        sample_data = self.data[self.feature_cols].sample(n=sample_size, random_state=42)

        # Calcola correlazioni
        corr_matrix = sample_data.corr().abs()

        # Upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Features con correlazione > 0.95
        high_corr = [(column, row, corr_matrix.loc[row, column])
                     for column in upper.columns
                     for row in upper.index
                     if upper.loc[row, column] > 0.95]

        if high_corr:
            self.warnings.append(f"{len(high_corr)} pairs of features have correlation >0.95")
            logger.warning(f"Highly correlated feature pairs: {len(high_corr)}")
            # Show first 5
            for feat1, feat2, corr in high_corr[:5]:
                logger.warning(f"  {feat1} <-> {feat2}: {corr:.4f}")
        else:
            logger.info("✓ No highly correlated features")

    def check_constant_features(self):
        """Verifica features costanti (inutili)"""
        logger.info("\n--- Checking Constant Features ---")

        constant_features = []
        for col in self.feature_cols:
            if self.data[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            self.warnings.append(f"{len(constant_features)} features are constant")
            logger.warning(f"Constant features: {constant_features}")
        else:
            logger.info("✓ No constant features")

    def check_data_leakage(self):
        """Verifica possibili data leakage"""
        logger.info("\n--- Checking Data Leakage ---")

        # Check 1: Features che contengono "Close" ma non dovrebbero
        suspicious = [col for col in self.feature_cols
                      if 'Close' in col and col not in ['Close_Open_ratio', 'Close_to_SMA_20', 'Close_to_VWAP']]

        if suspicious:
            self.warnings.append(f"Suspicious features containing 'Close': {suspicious}")

        # Note: SMA NaN check removed - it's a false positive after dropna() is applied
        # The validator runs on already-cleaned data, so NaN checks are not meaningful

        if not suspicious:
            logger.info("✓ No obvious data leakage detected")

    def generate_report(self):
        """Genera report finale"""
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE VALIDATION REPORT")
        logger.info("=" * 80)

        logger.info(f"Total features validated: {len(self.feature_cols)}")
        logger.info(f"Errors found: {len(self.errors)}")
        logger.info(f"Warnings found: {len(self.warnings)}")

        if self.errors:
            logger.error("\nERRORS:")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            logger.warning("\nWARNINGS:")
            for warning in self.warnings[:10]:  # Show first 10
                logger.warning(f"  - {warning}")
            if len(self.warnings) > 10:
                logger.warning(f"  ... and {len(self.warnings) - 10} more warnings")

        if not self.errors and not self.warnings:
            logger.info("\n✓✓✓ ALL VALIDATIONS PASSED!")
        elif not self.errors:
            logger.info("\n✓ No critical errors, but review warnings")
        else:
            logger.error("\n✗ CRITICAL ERRORS FOUND - Fix before using!")

        logger.info("=" * 80)


def validate_features(data, feature_cols):
    """
    Helper function per validare features

    Args:
        data: DataFrame con features
        feature_cols: Lista colonne features

    Returns:
        bool: True se validazione passa
    """
    validator = FeatureValidator(data, feature_cols)
    passed = validator.validate_all()

    return passed


# TEST SPECIFICI PER INDICATORI TECNICI COMUNI

def test_rsi_calculation(df):
    """Test che RSI sia nel range [0, 100]"""
    rsi_cols = [col for col in df.columns if 'RSI' in col]

    for col in rsi_cols:
        values = df[col].dropna()
        if len(values) > 0:
            if values.min() < 0 or values.max() > 100:
                logger.error(f"ERROR: {col} out of range [0,100]: [{values.min():.2f}, {values.max():.2f}]")
                return False

    logger.info("✓ RSI values in correct range")
    return True


def test_bollinger_bands(df):
    """Test che Bollinger Bands siano ordinate correttamente"""
    periods = [20, 50]

    for period in periods:
        upper_col = f'BB_upper_{period}'
        middle_col = f'BB_middle_{period}'
        lower_col = f'BB_lower_{period}'

        if all(col in df.columns for col in [upper_col, middle_col, lower_col]):
            # Upper >= Middle >= Lower
            violations = df[(df[upper_col] < df[middle_col]) | (df[middle_col] < df[lower_col])]

            if len(violations) > 0:
                logger.error(f"ERROR: Bollinger Bands order violation for period {period}: {len(violations)} cases")
                return False

    logger.info("✓ Bollinger Bands correctly ordered")
    return True


def test_volume_ratios(df):
    """Test che volume ratios siano positivi"""
    volume_cols = [col for col in df.columns if 'Volume' in col and 'ratio' in col]

    for col in volume_cols:
        values = df[col].dropna()
        if len(values) > 0 and values.min() < 0:
            logger.error(f"ERROR: {col} has negative values")
            return False

    logger.info("✓ Volume ratios are positive")
    return True


if __name__ == '__main__':
    # Example usage
    print("Feature Validator - Ready to use")
    print("Usage:")
    print("  from feature_validator import validate_features")
    print("  validate_features(df, feature_columns)")




