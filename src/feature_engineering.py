import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Union

from config import settings

class ForexFeatureEngineer:
    """
    Handles the creation of all predictive features for the Forex models.
    """
    def __init__(self):
        pass

    def prepare_forex_features(self, df: pd.DataFrame, symbol: str, predict_mode: bool = False) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """
        Prepares the final feature set for either training or prediction.

        Args:
            df: The input DataFrame with OHLCV data.
            symbol: The currency pair symbol (e.g., 'EUR/USD').
            predict_mode: If True, prepares data for a live prediction (returns only X).
                          If False, prepares data for training (returns X and y).

        Returns:
            If predict_mode is False, returns a tuple of (features_df, target_series).
            If predict_mode is True, returns only the features_df.
        """
        # --- Add all feature types ---
        df = self._add_forex_specific_indicators(df)
        df = self._add_time_based_features(df)
        df = self._add_volatility_features(df, symbol)
        df['pip_change'] = self.calculate_pips(df['close'].diff(), symbol)


        # --- Handle Target Variable ---
        if not predict_mode:
            # Create target: future price movement in pips
            future_price = df['close'].shift(-settings.PREDICTION_HORIZON)
            df['target'] = self.calculate_pips(future_price - df['close'], symbol)

        # Drop rows with NaN values created by indicators and shifts
        df.dropna(inplace=True)

        # --- Separate Features (X) from Target (y) ---
        drop_cols = ['open', 'high', 'low', 'volume', 'target']
        feature_columns = [col for col in df.columns if col not in drop_cols]
        X = df[feature_columns]

        if predict_mode:
            return X
        else:
            y = df['target']
            return X, y

    def calculate_pips(self, price_change: float, symbol: str) -> float:
        """Calculates pips from a given price change."""
        if '/JPY' in symbol:
            return price_change * 100
        else:
            return price_change * 10000

    def _add_forex_specific_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds standard technical indicators relevant to Forex."""
        df.ta.rsi(length=14, append=True)
        df.ta.stoch(k=14, d=3, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        adx = df.ta.adx(length=14)
        if adx is not None and not adx.empty:
            df = df.join(adx)
        
        macd = df.ta.macd(fast=12, slow=26)
        if macd is not None and not macd.empty:
            df = df.join(macd)

        return df

    def _add_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds time-based features, crucial for Forex session analysis."""
        df['london_session'] = ((df.index.hour >= 7) & (df.index.hour < 16)).astype(int)
        df['ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 22)).astype(int)
        df['tokyo_session'] = ((df.index.hour >= 23) | (df.index.hour < 8)).astype(int)
        df['day_of_week'] = df.index.dayofweek
        df['hour_of_day'] = df.index.hour
        return df

    def _add_volatility_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Adds volatility features, key for risk management."""
        atr = df.ta.atr(length=14)
        if atr is not None:
             df['atr_pips'] = self.calculate_pips(atr, symbol)
        
        bbands = df.ta.bbands(length=20)
        if bbands is not None and not bbands.empty:
            df['bb_width'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / df['close']
        
        return df

