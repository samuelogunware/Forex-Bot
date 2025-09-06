import numpy as np
import pandas as pd
from datetime import datetime

from config import settings
from src.feature_engineering import ForexFeatureEngineer

class ForexTradingPredictor:
    """
    Uses a trained model to generate trading signals and risk management parameters.
    """
    def __init__(self, model, feature_engineer: ForexFeatureEngineer, symbol: str):
        self.model = model
        self.feature_engineer = feature_engineer
        self.symbol = symbol

    def generate_forex_signals(self, latest_data: pd.DataFrame) -> dict:
        """
        Generates Forex trading signals with integrated stop-loss and take-profit levels.
        """
        if latest_data.empty or len(latest_data) < 50: # Need enough data for indicators
            return self._default_signal()

        # --- Prepare Features ---
        X_pred = self.feature_engineer.prepare_forex_features(latest_data.copy(), self.symbol, predict_mode=True)
        
        if X_pred.empty:
            return self._default_signal()

        latest_features = X_pred.iloc[[-1]] # Select the most recent row of features
        
        # --- Make Prediction ---
        predicted_pips = self.model.predict(latest_features)[0]
        
        # --- Calculate Risk Management Parameters ---
        # Use the ATR from the unprocessed data for a more direct measure
        atr_pips = self.feature_engineer.calculate_pips(latest_data.ta.atr(length=14).iloc[-1], self.symbol)

        if pd.isna(atr_pips) or atr_pips == 0:
             atr_pips = 10 # Default to 10 pips if ATR is invalid

        stop_loss_pips = atr_pips * settings.STOP_LOSS_ATR_MULTIPLIER
        take_profit_pips = stop_loss_pips * (settings.TAKE_PROFIT_ATR_MULTIPLIER / settings.STOP_LOSS_ATR_MULTIPLIER)

        # --- Generate Signal ---
        pip_threshold = stop_loss_pips * 0.5 
        signal = 'HOLD'
        if predicted_pips > pip_threshold:
            signal = 'BUY'
        elif predicted_pips < -pip_threshold:
            signal = 'SELL'
            
        return {
            'symbol': self.symbol,
            'signal': signal,
            'predicted_pips': predicted_pips,
            'stop_loss_pips': stop_loss_pips,
            'take_profit_pips': take_profit_pips,
            'timestamp': datetime.now(),
            'current_price': latest_data['close'].iloc[-1]
        }
        
    def _default_signal(self) -> dict:
        """Returns a neutral signal when a prediction cannot be made."""
        return {'symbol': self.symbol, 'signal': 'HOLD', 'predicted_pips': 0}

