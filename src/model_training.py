import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from config import settings
from src.data_processing import ForexDataProcessor
from src.feature_engineering import ForexFeatureEngineer

class ModelTrainer:
    """
    Handles the complete lifecycle of a model: training, validation, saving, and loading.
    """
    def __init__(self):
        self.models = {} # A cache for loaded/trained models

    def get_or_create_model(self, symbol: str, data_processor: ForexDataProcessor, feature_engineer: ForexFeatureEngineer) -> xgb.XGBRegressor:
        """
        Main entry point for getting a model.
        Tries to load from disk first, otherwise trains a new one.
        """
        if symbol in self.models:
            return self.models[symbol]
        
        try:
            model = self._load_model_from_disk(symbol)
            self.models[symbol] = model
            return model
        except FileNotFoundError:
            print(f"No model file found for {symbol}. Training a new one.")
            return self._train_new_model(symbol, data_processor, feature_engineer)

    def _load_model_from_disk(self, symbol: str) -> xgb.XGBRegressor:
        """Loads a pre-trained model from a .pkl file."""
        model_path = f'models/trained/{symbol.replace("/", "_")}_model.pkl'
        return joblib.load(model_path)

    def _train_new_model(self, symbol: str, data_processor: ForexDataProcessor, feature_engineer: ForexFeatureEngineer) -> xgb.XGBRegressor:
        """Orchestrates the process of training a new model."""
        print(f"Starting training for {symbol}...")
        try:
            # 1. Fetch Data
            data = data_processor.fetch_forex_data_oanda(
                symbol,
                settings.TIME_FRAME,
                count=settings.LOOKBACK_PERIOD * 24
            )
            if data.empty:
                print(f"Could not fetch training data for {symbol}. Cannot create model.")
                return None

            # 2. Engineer Features
            data = data_processor.clean_forex_data(data, symbol)
            X, y = feature_engineer.prepare_forex_features(data, symbol)

            if X.empty or y.empty:
                print(f"Not enough data to create features for {symbol}. Cannot create model.")
                return None

            # 3. Perform Training with Cross-Validation
            model, score = self._perform_training(X, y)
            print(f"{symbol} model trained with CV RMSE (pips): {score:.4f}")
            
            # 4. Save the final model
            self._save_model_to_disk(model, symbol)
            self.models[symbol] = model # Cache the new model
            return model

        except Exception as e:
            print(f"An error occurred during model training for {symbol}: {e}")
            return None

    def _perform_training(self, X: pd.DataFrame, y: pd.Series) -> Tuple[xgb.XGBRegressor, float]:
        """Performs the actual XGBoost training with time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=settings.N_SPLITS)
        scores = []

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='rmse'
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )

            preds = model.predict(X_test)
            # Calculate MSE first, then take the square root for RMSE
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            scores.append(rmse)
            print(f"  Fold {fold + 1}/{settings.N_SPLITS} RMSE (pips): {rmse:.4f}")

        # Train a final model on all available data for deployment
        final_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(X, y, verbose=False)
        
        return final_model, np.mean(scores)

    def _save_model_to_disk(self, model, symbol: str):
        """Saves the trained model to a .pkl file."""
        model_path = f'models/trained/{symbol.replace("/", "_")}_model.pkl'
        joblib.dump(model, model_path)
        print(f"Saving model to {model_path}")

