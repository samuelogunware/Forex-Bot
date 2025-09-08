import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import requests
import time
from typing import List, Dict

from config import settings

class ForexDataProcessor:
    """
    Handles all data fetching and preprocessing from various sources.
    """
    def __init__(self):
        self.oanda_client = oandapyV20.API(
            access_token=settings.OANDA_ACCESS_TOKEN,
            environment=settings.OANDA_ENVIRONMENT
        )

    def fetch_forex_data_oanda(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """
        Fetches historical Forex data from OANDA, handling pagination for large requests.
        """
        oanda_symbol = symbol.replace('/', '_')
        params = {"granularity": timeframe.upper(), "price": "M"} 

        all_candles = []
        remaining_count = count


        while remaining_count > 0:
            request_count = min(remaining_count, settings.OANDA_API_REQUEST_LIMIT)
            params['count'] = request_count
            
            try:
                r = instruments.InstrumentsCandles(instrument=oanda_symbol, params=params)
                self.oanda_client.request(r)
                
                candles = r.response.get('candles', [])
                if not candles:
                    break # No more data available
                
                all_candles = candles + all_candles
                
                first_timestamp_str = candles[0]['time']
                first_timestamp = pd.to_datetime(first_timestamp_str)
                params["to"] = first_timestamp.isoformat() + "Z"

                remaining_count -= len(candles)
                
                if len(candles) < request_count:
                    break # Fetched all available data in this range

            except Exception as e:
                print(f"Error fetching OANDA data for {symbol}: {e}. Falling back to alternative.")
                return self.fetch_forex_data_alternative(symbol, timeframe, count)

        if not all_candles:
            return pd.DataFrame()

        df = self._candles_to_dataframe(all_candles)
        return df.sort_index()

    def fetch_forex_data_alternative(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Fallback Forex data source using Alpha Vantage."""
    
        print(f"Note: Using Alpha Vantage as a fallback for {symbol}. Data may be less granular.")
        return pd.DataFrame()

    def _candles_to_dataframe(self, candles: List[Dict]) -> pd.DataFrame:
        """Converts OANDA candle data to a pandas DataFrame."""
        data = []
        for candle in candles:
            if not candle['complete']: continue
            data.append({
                'timestamp': pd.to_datetime(candle['time']),
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
                'volume': int(candle['volume']),
            })
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def clean_forex_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Cleans and preprocesses Forex data."""
        df = df.ffill().bfill()
        df['pip_change'] = (df['close'].diff() * self.get_pip_multiplier(symbol)).round(2)
        return df

    def get_pip_size(self, symbol: str) -> float:
        """Returns the pip size for a given currency pair."""
        return 0.01 if 'JPY' in symbol else 0.0001
        
    def get_pip_multiplier(self, symbol: str) -> int:
        """Returns the multiplier to convert price change to pips."""
        return 100 if 'JPY' in symbol else 10000

