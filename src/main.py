import pandas as pd
import numpy as np
from datetime import datetime
import time
import schedule
import os

from config import settings
from src.data_processing import ForexDataProcessor
from src.feature_engineering import ForexFeatureEngineer
from src.model_training import ModelTrainer
from src.prediction import ForexTradingPredictor

from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.exceptions import V20Error

class ForexTradingBot:
    """
    The main class for the Forex Trading Bot.
    Orchestrates data processing, model training, and prediction loop.
    """
    def __init__(self):
        self.data_processor = ForexDataProcessor()
        self.feature_engineer = ForexFeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.predictors = {}
        self._initialize_all_symbols()

    def _initialize_all_symbols(self):
        """Initializes models and predictors for all symbols in the settings."""
        for symbol in settings.FOREX_SYMBOLS:
            model = self.model_trainer.get_or_create_model(
                symbol, self.data_processor, self.feature_engineer
            )
            if model:
                self.predictors[symbol] = ForexTradingPredictor(model, self.feature_engineer, symbol)
            else:
                print(f"Failed to initialize predictor for {symbol} due to model creation failure.")

    def run_forex_prediction(self):
        """Runs the prediction and trade execution routine for all symbols."""
        print(f"\n>>>>> Running Prediction Cycle at {datetime.now()} <<<<<")
        for symbol in settings.FOREX_SYMBOLS:
            if symbol not in self.predictors:
                print(f"Skipping {symbol}, predictor not initialized.")
                continue
            
            try:
                latest_data = self.data_processor.fetch_forex_data_oanda(
                    symbol, settings.TIME_FRAME, count=200
                )

                if not latest_data.empty:
                    signal = self.predictors[symbol].generate_forex_signals(latest_data)
                    self.log_forex_prediction(symbol, signal)

                    if signal['signal'] != 'HOLD':
                        self.execute_forex_trade(symbol, signal)
                else:
                    print(f"Could not fetch latest data for {symbol}. Skipping prediction.")

            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")

    def execute_forex_trade(self, symbol: str, signal: dict):
        """Constructs and sends a live order to the OANDA API."""
        print(f"  --- EXECUTION ---: Attempting to place {signal['signal']} order for {symbol}.")

        pip_size = self.data_processor.get_pip_size(symbol)
        current_price = signal['current_price']
        
        # 1. Determine SL and TP prices
        if signal['signal'] == 'BUY':
            stop_loss_price = current_price - (signal['stop_loss_pips'] * pip_size)
            take_profit_price = current_price + (signal['take_profit_pips'] * pip_size)
            units = 5000 # Positive for a buy order
        else: # SELL
            stop_loss_price = current_price + (signal['stop_loss_pips'] * pip_size)
            take_profit_price = current_price - (signal['take_profit_pips'] * pip_size)
            units = -5000 # Negative for a sell order

        # 2. Define the order details
        order_definition = {
            "order": {
                "instrument": symbol.replace('/', '_'),
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "timeInForce": "GTC",
                    "price": f"{stop_loss_price:.5f}"
                },
                "takeProfitOnFill": {
                    "timeInForce": "GTC",
                    "price": f"{take_profit_price:.5f}"
                }
            }
        }

        # 3. Send the order to OANDA
        try:
            request = OrderCreate(accountID=settings.OANDA_ACCOUNT_ID, data=order_definition)
            response = self.data_processor.oanda_client.request(request)
            print(f"  +++ SUCCESS +++: Order placed for {symbol}. Transaction ID: {response['orderFillTransaction']['id']}")
        except V20Error as e:
            print(f"  --- ERROR ---: Failed to place order for {symbol}. OANDA API Error: {e}")
        except Exception as e:
            print(f"  --- ERROR ---: An unexpected error occurred during trade execution for {symbol}: {e}")


    def log_forex_prediction(self, symbol: str, signal: dict):
        """Logs the prediction results to a CSV file."""
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': symbol,
            'signal': signal.get('signal', 'HOLD'),
            'predicted_pips': signal.get('predicted_pips', 0),
            'stop_loss_pips': signal.get('stop_loss_pips', 0),
            'take_profit_pips': signal.get('take_profit_pips', 0),
            'current_price': signal.get('current_price', 0)
        }
        log_df = pd.DataFrame([log_entry])
        log_path = 'logs/forex_predictions_log.csv'
        log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
        print(f"  {symbol}: {signal.get('signal', 'HOLD')}, Pred Pips: {signal.get('predicted_pips', 0):.2f}, SL: {signal.get('stop_loss_pips', 0):.2f} pips")

    def run(self):
        """Main execution loop for the Forex bot."""
        print("Starting Forex Trading Bot...")
        if settings.TIME_FRAME == 'H1':
            schedule.every().hour.at(":01").do(self.run_forex_prediction)
        elif settings.TIME_FRAME == 'D1':
            schedule.every().day.at("00:01").do(self.run_forex_prediction)

        self.run_forex_prediction() # Run once on startup

        print("Bot is running. Waiting for scheduled tasks...")
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('models/trained'):
        os.makedirs('models/trained')

    forex_bot = ForexTradingBot()
    forex_bot.run()

