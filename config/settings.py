import os
from dotenv import load_dotenv

load_dotenv()

# --- Broker Configuration (OANDA for Forex) ---
OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID')
OANDA_ACCESS_TOKEN = os.getenv('OANDA_ACCESS_TOKEN')
OANDA_ENVIRONMENT = 'practice'  

# --- Data Source Configuration ---
FOREX_SYMBOLS = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
    'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP',
    'EUR/JPY', 'GBP/JPY'
]
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# --- Trading Parameters ---
TIME_FRAME = 'H1'  # M1, M5, M15, M30, H1, H4, D1
LOOKBACK_PERIOD = 365  # days
OANDA_API_REQUEST_LIMIT = 4500 # OANDA's limit is 5000

# --- Model Parameters ---
N_SPLITS = 5
PREDICTION_HORIZON = 4  # periods

# --- Risk Management ---
STOP_LOSS_ATR_MULTIPLIER = 2.0
TAKE_PROFIT_ATR_MULTIPLIER = 3.0 # Risk/Reward Ratio of 1.5 (3.0 / 2.0)
ACCOUNT_RISK_PER_TRADE = 0.01 # Risk 1% of the account balance per trade

