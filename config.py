
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')

# Application Settings
APP_TITLE = "Real-Time Stock Sentiment Predictor"
APP_ICON = "📈"
DEFAULT_SYMBOL = "BTCUSDT"
UPDATE_INTERVAL = 60  # seconds

# Model Configuration
MODEL_PATH = "data/models/sentiment_model.pkl"
LOG_PATH = "data/logs/predictions.csv"
SENTIMENT_THRESHOLD = 0.5

# Supported Trading Pairs
TRADING_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", 
    "ADAUSDT", "DOGEUSDT", "XRPUSDT"
]

# Chart Settings
CHART_COLORS = {
    'positive': '#00C853',
    'negative': '#FF1744',
    'neutral': '#FFD600'
}
