# src/data_collection.py
# Alternative using Binance API (no rate limits for public endpoints)

import requests
import pandas as pd
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.last_request_time = 0
        self.min_request_interval = 0.2  # Binance is more generous
        
    def _rate_limit(self):
        """Basic rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def get_realtime_price(self, symbol):
        """Fetch current price from Binance"""
        try:
            self._rate_limit()
            
            url = f"{self.base_url}/api/v3/ticker/24hr"
            params = {'symbol': symbol}
            
            logger.info(f"Fetching real-time price for {symbol}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'symbol': symbol,
                'price': float(data['lastPrice']),
                'volume_24h': float(data['quoteVolume']),
                'change_24h': float(data['priceChangePercent']),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching price: {str(e)}")
            return None

    def get_historical_data(self, symbol, interval='1h', limit=100):
        """Fetch historical klines from Binance"""
        try:
            self._rate_limit()
            
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # Binance max is 1000
            }
            
            logger.info(f"Fetching {limit} candles for {symbol}")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.error("No data received from Binance")
                return None
            
            # Parse klines data
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Select and convert columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            logger.info(f"Successfully fetched {len(df)} rows")
            return df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None

    def get_market_depth(self, symbol):
        """Fetch order book depth"""
        try:
            self._rate_limit()
            
            url = f"{self.base_url}/api/v3/depth"
            params = {'symbol': symbol, 'limit': 20}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching market depth: {str(e)}")
            return None
    
    def test_connection(self):
        """Test Binance API connection"""
        try:
            url = f"{self.base_url}/api/v3/ping"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("✅ Binance API connection successful")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Connection test failed: {str(e)}")
            return False