import requests
import pandas as pd
from binance.client import Client
from datetime import datetime
import config

class DataCollector:
    def __init__(self):
        self.client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    
    def get_realtime_price(self, symbol):
        """Fetch current price for a trading pair"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return {
                'symbol': symbol,
                'price': float(ticker['price']),
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error fetching price: {e}")
            return None
    
    def get_historical_data(self, symbol, interval='1h', limit=100):
        """Fetch historical klines data"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_volume', 'trades',
                'taker_base', 'taker_quote', 'ignore'
            ])
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # CRITICAL: Convert ALL price columns to float
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def get_market_depth(self, symbol):
        """Get order book depth"""
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=10)
            return {
                'bids': depth['bids'][:5],
                'asks': depth['asks'][:5]
            }
        except Exception as e:
            print(f"Error fetching market depth: {e}")
            return None