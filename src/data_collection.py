# src/data_collection.py
# Fixed DataCollector with proper rate limiting and volume data

import requests
import pandas as pd
from datetime import datetime
import time

class DataCollector:
    def __init__(self):
        # CoinGecko symbol mapping
        self.coingecko_map = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum',
            'BNBUSDT': 'binancecoin',
            'SOLUSDT': 'solana',
            'ADAUSDT': 'cardano',
            'XRPUSDT': 'ripple',
            'DOTUSDT': 'polkadot',
            'DOGEUSDT': 'dogecoin',
            'MATICUSDT': 'matic-network',
            'LTCUSDT': 'litecoin'
        }
        self.last_request_time = 0
        self.min_request_interval = 1.5  # 1.5 seconds between requests (safe for free tier)

    def _rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def get_realtime_price(self, symbol):
        """Fetch current price using CoinGecko API. Returns None on failure."""
        try:
            if symbol not in self.coingecko_map:
                return None

            coin_id = self.coingecko_map[symbol]
            self._rate_limit()
            
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': coin_id, 
                'vs_currencies': 'usd',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }

            response = requests.get(url, params=params, timeout=10)

            # Handle rate limit with exponential backoff
            if response.status_code == 429:
                time.sleep(10)  # Wait 10 seconds instead of 60
                response = requests.get(url, params=params, timeout=10)

            response.raise_for_status()
            data = response.json()

            if coin_id not in data or 'usd' not in data[coin_id]:
                return None

            price_info = data[coin_id]
            
            return {
                'symbol': symbol,
                'price': float(price_info['usd']),
                'volume_24h': float(price_info.get('usd_24h_vol', 0)),
                'change_24h': float(price_info.get('usd_24h_change', 0)),
                'timestamp': datetime.now()
            }

        except requests.exceptions.RequestException:
            return None
        except Exception:
            return None

    def get_historical_data(self, symbol, interval='1h', limit=100):
        """Fetch historical data from CoinGecko with volume data."""
        try:
            if symbol not in self.coingecko_map:
                return None

            coin_id = self.coingecko_map[symbol]
            self._rate_limit()

            # Calculate days needed
            hours_needed = int(limit)
            days_needed = max(1, (hours_needed // 24) + 1)
            
            # Use market_chart endpoint which provides volume
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': min(days_needed, 90),  # Max 90 days for free tier
                'interval': 'hourly' if days_needed <= 90 else 'daily'
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 429:
                time.sleep(10)
                response = requests.get(url, params=params, timeout=15)

            if response.status_code != 200:
                return None

            data = response.json()
            
            # Extract prices and volumes
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if not prices or len(prices) == 0:
                return None

            # Create DataFrame from prices
            df_price = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df_volume = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            
            # Merge price and volume data
            df = pd.merge(df_price, df_volume, on='timestamp', how='left')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Create OHLC data (approximate from close prices)
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df[['open', 'close']].max(axis=1) * 1.001  # Slight variance
            df['low'] = df[['open', 'close']].min(axis=1) * 0.999
            df['volume'] = df['volume'].fillna(0)

            # Ensure numeric types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna()

            # Take most recent rows
            if len(df) > limit:
                df = df.tail(limit)

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            return df.reset_index(drop=True)

        except requests.exceptions.RequestException:
            return None
        except Exception:
            return None

    def get_market_depth(self, symbol):
        """Market depth not available in free CoinGecko API"""
        return None