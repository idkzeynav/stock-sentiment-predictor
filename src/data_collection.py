
# src/data_collection.py
# Cleaned DataCollector (removed Streamlit debug/info calls - returns None or raises exceptions)

"""
A cleaned DataCollector that uses CoinGecko as a primary data source.
This file deliberately avoids writing debug/info to Streamlit so the UI remains clean.
"""

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

        self.use_coingecko = True

    def get_realtime_price(self, symbol):
        """Fetch current price using CoinGecko API. Returns None on failure."""
        try:
            if symbol not in self.coingecko_map:
                return None

            coin_id = self.coingecko_map[symbol]
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {'ids': coin_id, 'vs_currencies': 'usd'}

            response = requests.get(url, params=params, timeout=10)

            # Handle rate limit
            if response.status_code == 429:
                time.sleep(60)
                response = requests.get(url, params=params, timeout=10)

            response.raise_for_status()
            data = response.json()

            if coin_id not in data or 'usd' not in data[coin_id]:
                return None

            price = float(data[coin_id]['usd'])

            # Small delay to be kind to API
            time.sleep(0.5)

            return {
                'symbol': symbol,
                'price': price,
                'timestamp': datetime.now()
            }

        except requests.exceptions.RequestException:
            return None
        except Exception:
            return None

    def get_historical_data(self, symbol, interval='1h', limit=100):
        """Fetch historical data from CoinGecko OHLC endpoint. Returns a DataFrame or None.
        CoinGecko OHLC returns candles for days; this function maps hours->days and
        trims the returned data to the requested number of rows.
        """
        try:
            if symbol not in self.coingecko_map:
                return None

            coin_id = self.coingecko_map[symbol]

            # Map requested hours to days (CoinGecko accepts specific day buckets)
            hours_needed = int(limit)
            days_needed = max(1, (hours_needed // 24) + 1)
            valid_days = [1, 7, 14, 30, 90, 180, 365]
            days = next((d for d in valid_days if d >= days_needed), valid_days[-1])

            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            params = {'vs_currency': 'usd', 'days': days}

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 429:
                time.sleep(60)
                response = requests.get(url, params=params, timeout=15)

            if response.status_code != 200:
                return None

            data = response.json()
            if not isinstance(data, list) or len(data) == 0:
                return None

            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Placeholder volume (CoinGecko OHLC doesn't provide volume)
            df['volume'] = 0

            # Ensure numeric types and drop NaNs
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
            df = df.dropna()

            # Take the tail (most recent) rows to match requested limit
            if len(df) > limit:
                df = df.tail(limit)

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            return df.reset_index(drop=True)

        except requests.exceptions.RequestException:
            return None
        except Exception:
            return None

    def get_market_depth(self, symbol):
        # Not available via CoinGecko free API
        return None
