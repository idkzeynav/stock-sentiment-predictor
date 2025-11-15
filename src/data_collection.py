# src/data_collection.py
# Version with comprehensive debugging to identify issues

import requests
import pandas as pd
from datetime import datetime
import time
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        self.min_request_interval = 2.0
        self.max_retries = 3
        self.debug_info = []  # Store debug messages

    def get_debug_info(self):
        """Return all debug messages collected"""
        return self.debug_info

    def clear_debug_info(self):
        """Clear debug messages"""
        self.debug_info = []

    def _log_debug(self, message):
        """Add message to debug info and log it"""
        self.debug_info.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        logger.info(message)

    def _rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            self._log_debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def test_connection(self):
        """Test if CoinGecko API is accessible"""
        try:
            self._log_debug("Testing CoinGecko API connection...")
            url = "https://api.coingecko.com/api/v3/ping"
            response = requests.get(url, timeout=10)
            
            self._log_debug(f"Ping response status: {response.status_code}")
            
            if response.status_code == 200:
                self._log_debug("✅ CoinGecko API is accessible")
                return True
            else:
                self._log_debug(f"❌ API returned status {response.status_code}")
                self._log_debug(f"Response: {response.text[:200]}")
                return False
        except Exception as e:
            self._log_debug(f"❌ Connection test failed: {str(e)}")
            return False

    def get_realtime_price(self, symbol):
        """Fetch current price using CoinGecko API with retries."""
        self.clear_debug_info()
        
        for attempt in range(self.max_retries):
            try:
                if symbol not in self.coingecko_map:
                    self._log_debug(f"❌ Symbol {symbol} not found in mapping")
                    self._log_debug(f"Available symbols: {list(self.coingecko_map.keys())}")
                    return None

                coin_id = self.coingecko_map[symbol]
                self._log_debug(f"Fetching price for {symbol} -> {coin_id} (attempt {attempt + 1}/{self.max_retries})")
                
                self._rate_limit()
                
                url = "https://api.coingecko.com/api/v3/simple/price"
                params = {
                    'ids': coin_id, 
                    'vs_currencies': 'usd',
                    'include_24hr_vol': 'true',
                    'include_24hr_change': 'true'
                }
                
                self._log_debug(f"Request URL: {url}")
                self._log_debug(f"Request params: {params}")

                response = requests.get(url, params=params, timeout=15)
                self._log_debug(f"Response status: {response.status_code}")

                # Handle rate limit
                if response.status_code == 429:
                    wait_time = 15 * (attempt + 1)
                    self._log_debug(f"⚠️ Rate limited! Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()
                self._log_debug(f"Response data keys: {list(data.keys())}")

                if coin_id not in data:
                    self._log_debug(f"❌ Coin ID '{coin_id}' not in response")
                    self._log_debug(f"Response: {json.dumps(data, indent=2)[:500]}")
                    continue

                if 'usd' not in data[coin_id]:
                    self._log_debug(f"❌ 'usd' not in response for {coin_id}")
                    self._log_debug(f"Available keys: {list(data[coin_id].keys())}")
                    continue

                price_info = data[coin_id]
                
                result = {
                    'symbol': symbol,
                    'price': float(price_info['usd']),
                    'volume_24h': float(price_info.get('usd_24h_vol', 0)),
                    'change_24h': float(price_info.get('usd_24h_change', 0)),
                    'timestamp': datetime.now()
                }
                
                self._log_debug(f"✅ Successfully fetched price: ${result['price']:,.2f}")
                return result

            except requests.exceptions.Timeout:
                self._log_debug(f"⏱️ Request timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                continue
            except requests.exceptions.RequestException as e:
                self._log_debug(f"❌ Request error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                continue
            except Exception as e:
                self._log_debug(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
                import traceback
                self._log_debug(f"Traceback: {traceback.format_exc()[:500]}")
                return None
        
        self._log_debug(f"❌ Failed to fetch price after {self.max_retries} attempts")
        return None

    def get_historical_data(self, symbol, interval='1h', limit=100):
        """Fetch historical data from CoinGecko with comprehensive debugging."""
        self.clear_debug_info()
        
        for attempt in range(self.max_retries):
            try:
                if symbol not in self.coingecko_map:
                    self._log_debug(f"❌ Symbol {symbol} not found in mapping")
                    self._log_debug(f"Available symbols: {list(self.coingecko_map.keys())}")
                    return None

                coin_id = self.coingecko_map[symbol]
                self._log_debug(f"📊 Fetching historical data for {symbol} -> {coin_id}")
                self._log_debug(f"Parameters: interval={interval}, limit={limit}, attempt={attempt + 1}/{self.max_retries}")
                
                self._rate_limit()

                # Calculate days needed
                hours_needed = int(limit)
                days_needed = max(1, (hours_needed // 24) + 1)
                days_needed = min(days_needed, 365)
                
                self._log_debug(f"Requesting {days_needed} days of data ({hours_needed} hours)")
                
                # Use market_chart endpoint
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': days_needed,
                    'interval': 'hourly' if days_needed <= 90 else 'daily'
                }
                
                self._log_debug(f"Request URL: {url}")
                self._log_debug(f"Request params: {params}")

                response = requests.get(url, params=params, timeout=20)
                self._log_debug(f"Response status: {response.status_code}")
                self._log_debug(f"Response headers: {dict(response.headers)}")

                # Handle rate limit
                if response.status_code == 429:
                    wait_time = 20 * (attempt + 1)
                    self._log_debug(f"⚠️ Rate limited! Waiting {wait_time}s...")
                    self._log_debug(f"Response body: {response.text[:200]}")
                    time.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    self._log_debug(f"❌ API returned status code {response.status_code}")
                    self._log_debug(f"Response body: {response.text[:500]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                        continue
                    return None

                data = response.json()
                self._log_debug(f"Response data keys: {list(data.keys())}")
                
                # Validate response
                if 'prices' not in data:
                    self._log_debug(f"❌ 'prices' key not in response")
                    self._log_debug(f"Full response: {json.dumps(data, indent=2)[:500]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(5)
                        continue
                    return None
                
                if not data['prices']:
                    self._log_debug(f"❌ Prices array is empty")
                    if attempt < self.max_retries - 1:
                        time.sleep(5)
                        continue
                    return None

                prices = data['prices']
                volumes = data.get('total_volumes', [])
                
                self._log_debug(f"✅ Received {len(prices)} price points, {len(volumes)} volume points")

                # Create DataFrames
                self._log_debug("Creating price DataFrame...")
                df_price = pd.DataFrame(prices, columns=['timestamp', 'close'])
                self._log_debug(f"Price DataFrame shape: {df_price.shape}")
                
                if volumes and len(volumes) > 0:
                    self._log_debug("Merging volume data...")
                    df_volume = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                    df = pd.merge(df_price, df_volume, on='timestamp', how='left')
                    self._log_debug(f"Merged DataFrame shape: {df.shape}")
                else:
                    self._log_debug("⚠️ No volume data available, using zeros")
                    df = df_price.copy()
                    df['volume'] = 0
                
                # Convert timestamp
                self._log_debug("Converting timestamps...")
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                self._log_debug(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                
                # Create OHLC data
                self._log_debug("Creating OHLC data...")
                df['open'] = df['close'].shift(1).fillna(df['close'])
                df['high'] = df[['open', 'close']].max(axis=1) * 1.002
                df['low'] = df[['open', 'close']].min(axis=1) * 0.998
                df['volume'] = df['volume'].fillna(0)
                
                # Ensure numeric types
                self._log_debug("Converting to numeric types...")
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Check for NaN values
                nan_count = df[numeric_cols].isna().sum().sum()
                self._log_debug(f"NaN values found: {nan_count}")
                
                # Remove NaN rows
                df = df.dropna(subset=numeric_cols)
                self._log_debug(f"DataFrame shape after dropping NaN: {df.shape}")
                
                # Validate data quality
                if len(df) < 10:
                    self._log_debug(f"❌ Insufficient data after cleaning: {len(df)} rows")
                    if attempt < self.max_retries - 1:
                        time.sleep(5)
                        continue
                    return None
                
                # Take most recent rows
                if len(df) > limit:
                    self._log_debug(f"Trimming to {limit} most recent rows")
                    df = df.tail(limit)
                
                # Reorder columns
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df = df.reset_index(drop=True)
                
                # Final statistics
                self._log_debug(f"✅ Final DataFrame shape: {df.shape}")
                self._log_debug(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                self._log_debug(f"Average volume: ${df['volume'].mean():.2f}")
                
                return df

            except requests.exceptions.Timeout:
                self._log_debug(f"⏱️ Request timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                continue
            except requests.exceptions.RequestException as e:
                self._log_debug(f"❌ Request error: {type(e).__name__}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                continue
            except Exception as e:
                self._log_debug(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
                import traceback
                self._log_debug(f"Traceback: {traceback.format_exc()}")
                if attempt < self.max_retries - 1:
                    time.sleep(3)
                    continue
                return None
        
        self._log_debug(f"❌ Failed to fetch historical data after {self.max_retries} attempts")
        return None

    def get_market_depth(self, symbol):
        """Market depth not available in free CoinGecko API"""
        self._log_debug("Market depth not available with free API tier")
        return None