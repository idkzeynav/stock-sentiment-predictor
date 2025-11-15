# src/data_collection.py
# Fixed for CoinGecko FREE tier (no interval parameter)

import requests
import pandas as pd
from datetime import datetime
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
        self.min_request_interval = 2.5
        self.max_retries = 3
        self.debug_log = []
        
        logger.info("DataCollector initialized (Free tier mode)")

    def _add_debug(self, msg):
        """Add message to debug log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {msg}"
        self.debug_log.append(log_msg)
        logger.info(msg)
        if len(self.debug_log) > 50:
            self.debug_log = self.debug_log[-50:]

    def get_debug_info(self):
        """Return all debug messages"""
        return self.debug_log if self.debug_log else ["No debug information yet"]

    def clear_debug_info(self):
        """Clear debug log"""
        self.debug_log = []
        self._add_debug("Debug log cleared")

    def _rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def test_connection(self):
        """Test if CoinGecko API is accessible"""
        try:
            self._add_debug("Testing CoinGecko API connection...")
            url = "https://api.coingecko.com/api/v3/ping"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                self._add_debug("✅ CoinGecko API is accessible")
                return True
            else:
                self._add_debug(f"❌ API returned status {response.status_code}")
                return False
        except Exception as e:
            self._add_debug(f"❌ Connection test failed: {str(e)}")
            return False

    def get_realtime_price(self, symbol):
        """Fetch current price using CoinGecko API"""
        for attempt in range(self.max_retries):
            try:
                if symbol not in self.coingecko_map:
                    self._add_debug(f"❌ Unknown symbol: {symbol}")
                    return None

                coin_id = self.coingecko_map[symbol]
                self._add_debug(f"Fetching price for {symbol} ({coin_id}) - attempt {attempt + 1}")
                
                self._rate_limit()
                
                url = "https://api.coingecko.com/api/v3/simple/price"
                params = {
                    'ids': coin_id, 
                    'vs_currencies': 'usd',
                    'include_24hr_vol': 'true',
                    'include_24hr_change': 'true'
                }

                response = requests.get(url, params=params, timeout=15)

                if response.status_code == 429:
                    wait_time = 20 * (attempt + 1)
                    self._add_debug(f"⚠️ Rate limited! Waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    self._add_debug(f"❌ Status {response.status_code}: {response.text[:100]}")
                    continue

                data = response.json()

                if coin_id not in data or 'usd' not in data[coin_id]:
                    self._add_debug(f"❌ Invalid response structure")
                    continue

                price_info = data[coin_id]
                result = {
                    'symbol': symbol,
                    'price': float(price_info['usd']),
                    'volume_24h': float(price_info.get('usd_24h_vol', 0)),
                    'change_24h': float(price_info.get('usd_24h_change', 0)),
                    'timestamp': datetime.now()
                }
                
                self._add_debug(f"✅ Price: ${result['price']:,.2f}")
                return result

            except Exception as e:
                self._add_debug(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(3 * (attempt + 1))
        
        self._add_debug("❌ All attempts failed for realtime price")
        return None

    def get_historical_data(self, symbol, interval='1h', limit=100):
        """
        Fetch historical data from CoinGecko FREE tier
        Note: Free tier doesn't support interval parameter, so we get what we can
        """
        self._add_debug(f"=== Starting historical data fetch for {symbol} ===")
        
        for attempt in range(self.max_retries):
            try:
                if symbol not in self.coingecko_map:
                    self._add_debug(f"❌ Unknown symbol: {symbol}")
                    return None

                coin_id = self.coingecko_map[symbol]
                
                # Free tier gives us automatic granularity based on days:
                # 1 day = 5 minute intervals (288 points)
                # 2-90 days = hourly intervals
                # 91+ days = daily intervals
                
                # For hourly data, request 2-7 days (free tier will give hourly automatically)
                hours_needed = int(limit)
                
                if hours_needed <= 24:
                    days_needed = 1  # Will get 5-min data, we'll resample
                elif hours_needed <= 168:  # 7 days
                    days_needed = max(2, (hours_needed // 24) + 1)
                else:
                    days_needed = min(90, (hours_needed // 24) + 1)  # Max 90 for hourly
                
                self._add_debug(f"Requesting {days_needed} days (for ~{hours_needed} hours) - attempt {attempt + 1}")
                
                self._rate_limit()

                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': days_needed
                    # NO interval parameter for free tier!
                }
                
                self._add_debug(f"URL: {url}")
                self._add_debug(f"Params: days={days_needed} (free tier auto-granularity)")

                response = requests.get(url, params=params, timeout=20)
                self._add_debug(f"Response status: {response.status_code}")

                if response.status_code == 429:
                    wait_time = 25 * (attempt + 1)
                    self._add_debug(f"⚠️ Rate limited! Waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue

                if response.status_code == 401:
                    self._add_debug(f"❌ Unauthorized (401) - API may require authentication")
                    self._add_debug(f"Response: {response.text[:200]}")
                    return None

                if response.status_code != 200:
                    self._add_debug(f"❌ HTTP {response.status_code}: {response.text[:200]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                        continue
                    return None

                data = response.json()
                
                if 'prices' not in data:
                    self._add_debug(f"❌ No 'prices' in response. Keys: {list(data.keys())}")
                    return None
                
                if not data['prices']:
                    self._add_debug("❌ Empty prices array")
                    return None

                prices = data['prices']
                volumes = data.get('total_volumes', [])
                
                self._add_debug(f"✅ Received {len(prices)} price points, {len(volumes)} volume points")

                # Build DataFrame
                df_price = pd.DataFrame(prices, columns=['timestamp', 'close'])
                
                if volumes:
                    df_volume = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                    df = pd.merge(df_price, df_volume, on='timestamp', how='left')
                else:
                    df = df_price.copy()
                    df['volume'] = 0
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # If we got 5-min data but need hourly, resample
                if days_needed == 1 and len(df) > 100:
                    self._add_debug(f"Resampling {len(df)} 5-min points to hourly data")
                    df = df.set_index('timestamp')
                    df_resampled = df.resample('1H').agg({
                        'close': 'last',
                        'volume': 'sum'
                    }).reset_index()
                    df_resampled['open'] = df.resample('1H')['close'].first().values
                    df_resampled['high'] = df.resample('1H')['close'].max().values
                    df_resampled['low'] = df.resample('1H')['close'].min().values
                    df = df_resampled
                    self._add_debug(f"Resampled to {len(df)} hourly points")
                else:
                    # Create OHLC from close prices
                    df['open'] = df['close'].shift(1).fillna(df['close'])
                    df['high'] = df[['open', 'close']].max(axis=1) * 1.002
                    df['low'] = df[['open', 'close']].min(axis=1) * 0.998
                
                df['volume'] = df['volume'].fillna(0)
                
                # Convert to numeric
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=['close'])  # Keep rows with valid close price
                
                if len(df) < 10:
                    self._add_debug(f"❌ Only {len(df)} valid rows after cleaning")
                    if attempt < self.max_retries - 1:
                        continue
                    return None
                
                # Take most recent rows
                if len(df) > limit:
                    df = df.tail(limit)
                
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df = df.reset_index(drop=True)
                
                self._add_debug(f"✅ Returning {len(df)} rows")
                self._add_debug(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                self._add_debug(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                
                return df

            except Exception as e:
                self._add_debug(f"❌ Exception in attempt {attempt + 1}: {type(e).__name__}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                continue
        
        self._add_debug("❌ All attempts exhausted - returning None")
        return None

    def get_market_depth(self, symbol):
        """Market depth not available in free tier"""
        self._add_debug("Market depth not available with free CoinGecko API")
        return None