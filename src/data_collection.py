# src/data_collection.py
# Complete fixed version with CoinGecko API

import requests
import pandas as pd
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO)
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
            url = "https://api.coingecko.com/api/v3/ping"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logger.info("✅ CoinGecko API connection successful")
                return True
            else:
                logger.error(f"❌ API returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Connection test failed: {str(e)}")
            return False

    def get_realtime_price(self, symbol):
        """Fetch current price using CoinGecko API with retries."""
        for attempt in range(self.max_retries):
            try:
                if symbol not in self.coingecko_map:
                    logger.error(f"Symbol {symbol} not found in mapping")
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

                logger.info(f"Fetching real-time price for {symbol} (attempt {attempt + 1})")
                response = requests.get(url, params=params, timeout=15)

                # Handle rate limit
                if response.status_code == 429:
                    wait_time = 15 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                if coin_id not in data or 'usd' not in data[coin_id]:
                    logger.error(f"Invalid response structure for {symbol}")
                    continue

                price_info = data[coin_id]
                
                result = {
                    'symbol': symbol,
                    'price': float(price_info['usd']),
                    'volume_24h': float(price_info.get('usd_24h_vol', 0)),
                    'change_24h': float(price_info.get('usd_24h_change', 0)),
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Successfully fetched price: ${result['price']:,.2f}")
                return result

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return None
        
        logger.error(f"Failed to fetch price after {self.max_retries} attempts")
        return None

    def get_historical_data(self, symbol, interval='1h', limit=100):
        """Fetch historical data from CoinGecko with robust error handling."""
        for attempt in range(self.max_retries):
            try:
                if symbol not in self.coingecko_map:
                    logger.error(f"Symbol {symbol} not found in mapping")
                    return None

                coin_id = self.coingecko_map[symbol]
                self._rate_limit()

                # Calculate days needed
                hours_needed = int(limit)
                days_needed = max(1, (hours_needed // 24) + 1)
                days_needed = min(days_needed, 365)  # Max 1 year
                
                logger.info(f"Fetching {days_needed} days of data for {symbol} (attempt {attempt + 1})")
                
                # Use market_chart endpoint
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': days_needed,
                    'interval': 'hourly' if days_needed <= 90 else 'daily'
                }

                response = requests.get(url, params=params, timeout=20)

                # Handle rate limit
                if response.status_code == 429:
                    wait_time = 20 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    logger.error(f"API returned status code {response.status_code}")
                    if attempt < self.max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                        continue
                    return None

                data = response.json()
                
                # Validate response
                if 'prices' not in data or not data['prices']:
                    logger.error("No price data in response")
                    if attempt < self.max_retries - 1:
                        time.sleep(5)
                        continue
                    return None

                prices = data['prices']
                volumes = data.get('total_volumes', [])
                
                logger.info(f"Received {len(prices)} price points")

                # Create DataFrames
                df_price = pd.DataFrame(prices, columns=['timestamp', 'close'])
                
                if volumes and len(volumes) > 0:
                    df_volume = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                    df = pd.merge(df_price, df_volume, on='timestamp', how='left')
                else:
                    df = df_price.copy()
                    df['volume'] = 0
                
                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Create OHLC data from close prices
                df['open'] = df['close'].shift(1).fillna(df['close'])
                df['high'] = df[['open', 'close']].max(axis=1) * 1.002
                df['low'] = df[['open', 'close']].min(axis=1) * 0.998
                
                # Fill missing volumes
                df['volume'] = df['volume'].fillna(0)
                
                # Ensure all numeric columns are float
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove any NaN rows
                df = df.dropna(subset=numeric_cols)
                
                # Validate data quality
                if len(df) < 10:
                    logger.error(f"Insufficient data after cleaning: {len(df)} rows")
                    if attempt < self.max_retries - 1:
                        time.sleep(5)
                        continue
                    return None
                
                # Take most recent rows
                if len(df) > limit:
                    df = df.tail(limit)
                
                # Reorder columns
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df = df.reset_index(drop=True)
                
                logger.info(f"Successfully prepared {len(df)} rows of historical data")
                return df

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing data: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(3)
                    continue
                return None
        
        logger.error(f"Failed to fetch historical data after {self.max_retries} attempts")
        return None

    def get_market_depth(self, symbol):
        """Market depth not available in free CoinGecko API"""
        logger.info("Market depth not available with free API tier")
        return None