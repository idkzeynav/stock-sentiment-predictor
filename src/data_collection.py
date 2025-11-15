import requests
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import config
import streamlit as st
import time

class DataCollector:
    def __init__(self):
        """Initialize with CoinGecko as primary source (no geo-restrictions)"""
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
        
        st.info("✅ Using CoinGecko API (no geo-restrictions)")
        self.use_coingecko = True
        self.client = None
        self.use_api = False
    
    def get_realtime_price(self, symbol):
        """Fetch current price using CoinGecko API"""
        try:
            # Convert Binance symbol to CoinGecko ID
            if symbol not in self.coingecko_map:
                st.error(f"❌ Symbol {symbol} not supported yet. Available: {list(self.coingecko_map.keys())}")
                return None
            
            coin_id = self.coingecko_map[symbol]
            
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd'
            }
            
            st.info(f"🔍 Requesting: {url}")
            st.info(f"📊 Parameters: {params}")
            
            response = requests.get(url, params=params, timeout=10)
            
            st.info(f"📡 Status Code: {response.status_code}")
            
            response.raise_for_status()
            
            data = response.json()
            
            st.info(f"📊 Raw Response: {data}")
            
            if coin_id not in data or 'usd' not in data[coin_id]:
                st.error(f"❌ Invalid response format: {data}")
                return None
            
            price = float(data[coin_id]['usd'])
            
            st.success(f"✅ Price fetched successfully: ${price:,.2f}")
            
            return {
                'symbol': symbol,
                'price': price,
                'timestamp': datetime.now()
            }
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {e}")
            return None
        except Exception as e:
            st.error(f"❌ Error: {type(e).__name__}: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_historical_data(self, symbol, interval='1h', limit=100):
        """Fetch historical data from CoinGecko OHLC endpoint"""
        try:
            if symbol not in self.coingecko_map:
                st.error(f"❌ Symbol {symbol} not supported")
                return None
            
            coin_id = self.coingecko_map[symbol]
            
            # CoinGecko OHLC only accepts: 1, 7, 14, 30, 90, 180, 365 days
            # Map requested limit to valid days
            hours_needed = limit
            days_needed = hours_needed // 24 + 1
            
            # Choose closest valid day value that's >= days_needed
            valid_days = [1, 7, 14, 30, 90, 180, 365]
            days = next((d for d in valid_days if d >= days_needed), valid_days[-1])
            
            st.info(f"📊 Requested {limit} hours ≈ {days_needed} days, using {days} days")
            
            # Use OHLC endpoint (no auth required)
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': 'usd',
                'days': days
            }
            
            st.info(f"🔍 Fetching historical data from: {url}")
            st.info(f"📊 Parameters: {params}")
            
            response = requests.get(url, params=params, timeout=15)
            
            st.info(f"📡 Status Code: {response.status_code}")
            
            if response.status_code == 429:
                st.warning("⚠️ Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                response = requests.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                st.error(f"❌ API Error: Status {response.status_code}")
                st.error(f"Response: {response.text}")
                return None
            
            data = response.json()
            
            st.info(f"📊 Raw response type: {type(data)}")
            
            if not isinstance(data, list):
                st.error(f"❌ Expected list, got {type(data)}: {data}")
                return None
            
            st.info(f"📊 Received {len(data)} candles from API")
            
            if len(data) == 0:
                st.error("❌ No data returned from API")
                return None
            
            # CoinGecko OHLC format: [timestamp, open, high, low, close]
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['volume'] = 1000000  # Placeholder volume
            
            st.info(f"📊 DataFrame shape before processing: {df.shape}")
            
            # Convert to float
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Remove NaN
            df = df.dropna()
            
            st.info(f"📊 DataFrame shape after dropna: {df.shape}")
            
            # Take only the requested number of rows (from the end)
            if len(df) > limit:
                df = df.tail(limit)
            
            # Reorder columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            st.success(f"✅ Returning {len(df)} rows of historical data")
            
            return df
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {e}")
            st.error(f"Response text: {e.response.text if hasattr(e, 'response') else 'N/A'}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
        except Exception as e:
            st.error(f"❌ Error: {type(e).__name__}: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_market_depth(self, symbol):
        """Market depth not available in CoinGecko free API"""
        st.warning("⚠️ Market depth not available with CoinGecko API")
        return None