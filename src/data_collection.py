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
        """Fetch historical data from CoinGecko"""
        try:
            if symbol not in self.coingecko_map:
                st.error(f"❌ Symbol {symbol} not supported")
                return None
            
            coin_id = self.coingecko_map[symbol]
            
            # Calculate days needed
            hours_needed = limit
            days = max(1, hours_needed // 24 + 1)
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if interval == '1h' else 'daily'
            }
            
            st.info(f"🔍 Fetching historical data from: {url}")
            st.info(f"📊 Parameters: {params}")
            
            response = requests.get(url, params=params, timeout=10)
            
            st.info(f"📡 Status Code: {response.status_code}")
            
            response.raise_for_status()
            
            data = response.json()
            
            if 'prices' not in data:
                st.error(f"❌ Invalid response: {data}")
                return None
            
            prices = data['prices']
            
            st.info(f"📊 Received {len(prices)} price points")
            
            # Convert to DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # CoinGecko only gives prices, so we'll simulate OHLC
            df['open'] = df['close']
            df['high'] = df['close'] * 1.001  # Simulate with small variance
            df['low'] = df['close'] * 0.999
            df['volume'] = 0  # CoinGecko free API doesn't include volume
            
            # Take only the requested number of rows
            df = df.tail(limit)
            
            # Reorder columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            st.success(f"✅ Returning {len(df)} rows of historical data")
            
            return df
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {e}")
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