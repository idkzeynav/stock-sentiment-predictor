import requests
import pandas as pd
from binance.client import Client
from datetime import datetime
import config
import streamlit as st

class DataCollector:
    def __init__(self):
        """Initialize Binance client with error handling for Streamlit Cloud"""
        try:
            # Try to initialize with API keys if available
            api_key = config.BINANCE_API_KEY if config.BINANCE_API_KEY else None
            api_secret = config.BINANCE_API_SECRET if config.BINANCE_API_SECRET else None
            
            # Initialize without ping to avoid connection errors on startup
            self.client = Client(api_key, api_secret, requests_params={'timeout': 20})
            self.client.ping()  # Test connection
            self.use_api = True
        except Exception as e:
            # Fallback to public API (no authentication needed)
            st.warning("⚠️ Using public Binance API (no authentication). Some features may be limited.")
            self.client = None
            self.use_api = False
    
    def get_realtime_price(self, symbol):
        """Fetch current price for a trading pair"""
        try:
            if self.use_api and self.client:
                st.info("🔐 Using authenticated Binance client")
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                st.info(f"📊 Client response: {ticker}")
                return {
                    'symbol': symbol,
                    'price': float(ticker['price']),
                    'timestamp': datetime.now()
                }
            else:
                # Fallback to public REST API
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                
                # Debug: Show what we're requesting
                st.info(f"🔍 Requesting: {url}")
                
                response = requests.get(url, timeout=10)
                
                # Debug: Show status code
                st.info(f"📡 Status Code: {response.status_code}")
                
                response.raise_for_status()  # Raise error for bad status
                
                # Debug: Show raw response
                raw_data = response.text
                st.info(f"📡 Raw API Response: {raw_data}")
                
                data = response.json()
                
                # Debug: Show parsed data type and content
                st.info(f"📊 Response type: {type(data)}")
                st.info(f"📊 Is dict? {isinstance(data, dict)}")
                st.info(f"📊 Is list? {isinstance(data, list)}")
                
                # Handle both dict and list responses
                if isinstance(data, list):
                    st.info(f"📊 List length: {len(data)}")
                    if len(data) == 0:
                        st.error("❌ API returned empty list")
                        return None
                    # Take first item if list
                    data = data[0]
                    st.info(f"📊 Extracted from list: {data}")
                
                # Ensure data is a dict now
                if not isinstance(data, dict):
                    st.error(f"❌ Expected dict, got {type(data)}")
                    st.error(f"Data: {data}")
                    return None
                
                # Show available keys
                st.info(f"📊 Available keys: {list(data.keys())}")
                
                # Check if 'price' key exists
                if 'price' not in data:
                    st.error(f"❌ 'price' key not found!")
                    st.error(f"Full response: {data}")
                    return None
                
                # Successfully got price
                price = float(data['price'])
                st.success(f"✅ Price fetched successfully: ${price:,.2f}")
                
                return {
                    'symbol': symbol,
                    'price': price,
                    'timestamp': datetime.now()
                }
                
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error: {e}")
            st.error(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            st.error(f"❌ Error parsing data: {type(e).__name__}: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error: {type(e).__name__}: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_historical_data(self, symbol, interval='1h', limit=100):
        """Fetch historical klines data"""
        try:
            if self.use_api and self.client:
                # Use authenticated client
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
            else:
                # Fallback to public REST API
                url = f"https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': limit
                }
                response = requests.get(url, params=params, timeout=10)
                klines = response.json()
            
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
            st.error(f"Error fetching historical data: {e}")
            return None
    
    def get_market_depth(self, symbol):
        """Get order book depth"""
        try:
            if self.use_api and self.client:
                depth = self.client.get_order_book(symbol=symbol, limit=10)
            else:
                # Fallback to public REST API
                url = f"https://api.binance.com/api/v3/depth"
                params = {'symbol': symbol, 'limit': 10}
                response = requests.get(url, params=params, timeout=10)
                depth = response.json()
            
            return {
                'bids': depth['bids'][:5],
                'asks': depth['asks'][:5]
            }
        except Exception as e:
            st.error(f"Error fetching market depth: {e}")
            return None