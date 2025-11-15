# src/predictor.py
# Fixed predictor - removed all Streamlit dependencies

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logger = logging.getLogger(__name__)

class PricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df):
        """Create technical indicators as features"""
        try:
            if df is None or len(df) == 0:
                logger.error("No data provided for feature preparation")
                return None
                
            df = df.copy()
            
            # Ensure all columns are numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values after conversion
            df = df.dropna()
            
            # Check if we have enough data
            if len(df) < 25:
                logger.warning(f"Only {len(df)} data points available. Need at least 25 for reliable features.")
                return None
            
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            
            # Moving averages
            df['ma_5'] = df['close'].rolling(window=5, min_periods=5).mean()
            df['ma_10'] = df['close'].rolling(window=10, min_periods=10).mean()
            df['ma_20'] = df['close'].rolling(window=20, min_periods=20).mean()
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=10, min_periods=10).std()
            
            # Volume indicators
            df['volume_change'] = df['volume'].pct_change()
            
            # Remove rows with NaN (from rolling windows)
            df = df.dropna()
            
            # Replace any infinite values
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 10:
                logger.warning("Not enough valid data after feature calculation.")
                return None
            
            return df
        
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None
    
    def train(self, historical_data):
        """Train the prediction model"""
        try:
            # Validate input data
            if historical_data is None or len(historical_data) < 30:
                logger.error("Not enough historical data to train model. Need at least 30 data points.")
                return None
            
            df = self.prepare_features(historical_data)
            
            # Check if we have enough data after feature preparation
            if df is None or len(df) < 20:
                logger.error("Insufficient data after feature engineering. Need at least 20 valid rows.")
                return None
            
            feature_cols = ['returns', 'high_low_ratio', 'ma_5', 'ma_10', 
                           'ma_20', 'volatility', 'volume_change']
            
            # Verify all feature columns exist
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing feature columns: {missing_cols}")
                return None
            
            X = df[feature_cols].values
            y = df['close'].shift(-1).dropna().values
            X = X[:-1]  # Remove last row to match y
            
            # Final validation - check for NaN or infinite values
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.error("Data contains NaN or infinite values. Cannot train model.")
                return None
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid training data available.")
                return None
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            score = self.model.score(X_scaled, y)
            logger.info(f"Model trained successfully. R² score: {score:.4f}")
            return score
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
    
    def predict_next_price(self, historical_data):
        """Predict next price point"""
        try:
            if not self.is_trained:
                logger.warning("Model not trained. Cannot make prediction.")
                return None
            
            df = self.prepare_features(historical_data)
            
            if df is None or len(df) == 0:
                logger.error("Failed to prepare features for prediction.")
                return None
            
            feature_cols = ['returns', 'high_low_ratio', 'ma_5', 'ma_10', 
                           'ma_20', 'volatility', 'volume_change']
            
            X_latest = df[feature_cols].iloc[-1:].values
            X_scaled = self.scaler.transform(X_latest)
            
            prediction = self.model.predict(X_scaled)[0]
            current_price = df['close'].iloc[-1]
            
            return {
                'predicted_price': prediction,
                'current_price': current_price,
                'change_pct': ((prediction - current_price) / current_price) * 100
            }
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def save_model(self, path):
        """Save trained model"""
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'is_trained': self.is_trained
                }, f)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path):
        """Load trained model"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = data['is_trained']
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False