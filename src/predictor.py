import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

class PricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df):
        """Create technical indicators as features"""
        df = df.copy()
        
        # Ensure all columns are numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values after conversion
        df = df.dropna()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Moving averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=10).std()
        
        # Volume indicators
        df['volume_change'] = df['volume'].pct_change()
        
        df = df.dropna()
        return df
    
    def train(self, historical_data):
        """Train the prediction model"""
        df = self.prepare_features(historical_data)
        
        feature_cols = ['returns', 'high_low_ratio', 'ma_5', 'ma_10', 
                       'ma_20', 'volatility', 'volume_change']
        
        X = df[feature_cols].values
        y = df['close'].shift(-1).dropna().values
        X = X[:-1]  # Remove last row to match y
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return self.model.score(X_scaled, y)
    
    def predict_next_price(self, historical_data):
        """Predict next price point"""
        if not self.is_trained:
            return None
        
        df = self.prepare_features(historical_data)
        
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
    
    def save_model(self, path):
        """Save trained model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }, f)
    
    def load_model(self, path):
        """Load trained model"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = data['is_trained']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False