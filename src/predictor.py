# src/predictor.py
# Price prediction model - completely independent of Streamlit

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PricePredictor:
    """
    Machine Learning model for cryptocurrency price prediction.
    Uses Random Forest with technical indicators as features.
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = [
            'returns', 'high_low_ratio', 'ma_5', 'ma_10', 
            'ma_20', 'volatility', 'volume_change'
        ]
    
    def prepare_features(self, df):
        """
        Create technical indicators as features for the model.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated features or None if failed
        """
        try:
            if df is None or len(df) == 0:
                logger.error("No data provided for feature preparation")
                return None
            
            # Create a copy to avoid modifying original
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
            
            # Calculate technical indicators
            
            # 1. Price-based features
            df['returns'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            
            # 2. Moving averages
            df['ma_5'] = df['close'].rolling(window=5, min_periods=5).mean()
            df['ma_10'] = df['close'].rolling(window=10, min_periods=10).mean()
            df['ma_20'] = df['close'].rolling(window=20, min_periods=20).mean()
            
            # 3. Volatility (standard deviation of returns)
            df['volatility'] = df['returns'].rolling(window=10, min_periods=10).std()
            
            # 4. Volume indicators
            df['volume_change'] = df['volume'].pct_change()
            
            # Remove rows with NaN (from rolling windows)
            df = df.dropna()
            
            # Replace any infinite values
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 10:
                logger.warning("Not enough valid data after feature calculation.")
                return None
            
            logger.info(f"Successfully prepared {len(df)} rows with features")
            return df
        
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None
    
    def train(self, historical_data):
        """
        Train the prediction model on historical data.
        
        Args:
            historical_data: DataFrame with OHLCV data
            
        Returns:
            R² score of the trained model or None if failed
        """
        try:
            # Validate input data
            if historical_data is None or len(historical_data) < 30:
                logger.error("Not enough historical data to train model. Need at least 30 data points.")
                return None
            
            # Prepare features
            df = self.prepare_features(historical_data)
            
            # Check if we have enough data after feature preparation
            if df is None or len(df) < 20:
                logger.error("Insufficient data after feature engineering. Need at least 20 valid rows.")
                return None
            
            # Verify all feature columns exist
            missing_cols = [col for col in self.feature_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing feature columns: {missing_cols}")
                return None
            
            # Prepare X (features) and y (target)
            X = df[self.feature_cols].values
            y = df['close'].shift(-1).dropna().values  # Predict next price
            X = X[:-1]  # Remove last row to match y
            
            # Final validation - check for NaN or infinite values
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.error("Data contains NaN or infinite values. Cannot train model.")
                return None
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid training data available.")
                return None
            
            # Scale features and train model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Calculate R² score
            score = self.model.score(X_scaled, y)
            logger.info(f"Model trained successfully. R² score: {score:.4f}")
            return score
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
    
    def predict_next_price(self, historical_data):
        """
        Predict the next price point.
        
        Args:
            historical_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with prediction results or None if failed
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained. Cannot make prediction.")
                return None
            
            # Prepare features
            df = self.prepare_features(historical_data)
            
            if df is None or len(df) == 0:
                logger.error("Failed to prepare features for prediction.")
                return None
            
            # Get latest features
            X_latest = df[self.feature_cols].iloc[-1:].values
            X_scaled = self.scaler.transform(X_latest)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            current_price = float(df['close'].iloc[-1])
            
            # Calculate change percentage
            change_pct = ((prediction - current_price) / current_price) * 100
            
            result = {
                'predicted_price': float(prediction),
                'current_price': current_price,
                'change_pct': float(change_pct)
            }
            
            logger.info(f"Prediction: ${prediction:.2f} ({change_pct:+.2f}%)")
            return result
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def save_model(self, path):
        """
        Save trained model to disk.
        
        Args:
            path: File path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'is_trained': self.is_trained,
                    'feature_cols': self.feature_cols
                }, f)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path):
        """
        Load trained model from disk.
        
        Args:
            path: File path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = data['is_trained']
                if 'feature_cols' in data:
                    self.feature_cols = data['feature_cols']
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_feature_importance(self):
        """
        Get feature importance scores from the trained model.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.is_trained:
            return None
        
        try:
            importances = self.model.feature_importances_
            return dict(zip(self.feature_cols, importances))
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None