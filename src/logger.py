import pandas as pd
import os
from datetime import datetime
import config

class PredictionLogger:
    def __init__(self, log_path=config.LOG_PATH):
        self.log_path = log_path
        self._ensure_log_exists()
    
    def _ensure_log_exists(self):
        """Create log file if it doesn't exist"""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        if not os.path.exists(self.log_path):
            df = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'current_price', 'predicted_price',
                'sentiment', 'sentiment_score', 'user_input'
            ])
            df.to_csv(self.log_path, index=False)
    
    def log_prediction(self, data):
        """Log a prediction event"""
        log_entry = {
            'timestamp': datetime.now(),
            'symbol': data.get('symbol'),
            'current_price': data.get('current_price'),
            'predicted_price': data.get('predicted_price'),
            'sentiment': data.get('sentiment'),
            'sentiment_score': data.get('sentiment_score'),
            'user_input': data.get('user_input', '')
        }
        
        df = pd.DataFrame([log_entry])
        df.to_csv(self.log_path, mode='a', header=False, index=False)
    
    def get_logs(self, limit=100):
        """Retrieve recent logs"""
        try:
            df = pd.read_csv(self.log_path)
            return df.tail(limit)
        except Exception as e:
            print(f"Error reading logs: {e}")
            return pd.DataFrame()
    
    def get_statistics(self):
        """Get prediction statistics"""
        try:
            df = pd.read_csv(self.log_path)
            
            if df.empty:
                return {}
            
            stats = {
                'total_predictions': len(df),
                'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
                'avg_sentiment_score': df['sentiment_score'].mean(),
                'most_tracked_symbol': df['symbol'].mode()[0] if not df.empty else None
            }
            
            return stats
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {}