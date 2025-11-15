from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

class SentimentAnalyzer:
    """
    100% FREE Sentiment Analysis - No API Keys Required!
    
    Uses locally-installed libraries:
    - TextBlob: Rule-based sentiment analysis (completely free)
    - VADER: Specialized for social media text (completely free)
    
    Both work offline and require NO API keys or subscriptions.
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()  # FREE - no API needed
    
    def analyze_text(self, text):
        """
        Analyze sentiment of text using FREE local libraries.
        No API calls, no rate limits, no costs!
        """
        if not text or text.strip() == "":
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'vader_score': 0.0,
                'sentiment': 'neutral'
            }
        
        # TextBlob analysis (FREE - runs locally)
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # VADER analysis (FREE - runs locally)
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']  # -1 to 1
        
        # Combined sentiment score
        combined_score = (polarity + vader_compound) / 2
        
        # Classify sentiment with thresholds
        if combined_score > 0.1:
            sentiment = 'positive'
        elif combined_score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'vader_score': vader_compound,
            'combined_score': combined_score,
            'sentiment': sentiment,
            'confidence': abs(combined_score)
        }
    
    def batch_analyze(self, texts):
        """Analyze multiple texts and return aggregate sentiment"""
        if not texts:
            return {'sentiment': 'neutral', 'score': 0.0}
        
        results = [self.analyze_text(text) for text in texts]
        avg_score = np.mean([r['combined_score'] for r in results])
        
        if avg_score > 0.1:
            sentiment = 'positive'
        elif avg_score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': avg_score,
            'count': len(texts)
        }
    
    def analyze_crypto_news(self, text):
        """
        Specialized analysis for crypto/stock market text.
        Identifies common market sentiment keywords.
        """
        # Positive keywords
        positive_words = ['bullish', 'moon', 'surge', 'rally', 'breakout', 
                         'growth', 'gain', 'profit', 'up', 'high', 'pump']
        
        # Negative keywords
        negative_words = ['bearish', 'crash', 'dump', 'fall', 'drop', 'loss',
                         'down', 'low', 'risk', 'fear', 'sell-off']
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Get base sentiment
        base_sentiment = self.analyze_text(text)
        
        # Adjust based on crypto-specific keywords
        keyword_boost = (pos_count - neg_count) * 0.1
        adjusted_score = base_sentiment['combined_score'] + keyword_boost
        adjusted_score = max(-1, min(1, adjusted_score))  # Clamp to [-1, 1]
        
        if adjusted_score > 0.1:
            sentiment = 'positive'
        elif adjusted_score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            **base_sentiment,
            'adjusted_score': adjusted_score,
            'sentiment': sentiment,
            'positive_keywords': pos_count,
            'negative_keywords': neg_count
        }