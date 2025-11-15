# 📈 Real-Time Stock Price & Sentiment Predictor

A production-ready application that combines real-time cryptocurrency price tracking with NLP-based sentiment analysis and machine learning predictions.

## 🚀 Features

- **Real-time Price Tracking**: Live cryptocurrency prices from Binance
- **Price Prediction**: ML-powered next-hour price forecasting
- **Sentiment Analysis**: NLP analysis of market news and social media
- **Interactive Dashboard**: Streamlit-based visualization
- **Logging System**: Track predictions and user interactions
- **Cloud-Ready**: Easy deployment to Streamlit Cloud, Render, or AWS

## 📋 Prerequisites

- Python 3.8+
- Binance API credentials (optional for enhanced features)
- Git

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-sentiment-predictor.git
cd stock-sentiment-predictor
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API credentials
```

## 🎯 Usage

Run the application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Project Structure

See folder structure above for detailed organization.

## 🔧 Configuration

Edit `config.py` to customize:
- Trading pairs
- Update intervals
- Model parameters
- Chart colors

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Visit share.streamlit.io
3. Connect repository
4. Deploy!

### Docker
```bash
docker build -t stock-predictor .
docker run -p 8501:8501 stock-predictor
```

## 📝 License

MIT License

## 👨‍💻 Author

Your Name - [GitHub](https://github.com/yourusername)
```

### 12. `deployment/Dockerfile`
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 13. `deployment/docker-compose.yml`
```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
    restart: unless-stopped
```

---

## 🎯 Quick Start Commands

```bash
# Setup
mkdir stock-sentiment-predictor
cd stock-sentiment-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run app.py

# Deploy with Docker
docker-compose up -d
```

## 📚 Next Steps

1. Create all folders and files as shown above
2. Install dependencies
3. Set up API credentials in `.env`
4. Run the application
5. Customize as needed
6. Deploy to cloud

This is a complete, production-ready codebase ready to use!