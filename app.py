# app.py
# Fixed Streamlit dashboard with proper volume display and error handling

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Import custom modules
from src.data_collection import DataCollector
from src.sentiment_analyzer import SentimentAnalyzer
from src.predictor import PricePredictor
from src.logger import PredictionLogger
import config

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_components():
    collector = DataCollector()
    analyzer = SentimentAnalyzer()
    predictor = PricePredictor()
    logger = PredictionLogger()
    return collector, analyzer, predictor, logger

collector, analyzer, predictor, logger = init_components()

# Sidebar
st.sidebar.title("⚙️ Settings")
selected_symbol = st.sidebar.selectbox(
    "Select Trading Pair",
    config.TRADING_PAIRS,
    index=0
)

auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 120, 30)

# Main title
st.title(f"{config.APP_ICON} {config.APP_TITLE}")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Prediction", "💬 Sentiment", "📈 Analytics"])

# Tab 1: Dashboard
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    # Fetch real-time price
    price_data = collector.get_realtime_price(selected_symbol)

    if price_data:
        with col1:
            st.metric("Current Price", f"${price_data['price']:,.2f}")

        with col2:
            change_24h = price_data.get('change_24h', 0)
            st.metric("24h Change", f"{change_24h:+.2f}%", 
                     delta=f"{change_24h:+.2f}%")

        with col3:
            volume_24h = price_data.get('volume_24h', 0)
            if volume_24h > 1_000_000_000:
                st.metric("24h Volume", f"${volume_24h/1_000_000_000:.2f}B")
            elif volume_24h > 1_000_000:
                st.metric("24h Volume", f"${volume_24h/1_000_000:.2f}M")
            else:
                st.metric("24h Volume", f"${volume_24h:,.0f}")

        with col4:
            st.metric("Last Updated", price_data['timestamp'].strftime("%H:%M:%S"))
    else:
        st.warning("⚠️ Unable to fetch live price. Retrying...")

    # Price chart
    st.subheader("📈 Price Chart")
    hist_data = collector.get_historical_data(selected_symbol, '1h', 168)  # 7 days

    if hist_data is not None and not hist_data.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist_data['timestamp'],
            open=hist_data['open'].astype(float),
            high=hist_data['high'].astype(float),
            low=hist_data['low'].astype(float),
            close=hist_data['close'].astype(float),
            name=selected_symbol
        ))
        fig.update_layout(
            title=f"{selected_symbol} Price History (7 Days)",
            yaxis_title="Price (USD)",
            xaxis_title="Time",
            height=500,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume chart with proper data
        st.subheader("📊 Trading Volume")
        if hist_data['volume'].sum() > 0:
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=hist_data['timestamp'],
                y=hist_data['volume'],
                name="Volume",
                marker_color='rgba(50, 171, 96, 0.7)'
            ))
            fig_volume.update_layout(
                title="Trading Volume Over Time",
                yaxis_title="Volume (USD)",
                xaxis_title="Time",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        else:
            st.info("💡 Volume data is estimated. For real-time volume, check the 24h Volume metric above.")
    else:
        st.error("❌ Failed to load historical data. Please try again.")

# Tab 2: Prediction
with tab2:
    st.subheader("🔮 AI-Powered Price Prediction")

    with st.expander("ℹ️ How does the prediction work?", expanded=False):
        st.markdown("""
        ### 📊 Understanding AI Price Prediction
        
        Our machine learning model analyzes:
        - **200 hours** of historical price data
        - **Technical indicators**: Moving averages, RSI, volatility
        - **Price patterns**: Trends, momentum, support/resistance
        - **Volume analysis**: Trading activity patterns
        
        The **Random Forest algorithm** learns from past patterns to forecast the next hour's price.
        
        ⚠️ **Disclaimer**: This is a predictive tool, not financial advice. Always do your own research!
        """)

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🚀 Generate AI Prediction", type="primary", use_container_width=True):
            with st.spinner("🧠 Training AI model and generating prediction..."):
                try:
                    hist_data = collector.get_historical_data(selected_symbol, '1h', 200)

                    if hist_data is None or hist_data.empty:
                        st.error("❌ Failed to fetch sufficient historical data. Please try again.")
                    else:
                        # Train model
                        score = predictor.train(hist_data)

                        if score is not None:
                            st.success(f"✅ Model trained successfully! Accuracy (R²): {score:.4f}")

                            # Make prediction
                            prediction = predictor.predict_next_price(hist_data)

                            if prediction:
                                st.markdown("### 📊 Market Forecast")

                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("💰 Current Price", f"${prediction['current_price']:,.2f}")
                                with col_b:
                                    st.metric("🎯 Predicted Price", f"${prediction['predicted_price']:,.2f}")
                                with col_c:
                                    change = prediction['change_pct']
                                    if change > 0:
                                        signal = "📈 Bullish"
                                        color = "normal"
                                    elif change < 0:
                                        signal = "📉 Bearish"
                                        color = "inverse"
                                    else:
                                        signal = "➡️ Neutral"
                                        color = "off"

                                    st.metric("📊 Signal", signal, delta=f"{change:+.2f}%", delta_color=color)

                                # Trading signal
                                st.markdown("---")
                                if change > 1:
                                    st.success(f"🟢 **STRONG BULLISH**: AI predicts **+{change:.2f}%** increase")
                                elif change > 0.3:
                                    st.success(f"🟢 **BULLISH**: AI predicts **+{change:.2f}%** increase")
                                elif change < -1:
                                    st.error(f"🔴 **STRONG BEARISH**: AI predicts **{change:.2f}%** decrease")
                                elif change < -0.3:
                                    st.error(f"🔴 **BEARISH**: AI predicts **{change:.2f}%** decrease")
                                else:
                                    st.info(f"⚪ **NEUTRAL**: AI predicts **{abs(change):.2f}%** movement")

                                # Log prediction
                                logger.log_prediction({
                                    'symbol': selected_symbol,
                                    'current_price': prediction['current_price'],
                                    'predicted_price': prediction['predicted_price'],
                                    'sentiment': 'N/A',
                                    'sentiment_score': 0.0
                                })
                            else:
                                st.error("❌ Failed to generate prediction. Please try again.")
                        else:
                            st.error("❌ Model training failed. Not enough data or data quality issue.")
                            
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

    with col2:
        st.info("**📚 Guide:**\n\n1️⃣ Click predict button\n\n2️⃣ AI analyzes 200 hours\n\n3️⃣ Get price forecast\n\n4️⃣ View market signal")

# Tab 3: Sentiment Analysis
with tab3:
    st.subheader("💬 Market Sentiment Analysis")

    with st.expander("ℹ️ What is Market Sentiment?", expanded=False):
        st.markdown("""
        ### 📰 Understanding Sentiment Analysis
        
        **Market sentiment** = Overall investor mood toward an asset
        
        - 🟢 **Bullish** = Optimistic → Prices tend to rise
        - 🔴 **Bearish** = Pessimistic → Prices tend to fall
        - ⚪ **Neutral** = Uncertain → Sideways movement
        
        Our AI analyzes text using NLP algorithms (TextBlob + VADER) to gauge market mood.
        """)

    user_input = st.text_area(
        "📝 Paste news, tweets, or market analysis:",
        height=150,
        placeholder="Example: 'Bitcoin surges past $50K as institutional investors show strong interest...'")

    col1, col2 = st.columns([1, 3])

    with col1:
        analyze_btn = st.button("🧠 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_btn and user_input:
        with st.spinner("🔍 Analyzing sentiment..."):
            try:
                result = analyzer.analyze_text(user_input)

                st.markdown("### 📊 Sentiment Results")
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    sentiment = result['sentiment']
                    display_map = {
                        'positive': "🟢 Bullish",
                        'negative': "🔴 Bearish",
                        'neutral': "⚪ Neutral"
                    }
                    st.metric("Market Sentiment", display_map.get(sentiment, "⚪ Neutral"))

                with col_b:
                    st.metric("Polarity Score", f"{result['polarity']:.3f}")

                with col_c:
                    st.metric("VADER Score", f"{result['vader_score']:.3f}")

                # Signal interpretation
                st.markdown("---")
                combined = result['combined_score']
                if combined >= 0.5:
                    st.success("🟢 **STRONG BULLISH** - Highly optimistic sentiment")
                elif combined >= 0.1:
                    st.success("🟢 **MILD BULLISH** - Cautiously optimistic")
                elif combined <= -0.5:
                    st.error("🔴 **STRONG BEARISH** - Highly pessimistic sentiment")
                elif combined <= -0.1:
                    st.error("🔴 **MILD BEARISH** - Cautiously pessimistic")
                else:
                    st.info("⚪ **NEUTRAL** - Mixed sentiment")

                # Sentiment gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['combined_score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-1, -0.1], 'color': 'lightcoral'},
                            {'range': [-0.1, 0.1], 'color': 'lightyellow'},
                            {'range': [0.1, 1], 'color': 'lightgreen'}
                        ]
                    },
                    title={'text': "Sentiment Score"}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Log sentiment
                price_data = collector.get_realtime_price(selected_symbol)
                if price_data:
                    logger.log_prediction({
                        'symbol': selected_symbol,
                        'current_price': price_data['price'],
                        'predicted_price': None,
                        'sentiment': result['sentiment'],
                        'sentiment_score': result['combined_score']
                    })
                    
            except Exception as e:
                st.error(f"❌ Error analyzing sentiment: {str(e)}")

    elif analyze_btn and not user_input:
        st.warning("⚠️ Please enter text to analyze!")

# Tab 4: Analytics
with tab4:
    st.subheader("📈 Analytics & Logs")

    stats = logger.get_statistics()

    if stats:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Predictions", stats.get('total_predictions', 0))

        with col2:
            st.metric("Avg Sentiment", f"{stats.get('avg_sentiment_score', 0):.3f}")

        with col3:
            st.metric("Most Tracked", stats.get('most_tracked_symbol', 'N/A'))

        if 'sentiment_distribution' in stats and stats['sentiment_distribution']:
            st.markdown("### Sentiment Distribution")
            sentiment_df = pd.DataFrame(
                list(stats['sentiment_distribution'].items()),
                columns=['Sentiment', 'Count']
            )
            fig = px.pie(sentiment_df, values='Count', names='Sentiment')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Recent Activity")
    logs = logger.get_logs(50)

    if not logs.empty:
        st.dataframe(logs, use_container_width=True, height=400)
    else:
        st.info("No activity yet. Start making predictions!")

# Footer
st.markdown("---")
st.caption("Real-Time Crypto Sentiment Predictor v1.0 | Built with Streamlit & ML")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()