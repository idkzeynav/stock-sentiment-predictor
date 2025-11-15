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
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 120, 60)

# Main title
st.title(f"{config.APP_ICON} {config.APP_TITLE}")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Prediction", "💬 Sentiment", "📈 Analytics"])

# Tab 1: Dashboard
with tab1:
    col1, col2, col3 = st.columns(3)
    
    # Fetch real-time price
    price_data = collector.get_realtime_price(selected_symbol)
    
    if price_data:
        with col1:
            st.metric("Current Price", f"${price_data['price']:,.2f}")
        
        with col2:
            # Fetch historical data for change calculation
            hist_data = collector.get_historical_data(selected_symbol, '1h', 24)
            if hist_data is not None and len(hist_data) > 1:
                price_change = ((price_data['price'] - float(hist_data['close'].iloc[0])) / 
                              float(hist_data['close'].iloc[0])) * 100
                st.metric("24h Change", f"{price_change:+.2f}%", 
                         delta=f"{price_change:+.2f}%")
            else:
                st.metric("24h Change", "N/A")
        
        with col3:
            st.metric("Last Updated", price_data['timestamp'].strftime("%H:%M:%S"))
    
    # Price chart
    st.subheader("📈 Price Chart")
    hist_data = collector.get_historical_data(selected_symbol, '1h', 100)
    
    if hist_data is not None:
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
            title=f"{selected_symbol} Price History",
            yaxis_title="Price (USDT)",
            xaxis_title="Time",
            height=500
        )
        st.plotly_chart(fig, width="stretch")

    
    # Volume chart
    if hist_data is not None:
        fig_volume = px.bar(
            hist_data, 
            x='timestamp', 
            y='volume',
            title="Trading Volume"
        )
        fig_volume.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

# Tab 2: Prediction
with tab2:
    st.subheader("🔮 AI-Powered Price Prediction")
    
    # Information banner
    with st.expander("ℹ️ What am I seeing? Click to understand the prediction", expanded=False):
        st.markdown("""
        ### 📊 Understanding Your Price Prediction
        
        This **AI-powered prediction engine** analyzes historical market data to forecast the next price movement for {}.
        
        #### 🎯 What You're Seeing:
        
        **Current Price:** The live market price right now
        
        **Predicted Price:** Our AI's forecast for the next hour based on:
        - Recent price movements and trends
        - Trading volume patterns
        - Market volatility indicators
        - Moving averages (5, 10, 20 periods)
        - Historical price relationships
        
        **Expected Change:** The percentage difference between current and predicted price
        - 🟢 **Positive %** = Bullish signal (price expected to rise)
        - 🔴 **Negative %** = Bearish signal (price expected to fall)
        - ⚪ **Near 0%** = Neutral/Consolidation (sideways movement)
        
        #### 🤖 How It Works:
        
        1. **Data Collection:** Fetches 200 hours of historical price data
        2. **Feature Engineering:** Calculates technical indicators (RSI-like signals, volatility, momentum)
        3. **ML Training:** Random Forest algorithm learns patterns from past data
        4. **Prediction:** AI forecasts the next probable price point
        5. **Accuracy Score:** R² score shows model reliability (higher is better)
        
        #### ⚠️ Trading Advisory:
        
        - This is a **predictive tool**, not financial advice
        - Predictions are based on technical analysis only
        - Always combine with fundamental analysis and risk management
        - Past performance doesn't guarantee future results
        - Use stop-losses and proper position sizing
        
        #### 💡 Pro Tip:
        Combine this prediction with the **Sentiment Analysis** tab for a complete view of market conditions!
        """.format(selected_symbol))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Generate AI Prediction", type="primary", use_container_width=True):
            with st.spinner("🧠 Training AI model on historical data..."):
                hist_data = collector.get_historical_data(selected_symbol, '1h', 200)
                
                if hist_data is not None:
                    # Train model
                    score = predictor.train(hist_data)
                    st.success(f"✅ Model trained successfully! Accuracy Score (R²): {score:.4f}")
                    
                    # Make prediction
                    prediction = predictor.predict_next_price(hist_data)
                    
                    if prediction:
                        st.markdown("### 📊 Market Forecast")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("💰 Current Price", f"${prediction['current_price']:.2f}")
                        with col_b:
                            st.metric("🎯 AI Predicted Price", f"${prediction['predicted_price']:.2f}")
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
                            
                            st.metric("📊 Market Signal", signal,
                                    delta=f"{change:+.2f}%",
                                    delta_color=color)
                        
                        # Trading signal
                        st.markdown("---")
                        if change > 0.5:
                            st.success(f"🟢 **BULLISH SIGNAL:** AI predicts a **{change:.2f}% increase**. Price may move upward.")
                        elif change < -0.5:
                            st.error(f"🔴 **BEARISH SIGNAL:** AI predicts a **{abs(change):.2f}% decrease**. Price may move downward.")
                        else:
                            st.info(f"⚪ **NEUTRAL SIGNAL:** AI predicts **{abs(change):.2f}% movement**. Market may consolidate.")
                        
                        # Log prediction
                        logger.log_prediction({
                            'symbol': selected_symbol,
                            'current_price': prediction['current_price'],
                            'predicted_price': prediction['predicted_price'],
                            'sentiment': 'N/A',
                            'sentiment_score': 0.0
                        })
    
    with col2:
        st.info("**🎓 Quick Guide:**\n\n"
                "1️⃣ Click 'Generate AI Prediction'\n\n"
                "2️⃣ AI analyzes 200 hours of data\n\n"
                "3️⃣ Get price forecast + market signal\n\n"
                "4️⃣ See bullish/bearish/neutral outlook\n\n"
                "💡 **Tip:** Higher accuracy score = more reliable prediction")

# Tab 3: Sentiment Analysis
with tab3:
    st.subheader("💬 Market Sentiment Analysis")
    
    # Information banner
    with st.expander("ℹ️ What is Market Sentiment? Click to learn", expanded=False):
        st.markdown("""
        ### 📰 Understanding Market Sentiment
        
        **Market sentiment** is the overall attitude of investors toward a particular asset or market. It's the "mood" of the market.
        
        #### 🎯 How Sentiment Affects Prices:
        
        - **🟢 Bullish (Positive):** Optimistic outlook → More buyers → Price tends to rise
        - **🔴 Bearish (Negative):** Pessimistic outlook → More sellers → Price tends to fall  
        - **⚪ Neutral (Uncertainty):** Mixed signals → Sideways movement → Wait-and-see mode
        
        #### 🧠 What This Tool Does:
        
        Our **AI Sentiment Engine** analyzes text from:
        - News articles about crypto/stocks
        - Social media discussions
        - Market commentary
        - Press releases
        
        It uses **dual NLP algorithms** (TextBlob + VADER) to determine if the text is bullish, bearish, or neutral.
        
        #### 📊 Understanding Your Results:
        
        **Sentiment Score:** Ranges from -1 (extremely bearish) to +1 (extremely bullish)
        - **+0.5 to +1.0** = Strong Bullish 🟢
        - **+0.1 to +0.5** = Mild Bullish 🟢
        - **-0.1 to +0.1** = Neutral ⚪
        - **-0.5 to -0.1** = Mild Bearish 🔴
        - **-1.0 to -0.5** = Strong Bearish 🔴
        
        **Polarity:** How positive/negative the language is
        
        **VADER Score:** Specialized score for social media/informal text
        
        #### 💡 Trading Strategy:
        
        Smart traders combine **sentiment + technical analysis**:
        1. Check this tab for market mood (bullish/bearish)
        2. Check Prediction tab for AI price forecast
        3. Make informed decisions with both signals aligned
        
        **Example:** If sentiment is bullish AND AI predicts price increase → Strong buy signal
        """)
    
    user_input = st.text_area(
        "📝 Paste market news, tweets, or analysis here:",
        height=150,
        placeholder="Example: 'Bitcoin surges past $50K as institutional investors show strong interest. Market analysts predict continued bullish momentum with potential breakout to new all-time highs...'"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analyze_btn = st.button("🧠 Analyze Sentiment", type="primary", use_container_width=True)
    
    if analyze_btn and user_input:
        with st.spinner("🔍 Analyzing market sentiment..."):
            result = analyzer.analyze_text(user_input)
            
            # Display results
            st.markdown("### 📊 Sentiment Analysis Results")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                sentiment = result['sentiment']
                if sentiment == 'positive':
                    display_sentiment = "🟢 Bullish"
                    emoji = "😊📈"
                elif sentiment == 'negative':
                    display_sentiment = "🔴 Bearish"
                    emoji = "😟📉"
                else:
                    display_sentiment = "⚪ Neutral"
                    emoji = "😐➡️"
                
                st.metric(
                    "Market Sentiment",
                    display_sentiment,
                    help=f"{emoji} Investor mood indicator"
                )
            
            with col_b:
                st.metric("Polarity Score", f"{result['polarity']:.3f}",
                         help="Range: -1 (very negative) to +1 (very positive)")
            
            with col_c:
                st.metric("VADER Score", f"{result['vader_score']:.3f}",
                         help="Social media sentiment strength")
            
            # Market signal interpretation
            st.markdown("---")
            combined = result['combined_score']
            if combined >= 0.5:
                st.success("🟢 **STRONG BULLISH (Positive)** - Market sentiment is highly optimistic. Investors are confident. Price may rise.")
            elif combined >= 0.1:
                st.success("🟢 **MILD BULLISH (Positive)** - Market sentiment is cautiously optimistic. More buyers than sellers.")
            elif combined <= -0.5:
                st.error("🔴 **STRONG BEARISH (Negative)** - Market sentiment is highly pessimistic. Investors are worried. Price may fall.")
            elif combined <= -0.1:
                st.error("🔴 **MILD BEARISH (Negative)** - Market sentiment is cautiously pessimistic. More sellers than buyers.")
            else:
                st.info("⚪ **NEUTRAL (Uncertainty)** - Market sentiment is mixed. No clear direction. Investors are waiting for signals.")
            
            # Visualization
            st.markdown("### 📈 Sentiment Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['combined_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.1], 'color': config.CHART_COLORS['negative']},
                        {'range': [-0.1, 0.1], 'color': config.CHART_COLORS['neutral']},
                        {'range': [0.1, 1], 'color': config.CHART_COLORS['positive']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': result['combined_score']
                    }
                },
                title={'text': "Overall Market Sentiment"}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")
            
            # Trading recommendation
            st.markdown("### 💡 What This Means For Trading")
            if result['sentiment'] == 'positive':
                st.markdown("""
                - ✅ Consider **long positions** (buying)
                - ✅ Good time to **hold existing positions**
                - ⚠️ Watch for **overextension** if too bullish
                - 💡 Combine with technical analysis for entry points
                """)
            elif result['sentiment'] == 'negative':
                st.markdown("""
                - ⚠️ Consider **short positions** or **selling**
                - ⚠️ Protect profits with **stop losses**
                - 💡 Wait for sentiment improvement before buying
                - 📊 Look for **oversold conditions** as reversal signals
                """)
            else:
                st.markdown("""
                - 📊 Market is in **consolidation phase**
                - ⏳ Wait for **clearer signals** before trading
                - 🎯 Good time for **range trading strategies**
                - 📈 Watch for **breakout patterns** in either direction
                """)
            
            # Log sentiment
            price_data = collector.get_realtime_price(selected_symbol)
            if price_data:
                logger.log_prediction({
                    'symbol': selected_symbol,
                    'current_price': price_data['price'],
                    'predicted_price': None,
                    'sentiment': result['sentiment'],
                    'sentiment_score': result['combined_score'],
                    'user_input': user_input[:100]
                })
    
    elif analyze_btn and not user_input:
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Sample texts for testing
    with st.expander("📝 Try Sample Market Texts"):
        st.markdown("""
        **Bullish Example:**
        "Bitcoin reaches new all-time high as institutional adoption accelerates. Major banks announce crypto trading services. Market momentum is extremely strong with record trading volumes."
        
        **Bearish Example:**
        "Cryptocurrency market crashes amid regulatory concerns. Investors panic sell as prices plummet. Analysts warn of further downside with weak technical indicators."
        
        **Neutral Example:**
        "Bitcoin trading sideways around $40K. Market participants await Federal Reserve decision. Mixed signals from technical and fundamental indicators."
        """)
# # Tab 2: Prediction
# with tab2:
#     st.subheader("🔮 Price Prediction")
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         if st.button("🚀 Train & Predict", type="primary"):
#             with st.spinner("Training model..."):
#                 hist_data = collector.get_historical_data(selected_symbol, '1h', 200)
                
#                 if hist_data is not None:
#                     # Train model
#                     score = predictor.train(hist_data)
#                     st.success(f"Model trained! R² Score: {score:.4f}")
                    
#                     # Make prediction
#                     prediction = predictor.predict_next_price(hist_data)
                    
#                     if prediction:
#                         st.markdown("### 📊 Prediction Results")
                        
#                         col_a, col_b, col_c = st.columns(3)
#                         with col_a:
#                             st.metric("Current Price", f"${prediction['current_price']:.2f}")
#                         with col_b:
#                             st.metric("Predicted Price", f"${prediction['predicted_price']:.2f}")
#                         with col_c:
#                             st.metric("Expected Change", 
#                                     f"{prediction['change_pct']:+.2f}%",
#                                     delta=f"{prediction['change_pct']:+.2f}%")
                        
#                         # Log prediction
#                         logger.log_prediction({
#                             'symbol': selected_symbol,
#                             'current_price': prediction['current_price'],
#                             'predicted_price': prediction['predicted_price'],
#                             'sentiment': 'N/A',
#                             'sentiment_score': 0.0
#                         })
    
#     with col2:
#         st.info("**How it works:**\n\n"
#                 "1. Fetches historical data\n"
#                 "2. Trains Random Forest model\n"
#                 "3. Predicts next price point\n"
#                 "4. Shows expected change")

# # Tab 3: Sentiment Analysis
# with tab3:
#     st.subheader("💬 Sentiment Analysis")
    
#     user_input = st.text_area(
#         "Enter market news or social media text:",
#         height=150,
#         placeholder="e.g., Bitcoin reaches new all-time high as institutional adoption increases..."
#     )
    
#     col1, col2 = st.columns([1, 3])
    
#     with col1:
#         analyze_btn = st.button("🧠 Analyze Sentiment", type="primary")
    
#     if analyze_btn and user_input:
#         with st.spinner("Analyzing sentiment..."):
#             result = analyzer.analyze_text(user_input)
            
#             # Display results
#             st.markdown("### 📊 Sentiment Results")
            
#             col_a, col_b, col_c = st.columns(3)
            
#             with col_a:
#                 sentiment_emoji = {
#                     'positive': '😊', 
#                     'negative': '😟', 
#                     'neutral': '😐'
#                 }
#                 st.metric(
#                     "Sentiment",
#                     result['sentiment'].capitalize(),
#                     help=sentiment_emoji.get(result['sentiment'])
#                 )
            
#             with col_b:
#                 st.metric("Polarity", f"{result['polarity']:.3f}")
            
#             with col_c:
#                 st.metric("VADER Score", f"{result['vader_score']:.3f}")
            
#             # Visualization
#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=result['combined_score'],
#                 domain={'x': [0, 1], 'y': [0, 1]},
#                 gauge={
#                     'axis': {'range': [-1, 1]},
#                     'bar': {'color': "darkblue"},
#                     'steps': [
#                         {'range': [-1, -0.1], 'color': config.CHART_COLORS['negative']},
#                         {'range': [-0.1, 0.1], 'color': config.CHART_COLORS['neutral']},
#                         {'range': [0.1, 1], 'color': config.CHART_COLORS['positive']}
#                     ],
#                     'threshold': {
#                         'line': {'color': "red", 'width': 4},
#                         'thickness': 0.75,
#                         'value': result['combined_score']
#                     }
#                 },
#                 title={'text': "Sentiment Score"}
#             ))
#             fig.update_layout(height=300)
#             st.plotly_chart(fig, use_container_width=True)
            
#             # Log sentiment
#             price_data = collector.get_realtime_price(selected_symbol)
#             if price_data:
#                 logger.log_prediction({
#                     'symbol': selected_symbol,
#                     'current_price': price_data['price'],
#                     'predicted_price': None,
#                     'sentiment': result['sentiment'],
#                     'sentiment_score': result['combined_score'],
#                     'user_input': user_input[:100]
#                 })

# Tab 4: Analytics
with tab4:
    st.subheader("📈 Analytics & Logs")
    
    # Statistics
    stats = logger.get_statistics()
    
    if stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", stats.get('total_predictions', 0))
        
        with col2:
            st.metric("Avg Sentiment Score", 
                     f"{stats.get('avg_sentiment_score', 0):.3f}")
        
        with col3:
            st.metric("Most Tracked", stats.get('most_tracked_symbol', 'N/A'))
        
        # Sentiment distribution
        if 'sentiment_distribution' in stats:
            st.markdown("### Sentiment Distribution")
            sentiment_df = pd.DataFrame(
                list(stats['sentiment_distribution'].items()),
                columns=['Sentiment', 'Count']
            )
            fig = px.pie(sentiment_df, values='Count', names='Sentiment',
                        color='Sentiment',
                        color_discrete_map={
                            'positive': config.CHART_COLORS['positive'],
                            'negative': config.CHART_COLORS['negative'],
                            'neutral': config.CHART_COLORS['neutral']
                        })
            st.plotly_chart(fig, width="stretch")
    
    # Recent logs
    st.markdown("### 📋 Recent Activity")
    logs = logger.get_logs(50)
    
    if not logs.empty:
        st.dataframe(logs, width="stretch", height=400)
    else:
        st.info("No activity logged yet. Start making predictions!")

# Footer
st.markdown("---")
st.markdown(
    """
    
        Real-Time Stock Sentiment Predictor v1.0 | Built with Streamlit & ML
    
    """,
    unsafe_allow_html=True
)

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()