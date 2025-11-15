print("Testing imports...")

try:
    import streamlit as st
    print("✅ Streamlit installed")
except:
    print("❌ Streamlit failed")

try:
    import pandas as pd
    print("✅ Pandas installed")
except:
    print("❌ Pandas failed")

try:
    from textblob import TextBlob
    print("✅ TextBlob installed")
except:
    print("❌ TextBlob failed")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("✅ VADER installed")
except:
    print("❌ VADER failed")

try:
    from binance.client import Client
    print("✅ Binance installed")
except:
    print("❌ Binance failed")

try:
    import plotly
    print("✅ Plotly installed")
except:
    print("❌ Plotly failed")

print("\n🎉 Setup verification complete!")