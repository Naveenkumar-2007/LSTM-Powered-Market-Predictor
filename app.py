import streamlit as st 
import numpy as np
import pandas as pd 
import tensorflow as tf
import pickle
import os
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta

# Import matplotlib with fallback handling
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    st.error("⚠️ Matplotlib not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.7.2"])
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True


st.title("📈 Stock Price Predictor ")

stock = st.sidebar.text_input("Enter your stock symbol", value="TSLA")
# Set start date to ensure we have enough historical data (at least 90 days for 60-day lookback)
default_start = datetime(2024, 1, 1)  # Changed to ensure sufficient historical data
start_time = st.sidebar.date_input("Start Date", value=default_start)
end_time = st.sidebar.date_input("End Date", value=datetime.today())
forecast_days = st.sidebar.number_input("Future prediction days", min_value=1, max_value=30, value=7)


df = yf.download(stock, start=start_time, end=end_time)
df = df.sort_index()

# Check if we got any data
if df.empty:
    st.error(f"❌ No data found for stock symbol '{stock}' in the specified date range.")
    st.info("Please check the stock symbol or try a different date range.")
    st.stop()

# Check if we have the required columns
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"❌ Missing required columns: {missing_columns}")
    st.info("The downloaded data doesn't contain all required OHLCV columns.")
    st.stop()

features = df[['Open','High','Low','Close','Volume']]

# Remove any rows with NaN values
features = features.dropna()
if features.empty:
    st.error("❌ No valid data after cleaning (all rows contained NaN values).")
    st.info("Please try a different stock symbol or date range.")
    st.stop()

st.success(f"✅ Successfully loaded {len(features)} days of data for {stock}")

# Check if model files exist
model_files = ["model_returns.h5", "scaler_X.pkl", "scaler_y.pkl"]
missing_files = [f for f in model_files if not os.path.exists(f)]

if missing_files:
    st.warning(f"⚠️ Model files not found: {missing_files}")
    st.info("Running in DEMO MODE - Showing data visualization without ML predictions")
    
    # Show basic stock info without predictions
    st.subheader(f"📊 Stock Data for {stock}")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${features['Close'].iloc[-1]:.2f}")
    with col2:
        current_change = features['Close'].iloc[-1] - features['Close'].iloc[-2] if len(features) > 1 else 0
        st.metric("Daily Change", f"${current_change:.2f}", f"{current_change:.2f}")
    with col3:
        st.metric("Period High", f"${features['High'].max():.2f}")
    with col4:
        st.metric("Period Low", f"${features['Low'].min():.2f}")
    
    # Show charts
    st.subheader("📈 Price History")
    st.line_chart(features[['Open', 'High', 'Low', 'Close']])
    
    st.subheader("📊 Volume")
    st.bar_chart(features['Volume'])
    
    st.subheader("📋 Recent Data")
    st.dataframe(features.tail(10), use_container_width=True)
    
    # Simple moving averages as demo "predictions"
    st.subheader("📈 Technical Analysis (Demo)")
    ma_short = features['Close'].rolling(window=5).mean()
    ma_long = features['Close'].rolling(window=20).mean()
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(features.index, features['Close'], label='Close Price', linewidth=2)
    ax.plot(features.index, ma_short, label='5-day MA', alpha=0.7)
    ax.plot(features.index, ma_long, label='20-day MA', alpha=0.7)
    ax.set_title(f'{stock} Price Analysis')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.info("🚀 **Deploy with model files to enable AI-powered predictions!**")
    st.stop()

# If we reach here, model files exist
try:
    # Clear any existing TensorFlow sessions
    tf.keras.backend.clear_session()
    
    model = load_model("model_returns.h5")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.info("Please ensure 'model_returns.h5' file exists in the current directory.")
    st.stop()

try:
    with open("scaler_X.pkl","rb") as f:
        scaler_X = pickle.load(f)
    with open("scaler_y.pkl","rb") as f:
        scaler_y = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading scalers: {str(e)}")
    st.info("Please ensure 'scaler_X.pkl' and 'scaler_y.pkl' files exist in the current directory.")
    st.stop()

# Scale Features
try:
    # Ensure we have valid numerical data
    if features.values.shape[0] == 0:
        st.error("❌ No data available for scaling.")
        st.stop()
    
    # Check for any infinite or extremely large values
    if not np.isfinite(features.values).all():
        st.warning("⚠️ Data contains infinite or NaN values. Cleaning...")
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        if features.empty:
            st.error("❌ No valid data after cleaning infinite values.")
            st.stop()
    
    scaled_X = scaler_X.transform(features.values)
except Exception as e:
    st.error(f"❌ Error scaling data: {str(e)}")
    st.info("This might be due to data format issues or scaler compatibility problems.")
    st.write("Debug info:")
    st.write(f"Features shape: {features.shape}")
    st.write(f"Features sample:\n{features.head()}")
    st.stop()

# Last look_back window (60 days)
look_back = 60

# Check if we have enough data
if len(scaled_X) < look_back:
    st.error(f"Not enough historical data. Need at least {look_back} days, but only have {len(scaled_X)} days.")
    st.info("Please select an earlier start date or choose a different stock symbol.")
    st.stop()

x_input = scaled_X[-look_back:]

# Verify the shape before proceeding (silently)
if x_input.shape[0] != look_back or x_input.shape[1] != 5:
    st.error(f"Data shape mismatch. Expected ({look_back}, 5), got {x_input.shape}")
    st.info("This might be due to insufficient trading days or data quality issues.")
    st.stop()

# -----------------------------
# Forecast Loop
# -----------------------------
last_real_price = df['Close'].iloc[-1].item()
forecast_prices = []
forecast_returns = []

for i in range(forecast_days):
    try:
        x_reshaped = np.reshape(x_input, (1, look_back, x_input.shape[1]))
        
        # Predict return (scaled) with proper error handling
        with tf.device('/CPU:0'):  # Force CPU usage to avoid GPU issues
            pred_return_scaled = model.predict(x_reshaped, verbose=0)
        
        pred_return = scaler_y.inverse_transform(pred_return_scaled)[0,0]
        
        forecast_returns.append(pred_return)
        
        # Reconstruct price
        next_price = last_real_price * (1 + pred_return)
        forecast_prices.append(next_price)
        
        # Update last_real_price
        last_real_price = next_price
        
        # Update x_input (shift window, append new OHLCV with predicted Close)
        new_row = features.iloc[-1].copy()
        new_row['Close'] = next_price
        new_scaled = scaler_X.transform([new_row.values])
        
        x_input = np.vstack([x_input[1:], new_scaled])
        
    except Exception as e:
        st.error(f" Error in prediction loop at step {i+1}: {str(e)}")
        st.info("This might be due to model compatibility or data issues.")
        break

# -----------------------------
# Build Forecast DataFrame
# -----------------------------
if len(forecast_prices) == 0:
    st.error("No forecast data generated. Please check the model and data.")
    st.stop()

forecast_dates = pd.date_range(end_time + timedelta(days=1), periods=len(forecast_prices))
forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Forecast": forecast_prices
})


st.subheader(f"Forecast for next {forecast_days} days")

# Show live price comparison with automatic updates
col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="📈 Last Historical Price", 
        value=f"${df['Close'].iloc[-1].item():.2f}",
        delta=None
    )
with col2:
    try:
        # Get fresh live price data automatically
        live_ticker = yf.Ticker(stock)
        current_live_price = live_ticker.info.get('currentPrice', live_ticker.info.get('regularMarketPrice'))
        if current_live_price:
            price_diff = current_live_price - df['Close'].iloc[-1].item()
            st.metric(
                label="💰 Current Live Price", 
                value=f"${current_live_price:.2f}",
                delta=f"{price_diff:+.2f}"
            )
        else:
            st.metric(label="💰 Current Live Price", value="Updating...", delta=None)
    except:
        st.metric(label="💰 Current Live Price", value="Updating...", delta=None)

st.write(forecast_df)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df['Close'], label="Historical Price", color="red")
ax.plot(forecast_df['Date'], forecast_df['Forecast'], label="Forecast", color="green", marker="o")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{stock} Stock Price Prediction (Return-based LSTM)")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
