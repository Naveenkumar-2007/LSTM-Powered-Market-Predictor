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
# Use a more conservative date range to avoid data availability issues
today = datetime.today()
# Go back 2 years to ensure we have plenty of data
default_start = today - pd.Timedelta(days=730)  # 2 years of data
start_time = st.sidebar.date_input("Start Date", value=default_start)
end_time = st.sidebar.date_input("End Date", value=today)

# Validate date range
if start_time >= end_time:
    st.error("❌ Start date must be before end date!")
    st.stop()

# Check if date range is sufficient for ML predictions
days_requested = (end_time - start_time).days
if days_requested < 90:
    st.warning(f"⚠️ You've selected only {days_requested} days. For better ML predictions, consider using at least 90 days of historical data.")

forecast_days = st.sidebar.number_input("Future prediction days", min_value=1, max_value=30, value=7)


try:
    st.info(f"📡 Fetching data for {stock.upper()} from {start_time} to {end_time}...")
    
    # Add progress indicator
    with st.spinner('Downloading stock data...'):
        # Try downloading with different parameters to handle edge cases
        try:
            df = yf.download(stock.upper(), start=start_time, end=end_time, progress=False, auto_adjust=True, prepost=False, threads=True)
        except Exception as first_attempt:
            # Fallback: try without some advanced options
            st.info("Retrying with alternative download method...")
            df = yf.download(stock.upper(), start=start_time, end=end_time, progress=False)
    
    if df is None or df.empty:
        st.error(f"❌ No data found for stock symbol '{stock.upper()}' in the specified date range.")
        st.info("**Possible solutions:**")
        st.info("• Check if the stock symbol is correct (e.g., TSLA, AAPL, MSFT)")
        st.info("• Try a different date range (some stocks may not have data for all periods)")
        st.info("• Ensure you have internet connectivity")
        st.info("• Stock markets may be closed or data might be delayed")
        st.info("• Some symbols may require exchange suffixes (e.g., BTC-USD for Bitcoin)")
        
        # Suggest alternative symbols
        st.info("**Popular symbols to try:** AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META")
        st.stop()
    
    df = df.sort_index()
    st.success(f"✅ Successfully loaded {len(df)} days of data for {stock.upper()}")
    
except Exception as e:
    st.error(f"❌ Error downloading data: {str(e)}")
    st.info("**This could be due to:**")
    st.info("• Network connectivity issues")
    st.info("• Yahoo Finance API limitations or temporary outages")
    st.info("• Invalid stock symbol format")
    st.info("• Market data unavailability for the selected period")
    st.info("• Rate limiting (try again in a few moments)")
    st.info("**Try:** Check your internet connection and verify the stock symbol")
    
    # Suggest immediate alternatives
    st.info("**Quick test:** Try entering 'AAPL' or 'MSFT' to verify the app is working")
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
    
    # Show charts including today's data
    st.subheader("📈 Price History (Including Today)")
    
    # Get today's live price if available
    try:
        live_ticker = yf.Ticker(stock)
        live_info = live_ticker.info
        current_price = live_info.get('currentPrice', live_info.get('regularMarketPrice'))
        
        if current_price:
            # Create extended dataframe with today's estimated price
            today = datetime.today().date()
            last_date = features.index[-1].date()
            
            if today > last_date:
                st.info(f"📅 **Today's Live Price**: ${current_price:.2f}")
                
                # Add simple trend prediction for demo
                recent_trend = features['Close'].tail(5).pct_change().mean()
                tomorrow_estimate = current_price * (1 + recent_trend)
                
                st.write(f"📈 **Simple Trend Estimate for Tomorrow**: ${tomorrow_estimate:.2f}")
                st.caption("*This is a basic trend calculation, not AI prediction*")
    except:
        pass
    
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
    # Clear any existing TensorFlow sessions and reset state
    tf.keras.backend.clear_session()
    
    # Try multiple approaches to load the model
    model = None
    
    # Approach 1: Standard loading
    try:
        model = load_model("model_returns.h5", compile=False)
        # Manually compile if needed
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        st.success("✅ Model loaded successfully (standard method)")
    except Exception as e1:
        st.warning(f"⚠️ Standard loading failed: {str(e1)}")
        
        # Approach 2: Load without compilation
        try:
            model = tf.keras.models.load_model("model_returns.h5", compile=False)
            st.success("✅ Model loaded successfully (no compile)")
        except Exception as e2:
            st.warning(f"⚠️ No-compile loading failed: {str(e2)}")
            
            # Approach 3: Load weights only (if architecture is known)
            try:
                # Create a basic LSTM model architecture
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout
                
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(60, 5)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                
                # Try to load weights
                model.load_weights("model_returns.h5")
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                st.success("✅ Model reconstructed and weights loaded")
                
            except Exception as e3:
                st.error(f"❌ All model loading approaches failed:")
                st.error(f"1. Standard: {str(e1)}")
                st.error(f"2. No-compile: {str(e2)}")
                st.error(f"3. Weights-only: {str(e3)}")
                st.info("Switching to Demo Mode due to model loading issues")
                
                # Fall back to demo mode
                model = None
    
    if model is None:
        # Force demo mode
        st.warning("⚠️ Running in Demo Mode - Model couldn't be loaded")
        # Jump to demo mode logic (reuse the demo code)
        raise Exception("Model loading failed, switching to demo mode")

except Exception as e:
    st.warning("⚠️ Model loading issues detected. Running in **Demo Mode**")
    st.info("Showing stock analysis without ML predictions")
    
    # Demo Mode - Enhanced stock analysis
    st.subheader(f"📊 Stock Analysis for {stock} (Demo Mode)")
    
    # Display current metrics
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
    
    # Technical Analysis
    st.subheader("📈 Technical Analysis")
    
    # Calculate moving averages
    features['MA5'] = features['Close'].rolling(window=5).mean()
    features['MA20'] = features['Close'].rolling(window=20).mean()
    features['MA50'] = features['Close'].rolling(window=50).mean()
    
    # Simple trend prediction based on moving averages
    current_price = features['Close'].iloc[-1]
    ma5_current = features['MA5'].iloc[-1] if not pd.isna(features['MA5'].iloc[-1]) else current_price
    ma20_current = features['MA20'].iloc[-1] if not pd.isna(features['MA20'].iloc[-1]) else current_price
    
    # Simple trend analysis
    if ma5_current > ma20_current:
        trend = "📈 Upward"
        trend_color = "green"
    else:
        trend = "📉 Downward" 
        trend_color = "red"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Trend", trend)
    with col2:
        # Simple next-day estimate based on recent trend
        recent_returns = features['Close'].pct_change().tail(5).mean()
        next_day_estimate = current_price * (1 + recent_returns)
        st.metric("Tomorrow Estimate", f"${next_day_estimate:.2f}", f"{(next_day_estimate - current_price):+.2f}")
    
    # Chart with moving averages
    st.subheader("📊 Price Chart with Moving Averages")
    chart_data = features[['Close', 'MA5', 'MA20', 'MA50']].tail(100)  # Last 100 days
    st.line_chart(chart_data)
    
    # Volume analysis
    st.subheader("📊 Volume Analysis")
    st.bar_chart(features['Volume'].tail(30))  # Last 30 days volume
    
    # Recent data table
    st.subheader("📋 Recent Data (Last 10 days)")
    recent_data = features[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20']].tail(10)
    st.dataframe(recent_data.round(2), use_container_width=True)
    
    st.info("💡 **Demo Mode**: This shows technical analysis without AI predictions. Deploy with working model files for LSTM-powered forecasting!")
    st.stop()

try:
    with open("scaler_X.pkl","rb") as f:
        scaler_X = pickle.load(f)
    with open("scaler_y.pkl","rb") as f:
        scaler_y = pickle.load(f)
    st.success("✅ Scalers loaded successfully")
    scalers_available = True
except FileNotFoundError as e:
    st.error(f"❌ Scaler files not found: {str(e)}")
    st.info("Please ensure 'scaler_X.pkl' and 'scaler_y.pkl' files exist in the current directory.")
    st.warning("⚠️ Running in Demo Mode - Missing scaler files")
    scalers_available = False
    scaler_X = None
    scaler_y = None
except Exception as e:
    st.error(f"❌ Error loading scalers: {str(e)}")
    st.info("There might be compatibility issues with the scaler files.")
    st.warning("⚠️ Running in Demo Mode - Scaler loading error")
    scalers_available = False
    scaler_X = None
    scaler_y = None

# Check if both model and scalers are available
if not scalers_available:
    # Enhanced Demo Mode when scalers fail
    st.subheader(f"📊 Stock Analysis for {stock} (Demo Mode - No Scalers)")
    
    # Display current metrics
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
    
    # Technical Analysis
    st.subheader("📈 Technical Analysis")
    
    # Calculate moving averages
    features['MA5'] = features['Close'].rolling(window=5).mean()
    features['MA20'] = features['Close'].rolling(window=20).mean()
    features['MA50'] = features['Close'].rolling(window=50).mean()
    
    # Chart with moving averages
    st.subheader("📊 Price Chart with Moving Averages")
    chart_data = features[['Close', 'MA5', 'MA20', 'MA50']].tail(60)  # Last 60 days
    st.line_chart(chart_data)
    
    # Recent data
    st.subheader("📋 Recent Data (Last 5 days)")
    recent_data = features[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5)
    st.dataframe(recent_data.round(2), use_container_width=True)
    
    st.info("💡 **Demo Mode**: Upload working scaler files for ML predictions!")
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

st.subheader(f"Forecast including today + next {forecast_days-1} days")

# Add today's current price as the first "prediction" for reference
today_price = df['Close'].iloc[-1].item()
forecast_prices_with_today = [today_price] + forecast_prices
forecast_days_with_today = forecast_days + 1

# Generate dates starting from today
forecast_dates = pd.date_range(datetime.today().date(), periods=len(forecast_prices_with_today))
forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Price_Type": ["Current"] + ["Predicted"] * len(forecast_prices),
    "Forecast": forecast_prices_with_today
})

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

# Display the forecast table with better formatting
st.write("📊 **Forecast Table:**")
forecast_display = forecast_df.copy()
forecast_display['Forecast'] = forecast_display['Forecast'].apply(lambda x: f"${x:.2f}")
forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
st.dataframe(forecast_display, use_container_width=True)

# Plot with improved visualization
fig, ax = plt.subplots(figsize=(12, 6))

# Historical prices (excluding today if it overlaps)
historical_end_date = df.index[-1].date()
today_date = datetime.today().date()

if historical_end_date < today_date:
    # Historical data doesn't include today
    ax.plot(df.index, df['Close'], label="Historical Price", color="blue", linewidth=2)
else:
    # Historical data includes today, so exclude today from historical plot
    historical_df = df[df.index.date < today_date]
    if not historical_df.empty:
        ax.plot(historical_df.index, historical_df['Close'], label="Historical Price", color="blue", linewidth=2)

# Today's price (current/actual)
today_forecast = forecast_df[forecast_df['Price_Type'] == 'Current']
if not today_forecast.empty:
    ax.scatter(today_forecast['Date'], today_forecast['Forecast'], 
              label="Today's Price", color="orange", s=100, zorder=5)

# Future predictions
future_forecast = forecast_df[forecast_df['Price_Type'] == 'Predicted']
if not future_forecast.empty:
    ax.plot(future_forecast['Date'], future_forecast['Forecast'], 
           label="Predictions", color="green", marker="o", linewidth=2, markersize=6)
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{stock} Stock Price Prediction ")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
