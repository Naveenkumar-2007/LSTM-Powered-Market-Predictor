import streamlit as st 
import numpy as np
import pandas as pd 
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.title("📈 Stock Price Predictor ")

stock = st.sidebar.text_input("Enter your stock symbol", value="TSLA")
start_time = st.sidebar.date_input("Start Date", value=datetime(2025,1,1))
end_time = st.sidebar.date_input("End Date", value=datetime.today())
forecast_days = st.sidebar.number_input("Future prediction days", min_value=1, max_value=30, value=7)

try:
    # Download data with error handling
    df = yf.download(stock, start=start_time, end=end_time, auto_adjust=True, prepost=True, threads=True)
    
    if df.empty:
        st.error(f"No data found for symbol {stock}. Please check the symbol and date range.")
        st.stop()
    
    df = df.sort_index()
    features = df[['Open','High','Low','Close','Volume']]
    
    if len(features) < 60:
        st.error(f"Insufficient data. Need at least 60 days of data, but got {len(features)} days.")
        st.stop()

    # Load model and scalers
    model = load_model("model_returns.h5")
    with open("scaler_X.pkl","rb") as f:
        scaler_X = pickle.load(f)
    with open("scaler_y.pkl","rb") as f:
        scaler_y = pickle.load(f)

    scaled_X = scaler_X.transform(features.values)
    look_back = 60
    x_input = scaled_X[-look_back:]

    last_real_price = df['Close'].iloc[-1]
    forecast_prices = []

    for _ in range(forecast_days + 1):  # include today's prediction
        x_reshaped = np.reshape(x_input, (1, look_back, x_input.shape[1]))
        pred_return_scaled = model.predict(x_reshaped, verbose=0)
        pred_return = scaler_y.inverse_transform(pred_return_scaled)[0,0]
        
        next_price = float(last_real_price * (1 + pred_return))  
        forecast_prices.append(next_price)
        
        last_real_price = next_price
        new_row = features.iloc[-1].copy()
        new_row['Close'] = next_price
        new_scaled = scaler_X.transform([new_row.values])
        x_input = np.vstack([x_input[1:], new_scaled])

    forecast_dates = pd.date_range(end_time, periods=forecast_days+1)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast": forecast_prices
    })

    st.subheader(f"Forecast for today + next {forecast_days} days")
    st.dataframe(forecast_df)

    fig, ax = plt.subplots(figsize=(12,6))

    # Plot historical prices
    ax.plot(df.index, df['Close'], label="Historical Price", color="red", linewidth=2)

    # Plot forecast
    ax.plot(forecast_df['Date'], forecast_df['Forecast'], label="Forecast", color="green", linewidth=2, marker="o")

    # Highlight today's prediction
    ax.scatter(forecast_df['Date'].iloc[0], forecast_df['Forecast'].iloc[0],
               color="blue", s=120, marker="*", label="Today's Prediction", zorder=5)

    ax.set_title(f"{stock} Stock Price Prediction", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please try with a different stock symbol or date range.")