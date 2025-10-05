import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def data_load(ticker, start, end):
    data = yf.download(tickers=ticker, start=start, end=end, auto_adjust=False)
    if data.empty:
        raise ValueError(" No data found for given symbol or date range.")
    data.reset_index(inplace=True)
    return data

def feature_data(data, feature='Close', window_size=60):
    df = data[[feature]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler, scaled_data
