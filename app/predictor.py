import numpy as np
import pandas as pd
from datetime import timedelta

def predictor_builder(model, scaler, scaled_data, window_size=60, forecast_days=7, last_date=None):
    last_window = scaled_data[-window_size:]
    predictions = []
    current_input = last_window.reshape(1, window_size, 1)

    for _ in range(forecast_days):
        pred = model.predict(current_input, verbose=0)  # shape (1, 1)
        predictions.append(pred[0, 0])

        # reshape to match 3D LSTM input before concatenation
        pred_reshaped = np.reshape(pred, (1, 1, 1))
        current_input = np.concatenate((current_input[:, 1:, :], pred_reshaped), axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    future_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_days)
    future_df = pd.DataFrame({"Date": future_dates, "Prediction": predicted_prices.flatten()})
    return future_df
