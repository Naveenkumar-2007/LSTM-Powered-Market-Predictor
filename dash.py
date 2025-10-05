import streamlit as st
from datetime import datetime
from app.data_inhestion import data_load, feature_data
from app.model_Bulider import build_and_train_lstm
from app.predictor import predictor_builder
from app.visuvalizer import plot_predictions

st.set_page_config(page_title="📈 Future Stock Price Prediction", layout="wide")

st.title("📈 Future Stock Price Prediction")
st.markdown("Predict upcoming stock prices in real-time!")

ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
start = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
end = st.sidebar.date_input("End Date", datetime.now())
future_days = st.sidebar.number_input("Predict Future Days", min_value=1, max_value=30, value=7)

if st.sidebar.button("🚀 Predict Future"):
    try:
        with st.spinner("Training model & predicting future prices... ⏳"):
            df = data_load(ticker, start, end)
            X, y, scaler, scaled_data = feature_data(df)
            model = build_and_train_lstm(X, y)
            future_df = predictor_builder(
                model, scaler, scaled_data, forecast_days=future_days, last_date=df["Date"].iloc[-1]
            )

        st.subheader("📊 Last 5 Historical Records")
        st.dataframe(df.tail(5))

        st.subheader("🔮 Future Predictions")
        st.dataframe(future_df)

        st.pyplot(plot_predictions(df, future_df))

    except Exception as ex:
        st.error(f"⚠️ Error: {ex}")
