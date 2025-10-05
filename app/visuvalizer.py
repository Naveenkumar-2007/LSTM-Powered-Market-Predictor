import matplotlib.pyplot as plt

def plot_predictions(df, future_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Close'], label='Historical Data', color='blue')
    ax.plot(future_df['Date'], future_df['Prediction'], label='Predicted Future', color='green')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    return fig
