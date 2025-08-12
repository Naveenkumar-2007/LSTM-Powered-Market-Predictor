import streamlit as st 
import numpy as np
import pandas as pd 
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,LSTM,Dropout
import yfinance as yf
from datetime import datetime,timedelta
import plotly.graph_objects as go
st.title('📈Stock price Predictor')
stock=st.sidebar.text_input('Enter your stock symbol',value='GOOG')
start_time=st.sidebar.date_input('start Date',value=datetime(2018,1,1))
end_time=st.sidebar.date_input('End data',value=datetime.today())
forcast_data=st.sidebar.number_input('Future prediction Num',min_value=1,max_value=30,value=7)
df=yf.download(stock,start=start_time,end=end_time)
data=df[['Close']]


model=load_model('lstm.h5')
with open('scaler.pkl','rb') as f:
    scaler_data=pickle.load(f)

normal=MinMaxScaler(feature_range=(0,1))
normal_data=scaler_data.transform(data)
last_back=60
x_input=normal_data[-last_back:]

prediction=[]
for i in range(forcast_data):
    per=np.reshape(x_input,(1,x_input.shape[0],1))
    pred=model.predict(per)
    prediction.append(pred[0,0])
    x_input=np.append(x_input[1:],pred)
    x_input=x_input.reshape(-1,1)

prediction=scaler_data.inverse_transform(np.array(prediction).reshape(-1, 1))

forcast=pd.date_range(end_time+timedelta(days=1),periods=forcast_data)
forcast = [pd.Timestamp(x).to_pydatetime() for x in forcast]
forcast_pandas=pd.DataFrame({'Data':forcast,'Forecast':prediction.flatten()})
st.subheader(f'Forecast for next {forcast_data} days')
st.write(forcast_pandas)


import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(10, 5))


ax.plot(df.index, df['Close'], label='Historical Price', color='r')


ax.plot(forcast_pandas['Data'], forcast_pandas['Forecast'], label='Forecast', color='g', marker='o')

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title(f'{stock} Price Prediction')
ax.legend()

plt.xticks(rotation=45)

st.pyplot(fig)
