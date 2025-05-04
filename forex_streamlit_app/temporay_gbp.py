


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import mplcursors

import datetime
from datetime import date,timedelta

# title of the app
import random
import os

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

col1,col2=st.columns(2)
with col1:
     st.title('Pritex32 Market Price Forecasting insight') 
with col2:
     image='forex_streamlit_app/trading chart image.jfif'
     img = Image.open(image)          # load the image
     img = img.resize((100, 100))         # resize the image
     st.image(img)  

st.subheader('Author: Prisca Ukanwa')
col1,col2=st.columns(2)
with col1:
    st.write('notification: put in only the ticker ID of the stock')
with col2:
    st.write('Disclaimer: Not a financial advise')


import time

gbpusd=st.text_input('Enter the stock ID','GBPUSD')

import yfinance as yf

# Fetch GBP/USD forex data with a daily interval


# Fetch GBP/USD forex data from 2010 to present with daily interval
# Function to fetch GBP/USD data with caching
today = date.today().strftime("%Y-%m-%d")
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_forex_data():
    try:
        gbpusd = yf.Ticker("GBPUSD=X")
        hist = gbpusd.history(start="2010-01-01", end=today, interval="1d")
        return hist
    except yf.YFRateLimitError:
        st.error("Yahoo Finance rate limit reached. Please try again later.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Load data
data = get_forex_data()

# Check if data is available before displaying it
if data is not None and not data.empty:
    st.write("Data Loaded Successfully!")
    st.write(data.head())
else:
    st.warning("No data available to display.")

#renamimg columns






# to show the data downloaded
st.subheader('stock Data')
with st.expander('View All Data',expanded=False):
     st.write(data)
     
st.write('current_data',data.tail())
# select target variable






st.sidebar.header('App Details')
st.sidebar.write('Welcome to the **Future Price Forecasting App**! This application is designed to assist traders and investors in making informed decisions by providing accurate price forecasts for stocks currency pairs, or other financial instruments.')
st.sidebar.write(""" With this tool, you can:
- **Minimize trading losses** by anticipating market trends.
- **Make strategic investment decisions** with data-driven forecasts.
- **Plan for the future** using customizable prediction periods.""")
st.sidebar.subheader('How to Use the App')
st.sidebar.write("""
1. **Enter the Stock or Pair ID**: Input the identifier for the stock or currency pair you want to forecast.
2. **Select the Start Date**: Choose the starting point for your prediction period.
3. **Set the Prediction Period**: Indicate how many days, months, or years you want to forecast.
4. **Download the Forecast**: After generating the predictions, you can download the forecasted values in a convenient format.
""")


# contact developer buttion

# Create a button that links to your LinkedIn
st.sidebar.markdown(
    """
    <a href="https://www.linkedin.com/in/prisca-ukanwa-800a1117a/" target="_blank">
        <button style='background-color: #0072b1; color: white; padding: 10px 20px; border: none; border-radius: 5px;'>
            Contact Developer
        </button>
    </a>
    """,
    unsafe_allow_html=True
)


data.dropna(inplace=True)
data.drop_duplicates(inplace=True)


## convert date to index

# Convert the index to datetime
data.index = pd.to_datetime(data.index)
# Convert the index to datetime





rollmean=data['Open'].rolling(50).mean() # the moving average

# plotting the Open column
st.subheader('Price vs moving Average')

fig=px.line(data, x=data.index, y='Open',title='Open price yearly chart',labels={'Open': 'Open Price'},text='Open')
fig.add_scatter(x=data.index, y=rollmean, mode='lines', name='50 MA')
st.plotly_chart(fig)


# selecting target column
open_price=data[['Open']]






scaler=MinMaxScaler(feature_range=(0,1))
# scaling the data
scaler_data=scaler.fit_transform(open_price)
# loading the model

# feature sequences
x=[]
y=[]
for i in range (60,len(scaler_data)):
      x.append(scaler_data[i-60:i])
      y.append(scaler_data[i])
x=np.array(x)
y=np.array(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# build the layers
from keras .models import Sequential
from keras.layers import Dense,Dropout,LSTM

lstm=Sequential()
lstm.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
lstm.add(Dropout(0.5))

lstm.add(LSTM(units=50,return_sequences=True))
lstm.add(Dropout(0.5))
lstm.add(LSTM(units=50,return_sequences=True))
lstm.add(Dropout(0.5))
lstm.add(LSTM(units=50,return_sequences=False))
lstm.add(Dropout(0.5))
lstm.add(Dense(units=1))

# compile
lstm.compile(optimizer='adam',loss='mean_absolute_error')

model=lstm.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=1)



pred=lstm.predict(x_test)

inv_pred=scaler.inverse_transform(pred)

inv_y_test=scaler.inverse_transform(y_test)


#plotting the predicted vs actual values
from statsmodels.tsa .api import SimpleExpSmoothing

st.subheader('Actual values vs model predicted to validate accuracy')
fig9=plt.figure(figsize=(15,10))
fit1=SimpleExpSmoothing(inv_pred).fit(smoothing_level=0.02,optimized=False)
sns.lineplot(fit1.fittedvalues,color='red',label='Predicted values')
fit2=SimpleExpSmoothing(inv_y_test).fit(smoothing_level=0.02,optimized=False)
sns.lineplot(fit2.fittedvalues,color='blue',label='Actual')
plt.legend()
st.pyplot(fig9)

from sklearn.metrics import mean_squared_error
# the mean squared error
rmse=np.sqrt(mean_squared_error(inv_y_test,inv_pred))
st.write('mean_squared_error',rmse)





# putting the date range for future prediction

st.subheader('Forecast'.upper())

n_period = st.number_input(label='How many days of forecast?', min_value=1, value=30)



future_days = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_period,freq='D')


        
    
# values to predict 

last_days=scaler_data[-60:]
last_days=last_days.reshape(1,60,1)
future_prediction=[]
for _ in range(n_period):
    nxt_pred=lstm.predict(last_days)
    future_prediction.append(nxt_pred[0,0])
    last_days = np.append(last_days[:, 1:, :], [[[nxt_pred[0, 0]]]], axis=1)
    last_days[0,-1,0]=nxt_pred
    
forecast_array=np.array(future_prediction)
future_prediction=scaler.inverse_transform(forecast_array.reshape(-1,1))


# wether scaled values or x_test,it still the same thing




# dataframe for the forecated values
f_df = pd.DataFrame({'dates':future_days,
                     'open':future_prediction.flatten()}) 

f_df['dates']=pd.to_datetime(f_df['dates'])    

st.subheader('Forecasted values')
st.write('Click the Arrow ontop of this table to download this predictions')
st.dataframe(f_df)

# Create Plotly Express line plot
fig2 = px.line(
    f_df,
    x='dates',
    y='open',
    title='Predictions',
    labels={'open': 'Forecasted Values', 'dates': 'Date'},
    markers=True
)

# Customize marker and line colors
fig2.update_traces(
    line=dict(color='red', width=2),
    marker=dict(color='yellow', size=8, line=dict(color='red', width=1))
)

# Rotate x-axis labels and show grid
fig2.update_layout(
    xaxis_tickangle=-45,
    xaxis_title='Date',
    yaxis_title='Open',
    legend_title='Legend',
    showlegend=True
)

# Show in Streamlit
st.plotly_chart(fig2)



# forecasted vs actual section

st.subheader('Forecasted value and original data')


# plotting the entire df with the forecated values
# Create figure and axis
fig3 = plt.figure(figsize=(15, 8))
sns.lineplot(data=data, x=data.index, y='Open', label='Original values')# Plot original values
# Plot forecasted values
sns.lineplot(data=f_df, x='dates', y='open', label='Forecasted values', color='red')  
plt.title('Days of Predictions, Full View')# Set plot title and formatting
plt.xticks(rotation='vertical')
plt.grid(True)
plt.legend()
# Make the plot interactive
cursor= mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f"X: {sel.target[0]}\nY: {sel.target[1]}"))
plt.show()
# Display in Streamlit
st.pyplot(fig3)












df=data.copy()
df.index = pd.to_datetime(df.index)
# Resample Open price

monthly_open = df['Open'].resample('M').mean().dropna()
weekly_open = df['Open'].resample('W').mean().dropna()
daily_open = df['Open'].resample('D').mean().dropna()
four_hour_open = df['Open'].resample('4H').mean().dropna() # null values are droped becoz when resampling the data produces null values
hourly_open = df['Open'].resample('1H').mean().dropna()

colors = {
    'monthly': '#FF4136',  # red
    'weekly': '#2ECC40',   # green
    'daily': '#0074D9',    # blue
    '4h': '#FF851B',       # orange
    '1h': '#B10DC9'        # purple
}

col1,col2=st.columns(2)
with col1:
    st.subheader('monthly chart')
    fig_monthly = px.line(x=monthly_open.index, y=monthly_open.values, title='Monthly Average Open Price')
    fig_monthly.update_traces(line_color=colors['monthly'])
    st.plotly_chart(fig_monthly)

with col2:
    st.subheader('Weekly Chart')
    fig_weekly = px.line(x=weekly_open.index, y=weekly_open.values, title='Weekly Average Open Price', labels={'x': 'Date', 'y': 'Open Price'})
    fig_weekly.update_traces(line_color=colors['weekly'])
    st.plotly_chart(fig_weekly)

col1,col2=st.columns(2)
with col1:
    st.subheader('Daily Chart')
    fig_daily = px.line(x=daily_open.index, y=daily_open.values, title='Daily Average Open Price', labels={'x': 'Date', 'y': 'Open Price'})
    fig_daily.update_traces(line_color=colors['daily'])
    st.plotly_chart(fig_daily)

# 4-Hour chart
with col2:
    st.subheader('4-Hour Chart')
    fig_4h = px.line(x=four_hour_open.index, y=four_hour_open.values, title='4-Hour Average Open Price', labels={'x': 'Date', 'y': 'Open Price'})
    fig_4h.update_traces(line_color=colors['4h'])
    st.plotly_chart(fig_4h)

# Hourly chart
st.subheader('Hourly Chart')
fig_hourly = px.line(x=hourly_open.index, y=hourly_open.values, title='Hourly Average Open Price', labels={'x': 'Date', 'y': 'Open Price'})
fig_hourly.update_traces(line_color=colors['1h'])
st.plotly_chart(fig_hourly)


# this chart shows that in the next 6months prices will range at 1.7 price in gbpusd




# Ensure date input is valid before filtering
# selecting the open column from the main df

# plotting the entire df by filtering  to get a close view


