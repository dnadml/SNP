# Import required models
from datetime import datetime, timedelta
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import numpy as np
import pandas as pd
import ta
from ta import add_all_ta_features
import yfinance as yf
from twelvedata import TDClient
import warnings

# Twelve Data API Key
td = TDClient(apikey="6857b613682340e79585cdbcb1c98394")

def prep_data(drop_na:bool = True) -> DataFrame:
    ### Yahoo Finance Original ####
    data_yf = yf.download('^GSPC', period='60d', interval='5m')
    data_yf = data_yf.drop(columns=['Adj Close'])
    data_yf.index = data_yf.index.tz_localize(None)
    data_yf.rename(columns={'Close': 'close'}, inplace=True)
    data_yf

    data = data_yf


    # # Construct the necessary time series
    # ts = td.time_series(
    #     symbol="GSPC",
    #     interval="5min",
    #     outputsize=5000,
    #     timezone="America/New_York",
    # )
    # # return dataframe
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    # data = ts.with_bbands(ma_type="EMA").as_pandas()
    # data.index.names = ['Datetime']
    # data = data.sort_index()
    # data

    ################################### TWELVE DATA TRAINING ##################################
    ###########################################################################################
    #  # Construct the necessary time series
    # ts = td.time_series(
    #     symbol="GSPC",
    #     interval="5min",
    #     outputsize=5000,
    #     timezone="America/New_York",
    # )
    # # return dataframe
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    # data = ts.with_bbands(ma_type="EMA").as_pandas()
    # data.index.names = ['Datetime']
    # data = data.sort_index()

    # data

    # # Construct the necessary time series
    # ts = td.time_series(
    #     symbol="GSPC",
    #     interval="5min",
    #     outputsize=5000,
    #     end_date="2023-12-06",
    #     timezone="America/New_York",
    # )
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    # temp = ts.with_bbands(ma_type="EMA").as_pandas()

    # data = pd.concat([temp, data])
    # data = data.drop_duplicates()
    # data = data.sort_index()

    ###########################################################################################
    ###########################################################################################

    # Features
    data['SMA_Close_30'] = ta.trend.sma_indicator(data['close'], window=30)
    data['SMA_Close_20'] = ta.trend.sma_indicator(data['close'], window=20)
    data['SMA_Close_15'] = ta.trend.sma_indicator(data['close'], window=15)
    data['SMA_Close_10'] = ta.trend.sma_indicator(data['close'], window=10)
    data['SMA_Close_3'] = ta.trend.sma_indicator(data['close'], window=3)
    data['SMA_Low_3'] = ta.trend.sma_indicator(data['Low'], window=3)
    data['SMA_Low_1'] = ta.trend.sma_indicator(data['Low'], window=1)
    data['SMA_Open_3'] = ta.trend.sma_indicator(data['Open'], window=30)
    data['SMA_Open_1'] = ta.trend.sma_indicator(data['Open'], window=1)
    data['Prev_Close'] = data['close'].shift(1)

    # Features that Didnt Influence Well
    # data['EMA_50'] = ta.trend.ema_indicator(data['Close'], window=50)
    # data['Prev_Close'] = data['Close'].shift(1)
    # for lag in [5]:
    #     data[f'Low_lag_{lag}'] = data['Low'].shift(lag)
    #     data[f'High_lag_{lag}'] = data['High'].shift(lag)
    # data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    # data['SMA_200'] = data['Close'].rolling(window=200).mean()
    # data['Open_Rolling_10'] = data['Open'].rolling(window=10).mean()
    # data['Low_Rolling_10'] = data['Low'].rolling(window=10).mean()
    # data['Rolling_Avg_Diff_10'] = data['Open_Rolling_10'] - data['Low_Rolling_10']
    # data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    # data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    # data['Momentum'] = ta.momentum.ROCIndicator(data['Close']).roc()
    # data['Open_to_Low_Pct_Change'] = (data['Open'] - data['Low']) / data['Open'] * 100
    # data['Custom_VIX'] = data['Close'].rolling(window=10).std() * np.sqrt(252)
    # data['Low_rolling_std_5'] = data['Low'].rolling(window=5).std()
    # data['MACD'] = ta.trend.macd(data['Close'], window_fast=12, window_slow=26) 
    # data['Range'] = data['High'] - data['Low']  # Range
    # data['Volatility'] = data['Close'].pct_change().rolling(window=50).std()
    # data['VMA_10'] = data['Volume'].rolling(window=10).mean()
    # data['VMA_30'] = data['Volume'].rolling(window=30).mean()
    # data['VMA_60'] = data['Volume'].rolling(window=60).mean()
    # data['VROC'] = data['Volume'].pct_change()
    # data['VROC'] = data['VROC'].replace([np.inf, -np.inf], np.nan)
    # data['VROC'] = data['VROC'].fillna(0)

    data['NextClose'] = data['close'].shift(-1)
    
    # Drop NaN values
    if(drop_na):
        data.dropna(inplace=True)

    data.reset_index(inplace=True)

    return data

def round_down_time(dt:datetime, interval_minutes:int = 5) -> datetime:
    # Round down the time to the nearest interval
    rounded_dt = dt - timedelta(minutes=dt.minute % interval_minutes,
                                seconds=dt.second,
                                microseconds=dt.microsecond)

    return rounded_dt
