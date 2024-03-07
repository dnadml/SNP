# Import required models
from datetime import datetime, timedelta
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import numpy as np
import pandas as pd
import ta
import yfinance as yf


def prep_data(drop_na:bool = True) -> DataFrame:
    data = yf.download('^GSPC', period='60d', interval='5m')
    data
    # Features
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Prev_Close'] = data['Close'].shift(1)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['EMA_50'] = ta.trend.ema_indicator(data['Close'], window=50)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Open_Rolling_10'] = data['Open'].rolling(window=10).mean()
    data['Low_Rolling_10'] = data['Low'].rolling(window=10).mean()
    data['Rolling_Avg_Diff_10'] = data['Open_Rolling_10'] - data['Low_Rolling_10']
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['Momentum'] = ta.momentum.ROCIndicator(data['Close']).roc()

    for lag in [5]:
        data[f'Low_lag_{lag}'] = data['Low'].shift(lag)
        data[f'High_lag_{lag}'] = data['High'].shift(lag)

    

    # Features that Didnt Influence Well
    
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

    data['NextClose'] = data['Close'].shift(-1)
    
    # backup_data = data
    # backup_data.to_csv('C:/Users/Drew/Desktop/5 min backup.csv')


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

def scale_data(data:DataFrame) -> Tuple[MinMaxScaler, np.ndarray, np.ndarray]:
    X = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'CCI', 'Momentum', 'EMA_50', 'Low_lag_5', 'High_lag_5', 'Open_Rolling_10', 'Low_Rolling_10', 'Rolling_Avg_Diff_10']].values

    # Prepare target variable
    y = data[['NextClose']].values

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

    return scaler, X_scaled, y_scaled


