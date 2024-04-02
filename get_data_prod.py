# Import required models
from datetime import datetime, timedelta
from pandas import DataFrame
import ta
import yfinance as yf
import numpy as np
import pandas as pd

def prep_data(drop_na:bool = True) -> DataFrame:
    ### Yahoo Finance Original ####
    data = yf.download('^GSPC', period='60d', interval='5m')
    data = data.drop(columns=['Adj Close'])
    data.index = data.index.tz_localize(None)

    ##### Features ####
    data['5min_Return'] = data['Close'].pct_change()
    data['Cum_5min_Return'] = (1 + data['5min_Return']).cumprod()
    data['Cumulative_Range'] = (data['High'] - data['Low']).cumsum()
    data['Cumulative_Volume'] = data['Volume'].cumsum()
    data['EMA_Close_3'] = ta.trend.ema_indicator(data['Close'], window=3)
    data['Ichimoku_a'] = ta.trend.ichimoku_a(data['High'], data['Low'], window1=3, window2=9)
    data['NVI'] = ta.volume.negative_volume_index(data['Close'], data['Volume'])
    data['SMA_Close_2'] = ta.trend.sma_indicator(data['Close'], window=2)
    data['SMA_Close_3'] = ta.trend.sma_indicator(data['Close'], window=3)
    data['SMA_Low_2'] = ta.trend.sma_indicator(data['Low'], window=2)
    data['VWAP'] = (np.cumsum(data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3) / np.cumsum(data['Volume']))
    data['Weighted_Close'] = (data['Close'] * 2 + data['High'] + data['Low']) / 4

    # Remove Non Predictors
    data = data.drop(columns=['Volume', '5min_Return'])

    # data['NextClose'] = data['Close'].shift(-1)

    if(drop_na):
        data.dropna(inplace=True)

    data.reset_index(inplace=True)

    return data

