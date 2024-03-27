# Import required models
from datetime import datetime, timedelta
from pandas import DataFrame
import ta
import yfinance as yf

def prep_data(drop_na:bool = True) -> DataFrame:
    data = yf.download('^GSPC', period='60d', interval='5m')
    data = data.drop(columns=['Adj Close'])
    data.index = data.index.tz_localize(None)
    data.rename(columns={'Close': 'close'}, inplace=True)

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

    data['NextClose'] = data['close'].shift(-1)
    
    # Drop NaN values
    if(drop_na):
        data.dropna(inplace=True)

    data.reset_index(inplace=True)

    return data
