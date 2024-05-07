import pandas as pd
import ta
import yfinance as yf
from datetime import datetime, timedelta

def prep_data(drop_na:bool = True) -> pd.DataFrame:
    # import historical data
    temp = pd.read_pickle(f'/snpOracle/TESTING/historical_data/gspc_historical.pkl')

    # import yahoo finance data
    data = yf.download(tickers='^GSPC', period='7d', interval='1m')
    data.index = data.index.tz_localize(None)
    data = data.drop(columns=['Adj Close', 'Volume'])

    # combine data and de-dup
    data = pd.concat([temp, data])
    data = data[~data.index.duplicated(keep='last')]
    data = data.sort_index()
    # data = data.round(2)

    # Define intervals and their corresponding data lengths
    intervals = ['5min', '10min', '15min', '20min', '25min', '30min']
    data_lengths = {
        '5min': timedelta(days=45),
        '10min': timedelta(days=45),
        '15min': timedelta(days=60),
        '20min': timedelta(days=60),
        '25min': timedelta(days=95),
        '30min': timedelta(days=95)
    }

   # Define intervals
    intervals = ['5min', '10min', '15min', '20min', '25min', '30min']
    datasets = {}

    # Get the latest date in the dataset
    latest_date = data.index.max()

   # Resample each interval
    for interval in intervals:
        # Calculate the start date for data slicing based on the interval
        start_date = latest_date - data_lengths[interval]
        interval_data = data[data.index >= start_date]

        # Resample data
        # resampled_data = interval_data.resample(interval).ohlc()
        resampled_data = interval_data.resample(interval, label='right', closed='right').ohlc()
        resampled_data.columns = ['{}_{}'.format(val[1], val[0]) for val in resampled_data.columns]
        resampled_data.ffill(inplace=True)

        # Calculate features for each interval
        if interval == '5min':
            resampled_data['SMA_2_close_Close'] = resampled_data['close_Close'].rolling(window=2).mean()
            resampled_data['EMA_2'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=2)
            resampled_data['RSI'] = ta.momentum.rsi(resampled_data['close_Close'], window=14)
 
        elif interval == '10min':
            resampled_data['SMA_2'] = resampled_data['close_Close'].rolling(window=2).mean()
            resampled_data['EMA_2'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=2)
            resampled_data['RSI'] = ta.momentum.rsi(resampled_data['close_Close'], window=14)

        elif interval == '15min':
            resampled_data['RSI'] = ta.momentum.rsi(resampled_data['close_Close'], window=7)

        elif interval == '20min':
            resampled_data['EMA_10'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=10)
            resampled_data['SMA_5'] = resampled_data['close_Close'].rolling(window=5).mean()
            resampled_data['EMA_5'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=5)
            resampled_data['EMA_3'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=3)
            resampled_data['RSI'] = ta.momentum.rsi(resampled_data['close_Close'], window=14)

        elif interval == '25min':
            resampled_data['SMA_40'] = resampled_data['close_Close'].rolling(window=40).mean()
            resampled_data['EMA_40'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=40)
            resampled_data['EMA_20'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=20)
            resampled_data['RSI'] = ta.momentum.rsi(resampled_data['close_Close'], window=14)

        elif interval == '30min':
            resampled_data['EMA_40'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=40)
            resampled_data['EMA_20'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=20)
            resampled_data['RSI'] = ta.momentum.rsi(resampled_data['close_Close'], window=14)

                
        # # Add SMA and EMA for multiple columns and windows ################# TESTING ################
        # columns = ['open_Open', 'high_Open', 'low_Open', 'close_Open',
        #         'open_High', 'high_High', 'low_High', 'close_High', 'open_Low',
        #         'high_Low', 'low_Low', 'close_Low', 'open_Close', 'high_Close',
        #         'low_Close', 'close_Close']

        # for col in columns:
        #     for window in range(2, 3):  # Windows from 2 to 10
        #         # sma_col_name = f'SMA_{window}_{col}'
        #         # ema_col_name = f'EMA_{window}_{col}'
        #         resampled_data[f'Cum_Return_{col}'] = (1 + resampled_data[col].pct_change()).cumprod() - 1

        # Shift NextClose
        resampled_data['NextClose'] = resampled_data['close_Close'].shift(-1)

        resampled_data.reset_index(inplace=True)
        datasets[interval] = resampled_data

    return datasets

# Example of how to use this function
# data = prep_data()
# print(data['20min'])
