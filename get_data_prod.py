import pandas as pd
import ta
import yfinance as yf

def prep_data(drop_na:bool = True) -> pd.DataFrame:
    # download data
    data = yf.download(tickers='^GSPC', period='7d', interval='1m')
    data.index = data.index.tz_localize(None)
    data = data.drop(columns=['Adj Close', 'Volume'])
    
    # Define intervals
    intervals = ['5min', '10min', '15min', '20min', '25min', '30min']
    datasets = {}

    # resample each interval
    for interval in intervals:
        # resampled_data = data.resample(interval).ohlc().shift(1)
        resampled_data = data.resample(interval, label='right', closed='right').ohlc()
        resampled_data.columns = ['{}_{}'.format(val[1], val[0]) for val in resampled_data.columns]
        resampled_data.ffill(inplace=True)

        # features per interval
        if interval in ['5min', '10min']:
            resampled_data['SMA_5'] = resampled_data['close_Close'].rolling(window=5).mean()
            resampled_data['SMA_2'] = resampled_data['close_Close'].rolling(window=2).mean()
            resampled_data['EMA_5'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=5)

        if interval in ['15min', '20min']:
            resampled_data['SMA_10'] = resampled_data['close_Close'].rolling(window=10).mean()
            resampled_data['EMA_10'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=10)
            resampled_data['RSI'] = ta.momentum.rsi(resampled_data['close_Close'], window=14)

        if interval in ['25min', '30min']:
            resampled_data['SMA_20'] = resampled_data['close_Close'].rolling(window=20).mean()
            resampled_data['EMA_20'] = ta.trend.ema_indicator(resampled_data['close_Close'], window=20)
            resampled_data['MACD'] = ta.trend.macd_diff(resampled_data['close_Close'])

        # common features
        resampled_data['SMA_3'] = resampled_data['close_Close'].rolling(window=3).mean()

        resampled_data.reset_index(inplace=True)

        # drop NA
        if drop_na:
            resampled_data.dropna(inplace=True)

        datasets[interval] = resampled_data

    return datasets
# prep_data()