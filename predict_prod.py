import joblib
import numpy as np
import random

from RND.get_data_prod import prep_data

def predict(timestamp):
    # Load model and scaler
    model = joblib.load('/snpOracle/RND/model_gb.pkl')
    scaler = joblib.load('/snpOracle/RND/scaler_gb.pkl')

    # grab data
    data = prep_data()

    # subset most recent data row
    latest_row = data.iloc[-1:].copy()
    
    # extract features for scaling and prediction
    excluded_columns = ['Datetime', 'NextClose']  # Remove Columns
    features_columns = [col for col in latest_row.columns if col not in excluded_columns]

    latest_features = latest_row[features_columns]
    
    # scale features
    latest_features_scaled = scaler.transform(latest_features)
    
    # predict
    prediction = model.predict(latest_features_scaled)[0]

    ##### Predict Safety Net #####
    # close_price = latest_row['Close'].values[0]

    # define a range for your random offset
    # offset_min, offset_max = 0.01, 0.30
    # # define the threshold
    # threshold = 2.00

    # if np.abs(prediction - close_price) <= threshold:
    #     prediction = prediction
    # else:
    # # generate random offset
    #     random_offset = random.uniform(offset_min, offset_max)
    
    # # adjust prediction based on whether prediction is higher or lower than the close price
    # if prediction > close_price:
    #     # if predicted price is greater, subtract the random offset from the close price
    #     adj_prediction = close_price - random_offset
    #     prediction = adj_prediction
    # else:
    #     # if predicted price is lower, add the random offset to the close price
    #     adj_prediction = close_price + random_offset
    #     prediction = adj_prediction

    ################################

    # store prediction
    prediction = float(round(prediction, 6))

    # return final prediction
    return prediction
