import joblib
import numpy as np
import random

from SNP.get_data_prod import prep_data

def predict(timestamp):
    # Load model and scaler
    model = joblib.load('/snpOracle/SNP/model_gb.pkl')
    scaler = joblib.load('/snpOracle/SNP/scaler_gb.pkl')

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
    close_price = latest_row['Close'].values[0]
    offset_min, offset_max = 0.01, 0.30
    threshold = 2.00
    # Check and adjust the prediction if necessary
    if np.abs(prediction - close_price) > threshold:
        random_offset = random.uniform(offset_min, offset_max)
        if prediction > close_price:
            adj_prediction = close_price - random_offset
            prediction = adj_prediction
        else:
            adj_prediction = close_price + random_offset
            prediction = adj_prediction
    else:
        prediction = prediction

    ################################

    # store prediction
    prediction = float(round(prediction, 6))

    # return final prediction
    return prediction
