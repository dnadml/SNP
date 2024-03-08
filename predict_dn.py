from datetime import datetime
from pytz import timezone
import joblib
import numpy as np
import os
import pandas as pd


from get_data_dn import prep_data

def predict(timestamp):
    # load model and scaler
    model = joblib.load('/snpOracle/SNP/model_gb.pkl')
    scaler = joblib.load('/snpOracle/SNP/scaler_gb.pkl')

    # confirm no missing NA (could be duplicate step)
    data = prep_data(drop_na=False)

    # data = data[data['Datetime'] <= timestamp]  # Used for backtesting COMMENT OUT!
    
    # Print yahoo data for sanity check if need be
    # print("Last 10 rows of the fetched data for verification:")
    # print(data.tail(10))

    # ensure datetime
    latest_row = data.iloc[-1:].copy()
    
    # extract features for scaling and prediction
    features_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA_50']
    latest_features = latest_row[features_columns]
    
    # scale features for loading data
    latest_features_scaled = scaler.transform(latest_features)
    
    # predict
    prediction = model.predict(latest_features_scaled)[0]

    # format and print the prediction value
    formatted_prediction = np.array([[prediction]])

    # add prediction value
    latest_row['Predicted_NextClose'] = prediction

    # Store prediction in a variable
    final_prediction = latest_row['Predicted_NextClose'].values[0]


    ######################### Log Data #########################################
    current_prediction_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Make sure to assign the timestamp before reordering columns
    latest_row['Prediction_Timestamp'] = current_prediction_timestamp
    latest_row['Predicted_Close'] = prediction

    # Correct ordering: Ensure 'Prediction_Timestamp' is actually added before reordering
    cols = ['Prediction_Timestamp'] + [col for col in latest_row if col != 'Prediction_Timestamp']
    latest_row = latest_row[cols]

    # Assuming latest_row is a DataFrame. If it's a single row, no need to flatten.
    log_df = latest_row

    # Log to a CSV file
    csv_file_path = './predictions_log.csv'
    if not os.path.isfile(csv_file_path):
        log_df.to_csv(csv_file_path, mode='w', header=True, index=False)
    else:
        log_df.to_csv(csv_file_path, mode='a', header=False, index=False)
    
    ############################################################################    
    
    #### Uncomment return latest_row and comment return final_prediction to see table format of data ####
    #### This also needs to be changed when running the main script, use latest_row for main ####
        
    #return latest_row
        
    prediction = float(round(final_prediction,6))
    return prediction    
   

    # Test for return value
    # timestamp = "2024-03-06 12:00:00"
    # predict_result = predict(timestamp) 
