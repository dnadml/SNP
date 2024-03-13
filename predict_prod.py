import joblib
import numpy as np

from SNP.get_data_prod import prep_data

def predict(timestamp):
    # Load model and scaler
    model = joblib.load('./SNP/model_gb.pkl')
    scaler = joblib.load('./SNP/scaler_gb.pkl')

    # Confirm no missing NA
    data = prep_data(drop_na=False)

    # Ensure datetime
    latest_row = data.iloc[-1:].copy()
    
    # Extract features for scaling and prediction
    excluded_columns = ['Datetime', 'NextClose']  # Add any other columns you know should be excluded.
    features_columns = [col for col in latest_row.columns if col not in excluded_columns]

    latest_features = latest_row[features_columns]
    
    # Scale features for loading data
    latest_features_scaled = scaler.transform(latest_features)
    
    # Predict
    prediction = model.predict(latest_features_scaled)[0]

    # Store prediction in a variable
    prediction = float(round(prediction, 6))
    
    # Return the final prediction
    return prediction
