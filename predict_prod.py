import joblib

from SNP.get_data_prod import prep_data

def predict(timestamp):
    # time intervals
    intervals = ['5min', '10min', '15min', '20min', '25min', '30min']
    
    # load datsets
    data_dfs = prep_data()

    # empty predictions
    predictions = {}

    for interval in intervals:
        # load model files
        model = joblib.load(f'/snpOracle/SNP/model_gb_{interval}.pkl')
        scaler = joblib.load(f'/snpOracle/SNP/scaler_gb_{interval}.pkl')

        # get latest data for each row
        data = data_dfs[interval]
        latest_row = data.iloc[-1:].copy()

        # print(f"Last row of data for {interval}:")
        # print(latest_row)

        # features
        excluded_columns = ['Datetime', 'NextClose']
        features_columns = [col for col in latest_row.columns if col not in excluded_columns]

        latest_features = latest_row[features_columns]

        # scale
        latest_features_scaled = scaler.transform(latest_features)

        # predict
        prediction = model.predict(latest_features_scaled)[0]

        # store all predictions
        predictions[interval] = float(round(prediction, 6))

    return list(predictions.values())