import argparse
import time
import typing
import bittensor as bt
import numpy as np
from tensorflow.keras.models import load_model
import datetime
import os
from datetime import datetime
from pytz import timezone
import time

# Import classes and functions from predictionnet module
# Import functions from base_miner module
from predict_dn import predict
from get_data_dn import prep_data, scale_data

def get_prediction(timestamp: int) -> None:
    ny_timezone = timezone('America/New_York')
    current_time_ny = datetime.now(ny_timezone)
    timestamp = current_time_ny.isoformat()  
    prediction = predict(timestamp)
    print(prediction)
# Example timestamp for March 6th, 4 PM UTC
  # Example usage:

if __name__ == "__main__":
    get_prediction(0)  # You can pass any integer value for the timestamp parameter

