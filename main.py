from datetime import datetime
from pytz import timezone
import schedule
import time

# Import Data
from get_data_dn import prep_data
from model_dn import create_model
from predict_dn import predict

# data = prep_data(drop_na=False)

# Print yahoo data for sanity check if need be
# print("Last 10 rows of the fetched data for verification:")
# print(data.tail(10))


if __name__ == '__main__':
    ny_timezone = timezone('America/New_York')
    current_time_ny = datetime.now(ny_timezone)
    timestamp = current_time_ny.isoformat()

    prediction_df = predict(timestamp)
    
    # columns to display for validator
    columns_to_display = ['Datetime', 'Open', 'High', 'Low', 'close', 'Predicted_NextClose']
    
    # filter df for validator display
    filtered_df = prediction_df[columns_to_display]
    
    # print validator results
    print(f"{timestamp}{filtered_df.to_string(index=True)}")
    print(f"[[{prediction_df['Predicted_NextClose'].values[0]}]]")


################### Scheduler #######################
from datetime import datetime
from pytz import timezone
import schedule
import time

# Import Data
from get_data_dn import prep_data
from model_dn import create_model
from predict_dn import predict

def job():
    time.sleep(3) # Make sure Financial Data is available (Potentially acts as when Validator is called?  Adds 3 seonds to time intervals below)
    ny_timezone = timezone('America/New_York')
    current_time_ny = datetime.now(ny_timezone)
    timestamp = current_time_ny.isoformat()

    prediction_df = predict(timestamp)
    
    columns_to_display = ['Datetime', 'Open', 'High', 'Low', 'close', 'Predicted_NextClose']
    
    filtered_df = prediction_df[columns_to_display]
    
    print(f"{timestamp}{filtered_df.to_string(index=True)}")
    print(f"[[{prediction_df['Predicted_NextClose'].values[0]}]]")

# Schedule the job to run every 5 minutes at specific times
schedule.every().hour.at(":00").do(job)
schedule.every().hour.at(":05").do(job)
schedule.every().hour.at(":10").do(job)
schedule.every().hour.at(":15").do(job)
schedule.every().hour.at(":20").do(job)
schedule.every().hour.at(":25").do(job)
schedule.every().hour.at(":30").do(job)
schedule.every().hour.at(":35").do(job)
schedule.every().hour.at(":40").do(job)
schedule.every().hour.at(":45").do(job)
schedule.every().hour.at(":50").do(job)
schedule.every().hour.at(":55").do(job)

# Keep the script running indefinitely to execute scheduled jobs
while True:
    schedule.run_pending()
    time.sleep(0)