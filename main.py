from datetime import datetime
from pytz import timezone

from predict_prod import predict

def get_predictions(timestamp: int) -> None:
    ny_timezone = timezone('America/New_York')
    current_time_ny = datetime.now(ny_timezone)
    timestamp = current_time_ny.isoformat()  
    predictions = predict(timestamp)
    print(predictions)

if __name__ == "__main__":
    get_predictions(0)
