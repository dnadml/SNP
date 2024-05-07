import schedule
import time
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_script():
    try:
        # Running the model_test.py script
        subprocess.run(['python', 'model_train.py'], check=True)
        logging.info('Script executed successfully.')
    except subprocess.CalledProcessError as e:
        logging.error(f'Failed to execute script: {e}')

def idle_message():
    # Logs a message every minute when idle
    logging.info('Scheduler running...')

# Schedule the script to run 3 minutes before the bottom of each hour
schedule.every().hour.at(":57").do(run_script)

# Schedule an idle message every minute
schedule.every().minute.do(idle_message)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)
