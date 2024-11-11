import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from run_forecast import create_forecast_job
import logging

# Define the time range
start_date = datetime(2011, 1, 1)
end_date = datetime(2024, 11, 1)
#end_date = datetime(2011, 1, 10) # testing
interval = timedelta(days=7)
spinup = timedelta(days=1)
max_concurrent_jobs = 30 

def generate_date_ranges(start, end, interval):
    current = start
    while current < end:
        yield current, min(current + interval + spinup, end)
        current += interval

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    # Using ThreadPoolExecutor to manage concurrent jobs
    with ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        futures = []
        
        # Generate jobs for each interval
        for start, end in generate_date_ranges(start_date, end_date, interval):
            futures.append(executor.submit(create_forecast_job, start, end))
        
        # Wait for all job1s to complete
        for future in as_completed(futures):
            unique_id, output_json, log_file = future.result()
            logging.ingo(f"Job {unique_id} completed. Job JSON: {output_json}, Log: {log_file}")

