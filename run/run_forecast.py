import json
import subprocess
from datetime import datetime, timedelta
import os
import logging

def create_forecast_job(start_date, end_date):
    # Load the JSON template
    with open('run_hawaii_gfsa_nested_2023_08_06_3km.json', 'r') as file:
        data = json.load(file)
    
    # Format start and end dates
    start_str = start_date.strftime("%Y-%m-%d_%H-%M-%S")
    end_str = end_date.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Update JSON content
    data["start_utc"] = start_str
    data["end_utc"] = end_str
    
    # Generate filename based on start and end dates
    unique_name = f"run_hawaii_gfsa_{start_str}"
    input_json = os.path.abspath(f'wrfxpy/run.jobs/{unique_name}.json')
    log_file = os.path.abspath(f'wrfxpy/run.logs/{unique_name}.log')
    
    # Save modified JSON file
    with open(input_json, 'w') as file:
        json.dump(data, file, indent=4)

    logging.info(f"run {unique_name} start")
    logging.info(f"input is {input_json}")
    logging.info(f"log file is {log_file}")
    
    # Run the forecast script
    with open(log_file, 'w') as logfile:
        subprocess.run(['./forecast.sh', input_json], cwd="./wrfxpy", stdout=logfile, stderr=logfile)

    logging.info(f"run {unique_name} end")

    return unique_name, output_json, log_file

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    # Example usage for a 10-day period (modify as needed)
    start_date = datetime(2011, 1, 1)
    end_date = start_date + timedelta(days=10)
    create_forecast_job(start_date, end_date)

