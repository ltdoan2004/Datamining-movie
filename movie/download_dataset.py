import pandas as pd
import os
import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("podsyp/production-quality")

# Get a list of all files in the downloaded directory
files = os.listdir(path)

# Filter to find data_X.csv and data_Y.csv
dataframes = {}
for file in files:
    if file.endswith('.csv'):
        csv_path = os.path.join(path, file)
        df = pd.read_csv(csv_path, parse_dates=['date_time'])
        dataframes[file] = df

# Load the sensor data (minute-level) and the target data (quality data)
sensor_data = dataframes['data_X.csv']
target_data = dataframes['data_Y.csv']
sample = dataframes['sample_submission.csv']
# # Sort both datasets by date_time
# sensor_data = sensor_data.sort_values(by='date_time')
# target_data = target_data.sort_values(by='date_time')
# output_path = 'data_X.csv'
# sensor_data.to_csv(output_path, index=False)
output_path = 'sample_submission.csv'
sample.to_csv(output_path, index=False)


