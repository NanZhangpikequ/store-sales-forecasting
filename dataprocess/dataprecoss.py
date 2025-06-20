import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "output")
output_path = os.path.join(script_dir, "..", "data", "output")
os.makedirs(output_path, exist_ok=True)

files_to_process = ['test_merged_final.csv', 'train_merged_final.csv']

for filename in files_to_process:
    file_path = os.path.join(data_path, filename)
    
    # Check file exists
    if not os.path.exists(file_path):
        print(f"Skip: {filename}")
        continue
    
    print(f"Processing: {filename}")
    
    # Load file
    df = pd.read_csv(file_path)
    
    # exchange to datetime type
    df['date'] = pd.to_datetime(df['date'])
    
    # date feature
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['quarter'] = df['date'].dt.quarter
    
    # sin/cos encoder
    # month (1-12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # day of week(0-6)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # save data
    output_filename = filename.replace('.csv', '_with_features.csv')
    output_file_path = os.path.join(output_path, output_filename)
    df.to_csv(output_file_path, index=False)
    
    print(f"Done: {output_filename}")