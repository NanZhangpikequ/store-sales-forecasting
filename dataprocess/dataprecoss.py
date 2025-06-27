import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

data_path = '/kaggle/working/output'
output_path = '/kaggle/working/output'
os.makedirs(output_path, exist_ok=True)

files_to_process = ['train_merged_final.csv','test_merged_final.csv']

store_nbr_encoder = None
family_encoder = None
onpromotion_scaler = None
dcoilwtico_scaler = None
city_encoder = None
state_encoder = None
cluster_encoder = None

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
    
# == date feature ==
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

# == store_nbr label encoding ==
    if store_nbr_encoder is None:
        store_nbr_encoder = LabelEncoder()
        df['store_nbr_encoded'] = store_nbr_encoder.fit_transform(df['store_nbr'])
    else:
        df['store_nbr_encoded'] = store_nbr_encoder.transform(df['store_nbr'])

# == family label encoding ==
    if family_encoder is None:
        family_encoder = LabelEncoder()
        df['family_encoded'] = family_encoder.fit_transform(df['family'])
    else:
        df['family_encoded'] = family_encoder.transform(df['family'])

# == onpromotion log transform & z-score ==
    df['onpromotion_log'] = np.log1p(df['onpromotion'])
    if onpromotion_scaler is None:
        onpromotion_scaler = StandardScaler()
        df['onpromotion_scaled'] = onpromotion_scaler.fit_transform(df['onpromotion_log'].values.reshape(-1,1)).flatten()
    else:
        df['onpromotion_scaled'] = onpromotion_scaler.transform(df['onpromotion_log'].values.reshape(-1,1)).flatten()

# == dcoilwtico z-score ==
    if dcoilwtico_scaler is None:
        dcoilwtico_scaler = StandardScaler()
        df['dcoilwtico_scaled'] = dcoilwtico_scaler.fit_transform(df['dcoilwtico'].values.reshape(-1,1)).flatten()
    else:
        df['dcoilwtico_scaled'] = dcoilwtico_scaler.transform(df['dcoilwtico'].values.reshape(-1,1)).flatten()

# == city label encoding ==
    if city_encoder is None:
        city_encoder = LabelEncoder()
        df['city_encoded'] = city_encoder.fit_transform(df['city'])
    else:
        df['city_encoded'] = city_encoder.transform(df['city'])

# == state label encoding ==
    if state_encoder is None:
        state_encoder = LabelEncoder()
        df['state_encoded'] = state_encoder.fit_transform(df['state'])
    else:
        df['state_encoded'] = state_encoder.transform(df['state'])

# == cluster label encoding ==
    if cluster_encoder is None:
        cluster_encoder = LabelEncoder()
        df['cluster_encoded'] = cluster_encoder.fit_transform(df['cluster'])
    else:
        df['cluster_encoded'] = cluster_encoder.transform(df['cluster'])

    # save data
    output_filename = filename.replace('.csv', '_with_features.csv')
    output_file_path = os.path.join(output_path, output_filename)
    df.to_csv(output_file_path, index=False)
    
    print(f"Done: {output_filename}")
