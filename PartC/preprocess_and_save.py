import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_and_save(file_path, seq_length=24):
    # Load raw data in chunks to save memory
    chunk_iter = pd.read_csv(file_path, sep=';', na_values='?', chunksize=100000, low_memory=False)

    all_data = []

    for chunk_idx, chunk in enumerate(chunk_iter):
        print(f"Processing chunk {chunk_idx+1}")
        chunk = chunk.dropna(subset=['Global_active_power'])
        chunk['Global_active_power'] = chunk['Global_active_power'].interpolate(method='linear')
        all_data.append(chunk[['Global_active_power']].values)

    data = np.vstack(all_data)
    print(f"Total data length: {len(data)}")

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Save processed data and scaler
    np.save('scaled_power.npy', scaled_data)
    joblib.dump(scaler, 'scaler.save')

    print("Preprocessing complete. Data and scaler saved.")

if __name__ == "__main__":
    preprocess_and_save("household_power_consumption.txt")
# This script preprocesses the household power consumption data, normalises it, and saves the processed data and scaler for later use.