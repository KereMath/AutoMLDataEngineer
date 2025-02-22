import pandas as pd
import numpy as np

def introduce_missing_values(data, missing_rate=0.1):
    np.random.seed(42)
    data = data.copy()
    mask = np.random.rand(*data.shape) < missing_rate
    data[mask] = np.nan
    return data

if __name__ == "__main__":
    data = pd.read_csv('data/simulation_data.csv')
    fake_processed_data = introduce_missing_values(data, missing_rate=0.2)
    fake_processed_data.to_csv('data/fake_processed_data.csv', index=False)
