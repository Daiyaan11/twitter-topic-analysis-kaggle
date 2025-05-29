import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records.")
    return df
