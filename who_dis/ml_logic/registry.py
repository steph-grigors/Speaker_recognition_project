import pandas as pd

def load_cleaned_df(csv_path):
    df_cleaned = pd.read_csv(csv_path)
    return df_cleaned
