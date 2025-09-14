import pandas as pd
import numpy as np

DATA_DIR = "/home/user/Genesis/data/YCB_0824"
df = pd.read_csv(f"{DATA_DIR}/train.csv")

unique_csv_paths = df["csv_path"].unique()
data_cache = {
    path: pd.read_csv(DATA_DIR + "/csv/" + path)
    for path in unique_csv_paths
}

for i, row in df.iterrows():
    force_df = data_cache[row['csv_path']]
    