import pandas as pd


df = pd.read_csv("/home/mdxuser/sim/Genesis/data/train_simple.csv")
df = df.drop(columns=["analysis_result"])
df.to_csv("/home/mdxuser/sim/Genesis/data/train_simple.csv", index=False)