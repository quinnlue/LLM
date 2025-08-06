import pandas as pd

df = pd.read_parquet("validation.parquet")


seq = df.loc[0]['seq']

for i in seq:
    print(i)
