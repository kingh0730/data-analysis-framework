# %%
import pandas as pd
import numpy as np


# %%
df = pd.read_csv(
    "./temp/13---DispersionCalculation___Thu_Jun_16_10-41-14_2022_I6jKfWo.csv",
    # index_col=0,
    names=["index_id", "dispersion"],
    header=0,
)
df


# %%
# df.index.astype(int)
new_df = df.apply(pd.to_numeric, errors="coerce").dropna()
new_df

# %%
