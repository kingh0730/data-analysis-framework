# %%
import pandas as pd
import numpy as np


# %%
df = pd.read_csv("./temp/2018-12Wed_Jun__8_15-04-59_2022.csv", index_col=0)
df


# %%
df.loc[df.count(1) >= df.shape[1] * 0.9, df.count(0) >= df.shape[0] * 0.5]


# %%
