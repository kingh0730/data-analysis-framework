# %%
import pandas as pd
import numpy as np


# %%
df1 = pd.read_csv("./temp/2018-12Wed_Jun__8_15-04-59_2022.csv", index_col=0)
df1


# %%
df2 = pd.read_csv("./temp/2019-01Wed_Jun__8_15-06-47_2022.csv", index_col=0)
df2


# %%
(df1 == df1) | ((df1 != df1) & (df1 != df1))


# %%
(df2 == df2) | ((df2 != df2) & (df2 != df2))


# %%
(df1 == df2) | ((df1 != df1) & (df2 != df2))


# %%
