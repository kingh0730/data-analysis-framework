# %%
import pandas as pd
import numpy as np


# %%
df = pd.read_csv("./temp/2018-12Wed_Jun__8_15-04-59_2022.csv", index_col=0)
df


# %%
pops = df["65583"]
pops


# %%
divided_by_pop = df.apply(lambda s: s / df["65583"])


# %%
divided_by_pop.to_csv("./temp/2018-12_divided_by_pop_Wed_Jun__8_15-04-59_2022.csv")


# %%
