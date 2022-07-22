# %%
import pandas as pd
import numpy as np


# %%
series = pd.Series([1, 1, 2, 3, 4, np.nan, np.nan])
series


# %%
sum(series.duplicated())


# %%
df = pd.read_csv("./temp/2018-12Wed_Jun__8_15-04-59_2022.csv", index_col=0)
df


# %%
dup_counts = df.apply(lambda series: sum(series.duplicated()) / len(series))
dup_counts


# %%
dup_counts.to_csv("./hi.csv")
