# %%
from scipy.stats import variation
import numpy as np
import pandas as pd


# %%
df = pd.read_csv("./temp/2018-12Wed_Jun__8_15-04-59_2022.csv", index_col=0)
df


# %%
variation(df["month"])


# %%
variation(df["gov_id"])


# %%
pd.Series(variation(df, nan_policy="omit"), index=df.columns)


# %%
print(df.quantile(3 / 4))
print(df.quantile(1 / 4))
print(df.quantile(3 / 4) - df.quantile(1 / 4))


# %%
df.std()


# %%
diff = df.quantile(3 / 4) - df.quantile(1 / 4)
sums = df.quantile(3 / 4) + df.quantile(1 / 4)

diff / sums


# %%
