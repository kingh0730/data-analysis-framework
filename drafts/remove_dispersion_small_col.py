# %%
import pandas as pd
import numpy as np
from scipy.stats import variation

from utils import MeasureOfDispersion, calc_dispersions


# %%
df = pd.read_csv("./drafts/temp/2018-12Wed_Jun__8_15-04-59_2022.csv", index_col=0)
df


# %%
df.apply(lambda series: series if series.name in ["gov_id", "month"] else None)


# %%
df.loc[:, [col in ["gov_id", "month"] for col in df.columns]]


# %%
df.loc[:, [col in ["gov_id", "month"] for col in df.columns]]


# %%
dispersions = np.abs(
    pd.Series(variation(df, nan_policy="omit", ddof=1), index=df.columns)
)
dispersions


# %%
dispersions = calc_dispersions(df, MeasureOfDispersion.CV2)
dispersions


# %%
output_df = df.loc[:, [dispersions[col] > 0.5 for col in df.columns]]
output_df


# %%
dispersions["gov_id"]
