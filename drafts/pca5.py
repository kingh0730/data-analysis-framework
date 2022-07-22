# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

from utils import (
    GOVS_INFO,
    GovLevel,
    get_gov_level_ids,
    remove_outliers_series_with_median_deviation,
)


# %%
df = pd.read_csv("./drafts/temp/2022-03.csv", index_col=[0, 1, 2])
df


# %% Only districts
filtered = df[
    df.index.get_level_values("gov_id").isin(get_gov_level_ids(GovLevel.DISTRICT_LEVEL))
]
filtered


# %%
no_outliers = filtered.apply(
    lambda s: remove_outliers_series_with_median_deviation(s, 6)
)
no_outliers


# %%
scaled = (no_outliers - no_outliers.mean()) / no_outliers.std()
scaled


# %%
dropped_cols = scaled.loc[
    scaled.count(1) >= scaled.shape[1] * 0,
    scaled.count(0) >= scaled.shape[0] * 0.5,
]
dropped_cols


# %%
dropped_rows = dropped_cols.loc[
    dropped_cols.count(1) >= dropped_cols.shape[1] * 0.9,
    dropped_cols.count(0) >= dropped_cols.shape[0] * 0,
]
dropped_rows


# %%
filled_na = dropped_rows.fillna(dropped_rows.mean())
filled_na


# %%
dropped_na = filled_na


# %%
pca = PCA(n_components=3)
pca_result = pca.fit_transform(dropped_na)
pca_result


# %%
components = pd.DataFrame(pca.components_, columns=dropped_na.columns).T
components.sort_values(0, ascending=False)


# %%
new_df = pd.DataFrame(pca_result, index=dropped_na.index)
new_df["gov_name"] = [
    GOVS_INFO["name"][gov_id] for gov_id in new_df.index.get_level_values("gov_id")
]
new_df


# %%
px.scatter(new_df, x=0, y=1, hover_name="gov_name")


# %%
px.scatter_3d(new_df, x=0, y=1, z=2, hover_data=["gov_name"])


# %%
