# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

from utils import GOVS_INFO, GovLevel, get_gov_level_ids


# %%
df = pd.read_csv("./drafts/temp/2022-03.csv", index_col=[0, 1, 2])
df


# %% Only districts
filtered = df[
    df.index.get_level_values("gov_id").isin(get_gov_level_ids(GovLevel.DISTRICT_LEVEL))
]
filtered


# %%
scaled = (filtered - filtered.mean()) / filtered.std()
scaled


# %%
dropped_na = scaled.dropna(axis=1)
dropped_na


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
