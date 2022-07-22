# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

from utils import GOVS_INFO


# %%
pca = pd.read_csv(
    "./notebooks/temp/31---PCA___Mon_Jun_20_16-54-27_2022_9DBwLEK.csv", index_col=[0]
)
pca["gov_name"] = [GOVS_INFO["name"][gov_id] for gov_id in pca["gov_id"]]
pca


# %%
px.scatter_3d(pca, x="0", y="1", z="2", hover_data=["gov_id", "gov_name"])


# %%
