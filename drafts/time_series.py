# %%
from os import listdir
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import pdist, squareform


# %%
files = listdir("./temp/data")
list_data_frames = []
for file_path in files:
    if file_path[:4] == "2019":
        data_frame = pd.read_csv(f"./temp/data/{file_path}", nrows=1, index_col=0)
        list_data_frames.append(data_frame)
concatenated = pd.concat(list_data_frames)
concatenated


# %%
normalized = (concatenated - concatenated.mean()) / concatenated.std()
normalized["month"] = concatenated["month"]
normalized


# %%
transposed = normalized.transpose()
transposed


# %%
transposed.rename(index={"month": "index_id"}, inplace=True)
transposed.columns = transposed.loc["index_id"]
nice = transposed.iloc[2:]
nice


# %%
px.scatter_3d(nice, x=217.0, y=218.0, z=219.0)


# %%
distances = pdist(nice.values, metric="euclidean")
dist_matrix = squareform(distances)
dist_matrix


# %%
pd.DataFrame(dist_matrix, index=nice.index, columns=nice.index)


# %%
np.sqrt(sum((nice.iloc[2] - nice.iloc[3]) ** 2))


# %%
