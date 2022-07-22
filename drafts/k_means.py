# %%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


# %%
df = pd.read_csv("./temp/2022-03.csv", index_col=[0, 1, 2])
df


# %%
pops = df["65583"].dropna()
pops


# %%
reshaped = pops.to_numpy().reshape(-1, 1)
reshaped


# %%
k_means = KMeans(n_clusters=3).fit(reshaped)
k_means.cluster_centers_


# %%
labels = k_means.labels_
labels


# %%
pd.DataFrame(labels, index=pops.index)
