# %%
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans


# %%
k_means = pd.read_csv(
    "./temp/67935---KMeans___Wed_Jul_13_16-18-02_2022.csv", index_col=0
).rename(columns={"0": "k_means"})
k_means


# %%
df = pd.read_csv("./temp/2022-03-districts.csv", index_col=0)
df


# %%
pops = df["65583"].dropna()
pops


# %%
reshaped = pops.to_numpy().reshape(-1, 1)
reshaped


# %%
k_means = KMeans(n_clusters=3).fit(reshaped)
labels = k_means.labels_
labels


# %%
pops_k_means = pd.DataFrame(labels, index=pops.index).rename(columns={0: "k_means"})
pops_k_means


# %%
concatenated = pd.concat([df, pops_k_means], axis=1)
concatenated


# %%
px.scatter(concatenated, x="65583", y="65583", hover_data=["gov_id"], color="k_means")


# %%
